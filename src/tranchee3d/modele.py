"""Entraînement du U-Net (PyTorch, MPS sur Mac) et export ONNX.

Perte = entropie croisée + Dice + perte de distance (« boundary loss »).
La perte de distance multiplie la probabilité prédite de chaque classe d'objet par
la distance signée au tracé d'entraînement : prédire un câble loin du tracé coûte
d'autant plus cher qu'on en est loin.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .config import CLASSES, PONCTUELS, Config

N_OBJETS = len(CLASSES) - 1


def choisir_peripherique():
    """MPS (GPU Apple) si disponible, sinon CUDA, sinon CPU. L'inférence, elle, est toujours sur CPU (ONNX)."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def creer_modele(cfg: Config):
    """U-Net smp : encodeur `cfg.encodeur` pré-entraîné ImageNet, 4 canaux (RVB + hauteur relative), 5 classes."""
    import segmentation_models_pytorch as smp
    return smp.Unet(cfg.encodeur, encoder_weights=cfg.poids_encodeur or None,
                    in_channels=4, classes=len(CLASSES))


def distances_signees(masque: np.ndarray, pixel: float, plafond: float) -> np.ndarray:
    """(N_OBJETS, H, W) float16, en mètres : > 0 hors de l'objet, < 0 dedans, bornée à ± plafond.
    Une classe absente du chantier vaut +plafond partout."""
    d = np.full((N_OBJETS, *masque.shape), plafond, np.float16)
    for c in range(1, len(CLASSES)):
        objet = masque == c
        if objet.any():
            dehors = ndi.distance_transform_edt(~objet) * pixel
            dedans = ndi.distance_transform_edt(objet) * pixel
            d[c - 1] = np.clip(dehors - dedans, -plafond, plafond)
    return d


class Chantier:
    """Chantier préparé (fichier .npz produit par la commande `preparer`)."""

    def __init__(self, chemin, cfg: Config):
        d = np.load(chemin)
        self.nom = Path(chemin).parent.name
        self.entree = d["entree"]                        # (4, H, W) float16 : converti tuile par tuile (mémoire ÷ 2)
        self.masque = d["masque"]                        # (H, W) uint8
        self.zone = d["zone"]                            # (H, W) bool
        self.plafond = cfg.plafond_distance
        self.distance = distances_signees(self.masque, cfg.pixel, self.plafond)
        pas = 7                                          # sous-échantillonnage des centres possibles
        self.px_objet = np.argwhere(self.masque[::pas, ::pas] > 0) * pas
        self.px_zone = np.argwhere(self.zone[::pas, ::pas]) * pas
        if len(self.px_zone) == 0:
            self.px_zone = np.argwhere(np.ones_like(self.zone[::pas, ::pas])) * pas

    def tuile(self, centre, t):
        """Tuile t x t centrée : entrée, étiquettes, distances (hors image : fond, loin de tout)."""
        H, W = self.masque.shape
        l0 = int(centre[0]) - t // 2
        c0 = int(centre[1]) - t // 2
        x = np.zeros((4, t, t), np.float32)
        y = np.zeros((t, t), np.int64)
        d = np.full((N_OBJETS, t, t), self.plafond, np.float32)
        la, lb = max(l0, 0), min(l0 + t, H)
        ca, cb = max(c0, 0), min(c0 + t, W)
        if lb > la and cb > ca:
            ici = (slice(la - l0, lb - l0), slice(ca - c0, cb - c0))
            x[:, ici[0], ici[1]] = self.entree[:, la:lb, ca:cb]
            y[ici] = self.masque[la:lb, ca:cb]
            d[:, ici[0], ici[1]] = self.distance[:, la:lb, ca:cb]
        return x, y, d


def _redimensionner(x, y, d, t):
    """Tuile ramenée à t x t : bilinéaire pour l'entrée et les distances, plus proche voisin pour les classes."""
    import torch
    import torch.nn.functional as F
    xd = F.interpolate(torch.from_numpy(np.concatenate([x, d])[None]), size=(t, t), mode="bilinear",
                       align_corners=False)[0].numpy()
    yt = F.interpolate(torch.from_numpy(y[None, None].astype(np.float32)), size=(t, t), mode="nearest")[0, 0]
    return xd[:4], yt.numpy().astype(np.int64), xd[4:]


def _varier_fourreau(x, y, cfg: Config, rng) -> None:
    """Change légèrement la couleur des fourreaux (teinte, saturation) : rouges plus orangés,
    rosés ou délavés selon le fabricant, l'éclairage et la boue. Seuls les pixels rougeâtres
    sous l'étiquette fourreau sont modifiés (le tampon d'étiquette déborde sur la terre)."""
    from skimage.color import hsv2rgb, rgb2hsv
    if cfg.teinte_fourreau <= 0 and cfg.saturation_fourreau <= 0:
        return
    r, v, b = x[0], x[1], x[2]
    m = (y == CLASSES.index("fourreau_rouge")) & (r > v) & (r > b)
    if not m.any():
        return
    hsv = rgb2hsv(x[:3, m].T[:, None, :])
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.teinte_fourreau, cfg.teinte_fourreau)) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - cfg.saturation_fourreau, 1 + cfg.saturation_fourreau), 0, 1)
    x[:3, m] = hsv2rgb(hsv)[:, 0, :].T


def _lot_aleatoire(chantiers, cfg: Config, rng):
    xs, ys, ds = [], [], []
    t = cfg.tuile
    for _ in range(cfg.lot):
        ch = chantiers[rng.integers(len(chantiers))]
        sur_objet = len(ch.px_objet) and rng.random() < cfg.part_tuiles_objet
        source = ch.px_objet if sur_objet else ch.px_zone
        centre = source[rng.integers(len(source))] + rng.integers(-t // 3, t // 3 + 1, 2)
        # changement d'échelle : tuile plus grande ou plus petite, ramenée à t x t
        ts = int(round(t / rng.uniform(1 - cfg.echelle, 1 + cfg.echelle))) if cfg.echelle > 0 else t
        x, y, d = ch.tuile(centre, ts)
        if ts != t:
            x, y, d = _redimensionner(x, y, d, t)
        # augmentations géométriques (identiques pour l'entrée, les étiquettes et les distances)
        k = int(rng.integers(4))
        x, y, d = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(d, k, (1, 2))
        if rng.random() < 0.5:
            x, y, d = x[:, :, ::-1], y[:, ::-1], d[:, :, ::-1]
        # augmentations radiométriques (RVB seulement) : luminosité, contraste, dominante
        x = x.copy()
        vide = ~x.any(axis=0)      # pixels sans point : restent à 0 comme à l'inférence
        _varier_fourreau(x, y, cfg, rng)
        gain = rng.uniform(0.75, 1.25) * rng.uniform(0.93, 1.07, (3, 1, 1))
        x[:3] = np.clip((x[:3] - 0.5) * rng.uniform(0.8, 1.2) + 0.5, 0, 1) * gain
        x[:3] = np.clip(x[:3] + rng.uniform(-0.08, 0.08), 0, 1)
        y, d = y.copy(), d.copy()
        # bruit capteur : couleurs, hauteur relative (canal normalisé par 1,5 m) et erreur sur le sol
        if cfg.bruit_rvb > 0:
            x[:3] = np.clip(x[:3] + rng.normal(0, cfg.bruit_rvb, x[:3].shape), 0, 1)
        x[3] += (rng.normal(0, cfg.bruit_hauteur, x[3].shape) + rng.uniform(-1, 1) * cfg.decalage_hauteur) / 1.5
        x[:, vide] = 0
        # trou : pixels vides comme dans un nuage incomplet ; l'objet n'y est pas visible, donc pas étiqueté
        if rng.random() < cfg.proba_trou:
            h, w = (rng.uniform(0.05, cfg.taille_trou, 2) * t).astype(int) + 1
            l0, c0 = rng.integers(0, t - h + 1), rng.integers(0, t - w + 1)
            x[:, l0:l0 + h, c0:c0 + w] = 0
            y[l0:l0 + h, c0:c0 + w] = 0
            d[:, l0:l0 + h, c0:c0 + w] = cfg.plafond_distance
        xs.append(x)
        ys.append(y)
        ds.append(d)
    return np.stack(xs).astype(np.float32), np.stack(ys), np.stack(ds)


def perte_distance(logits, y, d, cfg: Config):
    """Boundary loss : somme(probabilité x distance signée normalisée) sur les classes
    d'objet, rapportée à la surface d'objet présente dans le lot. Vaut environ +1 si tout
    est prédit au plafond de distance ; négative quand les prédictions tombent dans les objets."""
    phi = d / cfg.plafond_distance
    if cfg.exposant_distance != 1.0:
        phi = phi.sign() * phi.abs() ** cfg.exposant_distance
    if not cfg.distance_sur_ponctuels:
        phi = phi.clone()
        phi[:, [c - 1 for c in PONCTUELS]] = 0      # boîtes et coffrets : CE + Dice seulement
    proba = logits.softmax(dim=1)[:, 1:]
    surface_objet = (y > 0).sum() + 1e-3 * y.numel()
    return (proba * phi).sum() / surface_objet


def poids_distance_courant(pas_fait: int, total: int, cfg: Config) -> float:
    """0 pendant l'attente (le modèle apprend d'abord toutes les classes avec CE + Dice),
    puis montée linéaire jusqu'à `poids_distance`. Appliquée dès le départ, cette perte
    pousse le modèle à ne plus rien prédire pour les classes rares."""
    debut = cfg.attente_distance * total
    avancement = (pas_fait - debut) / max(1.0, cfg.montee_distance * total)
    return cfg.poids_distance * float(np.clip(avancement, 0.0, 1.0))


def _tuiles_validation(ch: Chantier, t):
    H, W = ch.masque.shape
    for l in range(t // 2, H + t // 2, t):
        for c in range(t // 2, W + t // 2, t):
            if ch.zone[max(l - t // 2, 0): l + t // 2, max(c - t // 2, 0): c + t // 2].any():
                yield ch.tuile((l, c), t)


def valider(modele, chantiers, cfg: Config, periph):
    """Retourne (IoU par classe, écart moyen au tracé par classe d'objet, en mètres).

    L'écart est symétrique : moyenne de la distance des pixels prédits au tracé (fausses
    détections) et de la distance des pixels du tracé à la prédiction (manques). Il est
    borné au plafond, et mesuré tuile par tuile."""
    import torch
    n = len(CLASSES)
    plafond = cfg.plafond_distance
    confusion = np.zeros((n, n), np.int64)
    somme = np.zeros((2, N_OBJETS))      # [prédit -> tracé, tracé -> prédit]
    compte = np.zeros((2, N_OBJETS))
    modele.eval()
    with torch.no_grad():
        for ch in chantiers:
            for x, y, d in _tuiles_validation(ch, cfg.tuile):
                pred = modele(torch.from_numpy(x[None]).to(periph)).argmax(1)[0].cpu().numpy()
                confusion += np.bincount(y.ravel() * n + pred.ravel(), minlength=n * n).reshape(n, n)
                for c in range(1, n):
                    p, v = pred == c, y == c
                    if p.any():
                        somme[0, c - 1] += np.clip(d[c - 1][p], 0, None).sum()
                        compte[0, c - 1] += p.sum()
                    if v.any():
                        vers_pred = (np.minimum(ndi.distance_transform_edt(~p) * cfg.pixel, plafond)
                                     if p.any() else np.full(p.shape, plafond))
                        somme[1, c - 1] += vers_pred[v].sum()
                        compte[1, c - 1] += v.sum()
    inter = np.diag(confusion).astype(float)
    union = confusion.sum(0) + confusion.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    # une classe présente dans le tracé mais jamais prédite compte au plafond
    aller = np.where(compte[0] > 0, somme[0] / np.maximum(compte[0], 1), plafond)
    retour = np.where(compte[1] > 0, somme[1] / np.maximum(compte[1], 1), np.nan)
    ecart = np.where(compte[1] > 0, (aller + retour) / 2, np.nan)
    return iou, ecart


def entrainer(fichiers_train, fichiers_val, sortie, cfg: Config, graine=0, reprendre=None, fichiers_test=()):
    """Entraîne, garde les meilleurs poids selon `cfg.critere` et exporte en ONNX."""
    import segmentation_models_pytorch as smp
    import torch

    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(graine)
    torch.manual_seed(graine)
    periph = choisir_peripherique()
    print(f"Périphérique d'entraînement : {periph}")

    train = [Chantier(f, cfg) for f in fichiers_train]
    val = [Chantier(f, cfg) for f in fichiers_val]
    test = [Chantier(f, cfg) for f in fichiers_test]
    print(f"Chantiers : {len(train)} en entraînement, {len(val)} en validation, {len(test)} en test")
    print(f"Perte de distance : poids {cfg.poids_distance}, plafond {cfg.plafond_distance} m ; "
          f"critère de sélection : {cfg.critere}")

    modele = creer_modele(cfg)
    if reprendre:
        modele.load_state_dict(torch.load(reprendre, map_location="cpu"))
        print(f"Reprise depuis {reprendre}")
    modele = modele.to(periph)
    opt = torch.optim.AdamW(modele.parameters(), lr=cfg.taux_apprentissage, weight_decay=1e-4)
    total = cfg.epochs * cfg.iterations_par_epoch
    plan = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.taux_apprentissage, total_steps=total)
    ce = torch.nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss(mode="multiclass")

    meilleur, chemin_poids, pas_fait = None, sortie / "meilleur.pt", 0
    # historique lisible par l'atelier : une ligne JSON par epoch
    historique = open(sortie / "historique.jsonl", "w", encoding="utf-8")
    ecrire = lambda **e: (historique.write(json.dumps(e, ensure_ascii=False) + "\n"), historique.flush())
    nan_vers_none = lambda a: [None if np.isnan(v) else round(float(v), 5) for v in a]
    ecrire(type="debut", parametres=asdict(cfg), train=[c.nom for c in train], validation=[c.nom for c in val], test=[c.nom for c in test],
           reprendre=str(reprendre) if reprendre else None, peripherique=str(periph), t=time.time())
    for epoch in range(1, cfg.epochs + 1):
        debut_epoch = time.time()
        modele.train()
        cumul = cumul_dist = 0.0
        for _ in range(cfg.iterations_par_epoch):
            x, y, d = _lot_aleatoire(train, cfg, rng)
            x, y, d = (torch.from_numpy(a).to(periph) for a in (x, y, d))
            logits = modele(x)
            perte = ce(logits, y) + dice(logits, y)
            if cfg.poids_distance > 0:
                dist = perte_distance(logits, y, d, cfg)
                perte = perte + poids_distance_courant(pas_fait, total, cfg) * dist
                cumul_dist += dist.item()
            opt.zero_grad()
            perte.backward()
            opt.step()
            plan.step()
            pas_fait += 1
            cumul += perte.item()

        n_it = cfg.iterations_par_epoch
        if val:
            iou, ecart = valider(modele, val, cfg, periph)
            ecart_moyen = float(np.nanmean(ecart))
            iou_moyen = float(np.nanmean(iou[1:]))
            print(f"epoch {epoch:3d}  perte={cumul / n_it:.3f} (distance {cumul_dist / n_it:+.3f})  "
                  f"IoU={iou_moyen:.3f}  écart moyen={100 * ecart_moyen:.2f} cm  ["
                  + "  ".join(f"{c}: {100 * e:.2f}" for c, e in zip(CLASSES[1:], ecart)) + "]")
            score = -ecart_moyen if cfg.critere == "distance" else iou_moyen
            retenu = meilleur is None or score > meilleur
        else:
            # modèle final sur tous les chantiers : aucune mesure indépendante, on garde la dernière epoch
            iou, ecart = np.full(len(CLASSES), np.nan), np.full(N_OBJETS, np.nan)
            ecart_moyen = iou_moyen = float("nan")
            print(f"epoch {epoch:3d}  perte={cumul / n_it:.3f} (distance {cumul_dist / n_it:+.3f})  (sans validation)")
            score, retenu = epoch, True
        if retenu:
            meilleur, resume = score, (epoch, iou_moyen, ecart_moyen)
            torch.save(modele.state_dict(), chemin_poids)
        ecrire(type="epoch", epoch=epoch, epochs=cfg.epochs, perte=cumul / n_it, distance=cumul_dist / n_it,
               iou_moyen=nan_vers_none([iou_moyen])[0], ecart_moyen=nan_vers_none([ecart_moyen])[0], iou=dict(zip(CLASSES[1:], nan_vers_none(iou[1:]))),
               ecart=dict(zip(CLASSES[1:], nan_vers_none(ecart))), retenu=retenu,
               duree=round(time.time() - debut_epoch, 1))

    modele.load_state_dict(torch.load(chemin_poids, map_location="cpu"))
    mesure_test = None
    if test:   # mesure unique, sur des chantiers qui n'ont servi ni à apprendre ni à choisir l'epoch
        iou, ecart = valider(modele, test, cfg, periph)
        mesure_test = {"iou_moyen": nan_vers_none([np.nanmean(iou[1:])])[0], "ecart_moyen": nan_vers_none([np.nanmean(ecart)])[0],
                       "iou": dict(zip(CLASSES[1:], nan_vers_none(iou[1:]))), "ecart": dict(zip(CLASSES[1:], nan_vers_none(ecart)))}
        print(f"Test ({', '.join(c.nom for c in test)}) : IoU {mesure_test['iou_moyen']}, "
              f"écart par classe (m) {mesure_test['ecart']}")
    chemin_onnx = exporter_onnx(modele.cpu(), sortie / "modele.onnx", cfg)
    print(f"Modèle retenu : epoch {resume[0]}, IoU {resume[1]:.3f}, "
          f"écart moyen {100 * resume[2]:.2f} cm  ->  {chemin_onnx}")
    ecrire(type="fin", test=mesure_test, epoch=resume[0], iou_moyen=nan_vers_none([resume[1]])[0],
           ecart_moyen=nan_vers_none([resume[2]])[0], onnx=str(chemin_onnx))
    historique.close()
    return chemin_onnx


def exporter_onnx(modele, chemin, cfg: Config):
    import torch
    modele.eval()
    exemple = torch.zeros(1, 4, cfg.tuile, cfg.tuile)
    torch.onnx.export(
        modele, exemple, str(chemin), input_names=["entree"], output_names=["logits"],
        dynamic_axes={"entree": {0: "lot"}, "logits": {0: "lot"}}, opset_version=17, dynamo=False,
    )
    # prétraitement attendu par ces poids : l'inférence avertit si sa config diffère
    Path(chemin).with_suffix(".json").write_text(json.dumps(
        {"pixel": cfg.pixel, "tuile": cfg.tuile, "encodeur": cfg.encodeur, "classes": CLASSES,
         "fenetre_sol": cfg.fenetre_sol, "centile_sol": cfg.centile_sol},
        indent=1), encoding="utf-8")
    return chemin
