"""Atelier de débogage : exécution de la chaîne étape par étape sur un nuage.

Chaque étape écrit un fichier dans <sortie>/<chantier>/ (PNG pour les rasters, binaire pour les
points, JSON pour les tracés) que la page `assets/atelier.html` affiche dans un onglet, et que l'on
peut rouvrir dans QGIS / CloudCompare. Le serveur HTTP est dans `serveur.py`, la gestion des
entraînements dans `entrainements.py`.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASSES, Config

# ordre des onglets de la page ; à partir de "detection", un modèle ONNX est nécessaire
ETAPES = ["nuage", "ortho", "tranchee", "detection", "nettoyage", "traces", "reseaux", "sous_masque", "ouvrages",
          "resultat"]
AVEC_MODELE = ETAPES.index("detection")
# couleur d'affichage des points par classe (index = classe), et rampe type viridis pour les rasters continus
COULEURS_CLASSES = np.array([(0, 0, 0), (55, 182, 255), (255, 77, 94), (255, 201, 60), (61, 220, 151)], np.uint8)
RAMPE = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], float)
LARGEUR_PNG_MAX = 8000          # les PNG d'étape sont réduits au-delà : la page n'a pas besoin de plus
COULEUR_FOND_TRANCHEE = np.array([255, 140, 0])   # orange en surimpression sur le fond de tranchée
GRIS_POINTS_CACHES = 200        # points écartés (sous la surface) affichés en gris


# ---------------------------------------------------------------- écriture des fichiers d'étape

def _rampe(a: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Raster continu -> image RVB selon RAMPE entre vmin et vmax ; NaN en noir."""
    t = np.clip((np.nan_to_num(a, nan=vmin) - vmin) / max(vmax - vmin, 1e-9), 0, 1) * (len(RAMPE) - 1)
    rvb = np.stack([np.interp(t, np.arange(len(RAMPE)), RAMPE[:, k]) for k in range(3)], -1)
    rvb[np.isnan(a)] = 0
    return rvb.astype(np.uint8)


def _png(chemin: Path, rvb: np.ndarray, largeur_max: int = LARGEUR_PNG_MAX) -> None:
    """Enregistre une image RVB, réduite si son grand côté dépasse largeur_max."""
    im = Image.fromarray(rvb)
    if max(im.size) > largeur_max:
        e = largeur_max / max(im.size)
        im = im.resize((max(1, int(im.width * e)), max(1, int(im.height * e))), Image.BILINEAR)
    im.save(chemin)


def _points(chemin: Path, xyz: np.ndarray, rvb: np.ndarray, centre, max_points: int, graine=0) -> int:
    """Nuage pour la page : float32 xyz centrés puis uint8 rvb, sous-échantillonné au hasard au-delà
    de max_points (le navigateur ne tient pas des millions de points). Retourne le nombre écrit."""
    if len(xyz) > max_points:
        choix = np.sort(np.random.default_rng(graine).choice(len(xyz), max_points, replace=False))
        xyz, rvb = xyz[choix], rvb[choix]
    with open(chemin, "wb") as f:
        f.write((xyz - centre).astype(np.float32).tobytes())
        f.write(np.ascontiguousarray(rvb, np.uint8).tobytes())
    return len(xyz)


def _troncons(troncons, centre) -> list[dict]:
    """Tronçons en JSON, coordonnées centrées comme le nuage affiché."""
    return [{"classe": CLASSES[t.classe], "origine": t.origine, "id_cable": t.id_cable,
             "longueur": round(t.longueur, 3), "confiance": round(t.confiance, 3),
             "pts": (t.pts - centre).round(4).tolist()} for t in troncons]


def _ponctuels(ponctuels, centre) -> list[dict]:
    """Boîtes et coffrets en JSON, coordonnées centrées."""
    return [{"classe": CLASSES[p.classe], "xyz": (p.xyz - centre).round(4).tolist(),
             "confiance": round(p.confiance, 3)} for p in ponctuels]


def _verite_json(chemin, decalage) -> dict:
    """GeoJSON de vérité -> lignes et points 3D centrés (les géométries sans Z sont ignorées)."""
    from shapely.geometry import shape
    lignes, points = [], []
    for f in json.loads(Path(chemin).read_text(encoding="utf-8"))["features"]:
        g = shape(f["geometry"])
        if not g.has_z:
            continue
        for p in getattr(g, "geoms", [g]):
            c = (np.array(p.coords) - decalage).round(4).tolist()
            (points.append(c[0]) if p.geom_type == "Point" else lignes.append(c))
    return {"lignes": lignes, "points": points}


# ---------------------------------------------------------------- exécution

class Atelier:
    """Exécute un nuage étape par étape (ETAPES) dans un thread, une exécution à la fois.

    `etat` est le dictionnaire que la page interroge : statut, message et durée de chaque étape,
    plus `infos` (tracés, vérité, évaluation) ; il est remplacé à chaque lancement.
    """

    def __init__(self, donnees: Path, modeles: Path, sortie: Path, cfg: Config, max_points: int):
        self.donnees, self.modeles, self.sortie, self.cfg = donnees, modeles, sortie, cfg
        self.max_points = max_points
        self.verrou = threading.Lock()
        self.etat = {"execution": 0, "en_cours": False, "dossier": None, "etapes": {}, "infos": {}}

    # ------------------------------------------------------------- inventaire
    @property
    def televersements(self) -> Path:
        return self.sortie / "televersements"

    def liste(self) -> dict:
        """Nuages disponibles (donnees/*/ et téléversés) et modèles ONNX."""
        nuages = sorted(p for ext in ("*.laz", "*.las") for p in self.donnees.glob(f"*/{ext}"))
        nuages += sorted(p for ext in ("*.laz", "*.las") for p in self.televersements.glob(ext))
        return {"nuages": [{"chemin": str(p), "nom": f"{p.parent.name}/{p.name}",
                            "verite": (p.parent / "verite.geojson").exists()} for p in nuages],
                "modeles": [str(p) for p in sorted(self.modeles.glob("**/*.onnx"))]}

    def televerser(self, nom: str, flux, taille: int) -> str:
        """Enregistre un .las/.laz envoyé par le navigateur. Le nom est réduit à son dernier segment
        (pas de chemin), et l'on ne lit que la taille annoncée."""
        nom = Path(nom).name
        if Path(nom).suffix.lower() not in (".las", ".laz"):
            raise ValueError("Seuls les fichiers .las et .laz sont acceptés.")
        self.televersements.mkdir(parents=True, exist_ok=True)
        with open(self.televersements / nom, "wb") as f:
            reste = taille
            while reste > 0:
                bloc = flux.read(min(reste, 1 << 20))
                if not bloc:
                    raise ValueError("Téléversement interrompu.")
                f.write(bloc)
                reste -= len(bloc)
        return str(self.televersements / nom)

    # ------------------------------------------------------------- exécution
    def lancer(self, nuage: str, modele: str | None, ortho: str | None) -> None:
        """Démarre l'exécution dans un thread. Sans modèle, les étapes à partir de la détection sont
        marquées `sans_modele` et l'exécution s'arrête avant."""
        inventaire = self.liste()
        if nuage not in {n["chemin"] for n in inventaire["nuages"]}:
            raise ValueError("Nuage inconnu.")
        if modele and modele not in inventaire["modeles"]:
            raise ValueError("Modèle inconnu.")
        with self.verrou:
            if self.etat["en_cours"]:
                raise ValueError("Une exécution est déjà en cours.")
            # dossier de sortie : nom du chantier, ou nom du fichier pour un nuage téléversé
            nom = Path(nuage).stem if Path(nuage).parent == self.televersements else Path(nuage).parent.name
            dossier = self.sortie / nom
            dossier.mkdir(parents=True, exist_ok=True)
            derniere = len(ETAPES) if modele else AVEC_MODELE
            self.etat = {"execution": self.etat["execution"] + 1, "en_cours": True, "dossier": nom,
                         "etapes": {e: {"statut": "attente" if i < derniere else "sans_modele"}
                                    for i, e in enumerate(ETAPES)}, "infos": {}}
        threading.Thread(target=self._executer, args=(Path(nuage), modele, ortho, dossier),
                         daemon=True).start()

    def _executer(self, chemin_nuage: Path, modele, ortho, dossier: Path) -> None:
        """Corps du thread : enchaîne les étapes, s'arrête à la première erreur (statut `erreur`)."""
        # imports ici : la chaîne (rasterio, onnxruntime, shapely…) n'est chargée qu'à la première exécution
        from .evaluation import evaluer
        from .export import ecrire_geojson, image_controle, vers_geojson
        from .facade import ecrire_faces, orthos_aux_extremites
        from .inference import predire
        from .nuage import lire_nuage
        from .ortho import appliquer_ortho_fournie, generer_rasters
        from .preparation import orthos_de
        from .raccordement import raccorder, raccorder_ouvrages
        from .reseau_existant import par_classe, reseau_existant
        from .retour3d import classer_points, extraire
        from .tranchee import detecter_tranchee, tenseur_entree

        cfg, infos = self.cfg, self.etat["infos"]
        verite = chemin_nuage.parent / "verite.geojson"
        verite = verite if verite.exists() else None
        # ortho fournie à côté du nuage : prioritaire sur l'ortho générée, comme à la préparation
        orthos = [ortho] if isinstance(ortho, str) else (ortho or orthos_de(chemin_nuage.parent))

        def etape(nom, f):
            """Exécute une étape en tenant l'état à jour pour la page (statut, message, durée)."""
            e = self.etat["etapes"][nom]
            e["statut"], t0 = "en_cours", time.time()
            try:
                e["message"] = f()
                e["statut"] = "ok"
            except Exception as err:
                e["statut"], e["message"] = "erreur", f"{type(err).__name__} : {err}"
                traceback.print_exc()
                raise
            finally:
                e["duree"] = round(time.time() - t0, 2)

        s = {}   # objets partagés entre étapes (nuage, rasters, tranchée, classes, tronçons…)

        def f_nuage():
            s["nuage"] = n = lire_nuage(chemin_nuage)
            s["centre"] = c = (n.xyz.min(axis=0) + n.xyz.max(axis=0)) / 2   # la page travaille centrée sur 0
            affiches = _points(dossier / "nuage.bin", n.xyz, n.rvb, c, self.max_points)
            infos["decalage"] = (n.origine + c).tolist()
            infos["verite"] = _verite_json(verite, n.origine + c) if verite else None
            return f"{len(n):,} points ; {affiches:,} affichés".replace(",", " ")

        def f_ortho():
            s["r"] = r = generer_rasters(s["nuage"], cfg)
            for o in orthos:
                appliquer_ortho_fournie(r, o, s["nuage"].origine)
            _png(dossier / "ortho.png", r.rvb)
            h = r.hauteur
            _png(dossier / "hauteur.png", _rampe(h, *np.nanpercentile(h, [1, 99])))
            vide = 100 * np.isnan(h).mean()
            return f"{r.forme[1]} x {r.forme[0]} px de {100 * r.pixel:.1f} cm ({r.source_ortho}), {vide:.0f} % vides"

        def f_tranchee():
            s["t"] = t = detecter_tranchee(s["r"], cfg)
            image_controle(dossier / "tranchee.png", s["r"].rvb, zone=t.zone, largeur_max=LARGEUR_PNG_MAX)
            fond = s["r"].rvb.copy()
            fond[t.fond] = (0.4 * fond[t.fond] + 0.6 * COULEUR_FOND_TRANCHEE).astype(np.uint8)
            _png(dossier / "fond.png", fond)
            _png(dossier / "hauteur_relative.png", _rampe(t.hauteur_relative, -1.2, 0.3))
            return f"zone d'intérêt {100 * t.zone.mean():.0f} %, fond de tranchée {100 * t.fond.mean():.0f} % de l'image"

        def f_detection():
            s["classes"], s["confiance"] = c, p = predire(tenseur_entree(s["r"], s["t"]), s["t"].zone, modele, cfg)
            image_controle(dossier / "detection.png", s["r"].rvb, classes=c, zone=s["t"].zone, largeur_max=LARGEUR_PNG_MAX)
            _png(dossier / "confiance.png", _rampe(np.where(s["t"].zone, p, np.nan), 0.3, 1.0))
            px = {CLASSES[k]: int((c == k).sum()) for k in range(1, len(CLASSES))}
            return "pixels : " + ", ".join(f"{k} {v}" for k, v in px.items())

        def f_nettoyage():
            n = s["nuage"]
            lig, col, surface, classe_pt = classer_points(s["classes"], s["r"], n, cfg)
            garde = classe_pt > 0                                 # points de surface sous un masque
            caches = ~surface & (s["classes"][lig, col] > 0)      # sous un masque mais sous la surface : écartés
            _points(dossier / "classes.bin", n.xyz[garde], COULEURS_CLASSES[classe_pt[garde]], s["centre"], self.max_points)
            _points(dossier / "caches.bin", n.xyz[caches], np.full((caches.sum(), 3), GRIS_POINTS_CACHES, np.uint8),
                    s["centre"], self.max_points)
            return f"{garde.sum():,} points sous les masques ; {caches.sum():,} écartés (cachés sous la surface)".replace(",", " ")

        def f_traces():
            s["troncons"], s["ponctuels"] = extraire(s["classes"], s["confiance"], s["r"], s["nuage"], cfg)
            infos["traces"] = _troncons(s["troncons"], s["centre"])
            infos["ponctuels"] = _ponctuels(s["ponctuels"], s["centre"])
            return f"{len(s['troncons'])} tronçons bruts, {len(s['ponctuels'])} ponctuels"

        def f_reseaux():
            s["raccordes"] = rac = raccorder(s["troncons"], cfg)
            infos["raccordes"] = _troncons(rac, s["centre"])
            return f"{len({t.id_cable for t in rac})} câble(s) après raccordement"

        def f_sous_masque():
            ded = [t for t in s["raccordes"] if t.origine == "deduit"]
            return f"{len(ded)} tronçon(s) déduit(s), {sum(t.longueur for t in ded):.2f} m"

        def f_ouvrages():
            infos["faces"] = []
            if cfg.faces_extremites:    # avant le prolongement : on regarde devant les bouts réellement vus
                faces = orthos_aux_extremites(s["raccordes"], s["nuage"], s["r"], s["t"], cfg)
                ecrire_faces(faces, dossier, s["nuage"], s["r"], cfg)   # à plat : le serveur ne sert pas les sous-dossiers
                infos["faces"] = [{"image": f"face_{i:02d}.png", "orientation": f.orientation}
                                  for i, f in enumerate(faces, 1)]
            avant = len(s["raccordes"])
            s["raccordes"] = rac = raccorder_ouvrages(s["raccordes"], s["ponctuels"], cfg)
            infos["ouvrages"] = _troncons(rac[avant:], s["centre"])
            return (f"{len(rac) - avant} prolongement(s) vers boîtes / coffrets, {sum(t.longueur for t in rac[avant:]):.2f} m ; "
                    f"{len(infos['faces'])} ortho(s) de face")

        def f_resultat():
            n = s["nuage"]
            fc = vers_geojson(s["raccordes"], s["ponctuels"], n.origine, n.crs)
            ecrire_geojson(dossier / "resultat.geojson", fc)
            if verite:
                infos["evaluation"] = evaluer(dossier / "resultat.geojson", verite, cfg)
            obs = sum(t.longueur for t in s["raccordes"] if t.origine == "observe")
            existant = reseau_existant(n, s["raccordes"], dossier, cfg)
            if existant is None:
                return f"{obs:.2f} m observés -> {dossier / 'resultat.geojson'} (réseau existant indisponible)"
            ecrire_geojson(dossier / "reseau_existant.geojson", existant)
            infos["reseau_existant"] = par_classe(existant, n.origine + s["centre"])
            nb = len(infos["reseau_existant"].get("raccord_probable", []))
            return f"{obs:.2f} m observés, {nb} raccord(s) probable(s) au réseau existant -> {dossier / 'resultat.geojson'}"

        fonctions = [f_nuage, f_ortho, f_tranchee, f_detection, f_nettoyage, f_traces, f_reseaux, f_sous_masque,
                     f_ouvrages, f_resultat]
        try:
            for nom, f in zip(ETAPES, fonctions):
                if self.etat["etapes"][nom]["statut"] == "sans_modele":
                    break
                etape(nom, f)
        except Exception:
            pass        # l'erreur est déjà dans l'état de l'étape et sur stderr
        finally:
            self.etat["en_cours"] = False
