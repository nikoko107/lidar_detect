"""Orthos de face aux extrémités libres des câbles, pour chercher les coffrets.

Vu du dessus, un coffret se réduit à son toit. Devant chaque bout de câble, on découpe un petit
volume du nuage et on le projette sur un plan vertical : image RVB + profondeur, le point le plus
proche de l'observateur l'emporte. L'image est tournée face au plan vertical dominant du volume
(la face du coffret) ; à défaut, on regarde dans la direction du câble.

Cette étape ne détecte rien : elle produit les images, à contrôler à l'œil.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .config import CLASSES, Config
from .nuage import Nuage
from .ortho import Rasters
from .raccordement import _tangente_sortante, bouts_libres
from .retour3d import Troncon
from .tranchee import Tranchee

VERTICALE = np.array([0.0, 0.0, 1.0])


@dataclass
class OrthoFace:
    rvb: np.ndarray          # (H, W, 3) uint8
    profondeur: np.ndarray   # (H, W) float32 : distance au plan image le long de la visée (m), NaN si vide
    coin: np.ndarray         # (3,) local : coin bas gauche de l'image, sur le plan image
    u: np.ndarray            # (3,) unitaire, horizontal : sens des colonnes
    visee: np.ndarray        # (3,) unitaire, horizontal : de l'observateur vers l'objet
    pixel: float
    bout: np.ndarray         # (3,) local : extrémité du câble
    z_sol: float             # altitude locale du sol devant l'extrémité
    orientation: str         # "plan" (face au plan vertical dominant) ou "cable" (dans l'axe du câble)
    classe: int = 0
    id_cable: int = -1

    def px_vers_xyz(self, lig, col) -> np.ndarray:
        """(ligne, colonne) -> point 3D local vu dans ce pixel (NaN si le pixel est vide)."""
        lig, col = np.asarray(lig), np.asarray(col)
        H = self.profondeur.shape[0]
        return (self.coin + ((col + 0.5) * self.pixel)[..., None] * self.u
                + ((H - lig - 0.5) * self.pixel)[..., None] * VERTICALE
                + self.profondeur[lig, col][..., None] * self.visee)


def _plan_vertical(pts: np.ndarray, cfg: Config):
    """Plan vertical dominant : (normale horizontale (2,), centre en plan (2,)) ou None.
    Un plan vertical est une droite en plan : on balaie les orientations et on garde la bande de
    `face_plan_epaisseur` la plus peuplée, puis on affine la direction sur ses points."""
    if len(pts) < cfg.face_plan_points_min:
        return None
    part = min(1.0, cfg.face_plan_echantillon / len(pts))      # part tirée : le seuil de points suit
    if part < 1:
        pts = pts[np.random.default_rng(0).choice(len(pts), cfg.face_plan_echantillon, replace=False)]
    xy, ep = pts[:, :2], cfg.face_plan_epaisseur
    meilleur = (0, None, None)
    for a in np.radians(np.arange(0.0, 180.0, cfg.face_plan_pas_angle)):
        normale = np.array([np.cos(a), np.sin(a)])
        s = xy @ normale
        compte = np.append(np.bincount(np.floor((s - s.min()) / ep).astype(np.int64)), 0)
        paires = compte[:-1] + compte[1:]          # bandes chevauchantes : un plan à cheval sur deux cases
        i = int(paires.argmax())
        if paires[i] > meilleur[0]:
            meilleur = (int(paires[i]), normale, s.min() + (i + 1) * ep)
    n_bande, normale, position = meilleur
    if n_bande < cfg.face_plan_points_min * part:
        return None
    residu = xy @ normale - position
    z5, z95 = np.percentile(pts[np.abs(residu) <= ep, 2], [5, 95])
    if z95 - z5 < cfg.face_plan_hauteur_min:       # bande basse : bordure, tas de terre, pas une face
        return None
    # affinage : droite ajustée sur la bande, puis sur une bande trois fois plus fine (les points
    # d'une face voisine, au coin, font tourner le premier ajustement de quelques degrés)
    for tolerance in (ep, ep / 3):
        dans = np.abs(residu) <= tolerance
        centre = xy[dans].mean(axis=0)
        _, _, vt = np.linalg.svd(xy[dans] - centre, full_matrices=False)
        normale = np.array([-vt[0, 1], vt[0, 0]])  # perpendiculaire à la plus grande étendue
        residu = (xy - centre) @ normale
    return normale, centre


def ortho_de_face(nuage: Nuage, bout: np.ndarray, direction: np.ndarray, z_sol: float, cfg: Config,
                  classe: int = 0, id_cable: int = -1) -> OrthoFace | None:
    """Ortho de face du volume situé devant `bout`, dans la `direction` (sortante) du câble.
    None s'il y a moins de `face_points_min` points au-dessus du sol (rien à voir)."""
    d = direction[:2] / np.linalg.norm(direction[:2])
    lateral = np.array([-d[1], d[0]])
    rayon = np.hypot(max(cfg.face_portee, cfg.face_recul), cfg.face_demi_largeur)
    proches = np.flatnonzero((np.abs(nuage.xyz[:, 0] - bout[0]) <= rayon) & (np.abs(nuage.xyz[:, 1] - bout[1]) <= rayon))
    p = nuage.xyz[proches]
    rel, h = p[:, :2] - bout[:2], p[:, 2] - z_sol
    avance = rel @ d
    garde = ((avance >= -cfg.face_recul) & (avance <= cfg.face_portee) & (np.abs(rel @ lateral) <= cfg.face_demi_largeur)
             & (h >= cfg.face_z_min) & (h <= cfg.face_hauteur))
    if garde.sum() < cfg.face_points_min:
        return None
    p, rvb, h = p[garde], nuage.rvb[proches[garde]], h[garde]

    plan = _plan_vertical(p[h >= cfg.face_plan_z_min], cfg)
    if plan is None:
        visee, orientation = d, "cable"
    else:
        visee, centre = plan
        if visee @ (centre - bout[:2]) < 0:        # l'observateur est du côté du câble
            visee = -visee
        orientation = "plan"
    u = np.array([visee[1], -visee[0]])            # droite de l'observateur (visée x verticale)

    px = cfg.pixel_face
    cu, prof = p[:, :2] @ u, p[:, :2] @ visee
    u0, p0, z0 = cu.min(), prof.min(), z_sol + cfg.face_z_min
    W = int(np.ceil((cu.max() - u0) / px)) + 1
    H = int(np.ceil((cfg.face_hauteur - cfg.face_z_min) / px)) + 1
    col = np.floor((cu - u0) / px).astype(np.int64).clip(0, W - 1)
    lig = (H - 1 - np.floor((p[:, 2] - z0) / px).astype(np.int64)).clip(0, H - 1)
    image = np.zeros((H, W, 3), np.uint8)
    profondeur = np.full((H, W), np.nan, np.float32)
    ordre = np.argsort(-prof, kind="stable")       # du plus loin au plus proche : le plus proche l'emporte
    plat = (lig * W + col)[ordre]
    profondeur.reshape(-1)[plat] = prof[ordre] - p0
    image.reshape(-1, 3)[plat] = rvb[ordre]

    vide = np.isnan(profondeur)
    if cfg.bouchage_px > 0 and vide.any():
        dist, (il, ic) = ndi.distance_transform_edt(vide, return_indices=True)
        trou = vide & (dist <= cfg.bouchage_px)
        profondeur[trou], image[trou] = profondeur[il[trou], ic[trou]], image[il[trou], ic[trou]]

    en3d = lambda v: np.array([v[0], v[1], 0.0])
    return OrthoFace(image, profondeur, u0 * en3d(u) + p0 * en3d(visee) + z0 * VERTICALE, en3d(u), en3d(visee),
                     px, np.asarray(bout, float), float(z_sol), orientation, classe, id_cable)


def _sol_devant(r: Rasters, t: Tranchee, bout: np.ndarray, d: np.ndarray, cfg: Config) -> float | None:
    """Altitude du sol estimé (hauteur - hauteur relative), médiane autour du centre du volume."""
    centre = bout[:2] + d * cfg.face_portee / 2
    lig, col = r.xy_vers_px(centre[0], centre[1])
    k = max(1, int(round(cfg.face_demi_largeur / r.pixel)))
    H, W = r.forme
    fen = np.s_[max(0, lig - k):max(0, lig + k + 1), max(0, col - k):max(0, col + k + 1)]
    sol = r.hauteur[fen] - t.hauteur_relative[fen]
    sol = sol[np.isfinite(sol)]
    return float(np.median(sol)) if sol.size else None


def orthos_aux_extremites(troncons: list[Troncon], nuage: Nuage, r: Rasters, t: Tranchee, cfg: Config) -> list[OrthoFace]:
    """Une ortho de face par extrémité libre de câble (tronçons raccordés, avant prolongement aux ouvrages)."""
    portee = int(round(cfg.portee_tangente / cfg.pas))
    faces = []
    for tr, b in bouts_libres(troncons, cfg.trou_observe):
        d = _tangente_sortante(tr.pts, 0 if b == 0 else 1, portee)[:2]
        if np.linalg.norm(d) < 1e-6:               # bout vertical : pas de direction en plan
            continue
        d = d / np.linalg.norm(d)
        z_sol = _sol_devant(r, t, tr.pts[b], d, cfg)
        if z_sol is None:
            continue
        face = ortho_de_face(nuage, tr.pts[b], d, z_sol, cfg, tr.classe, tr.id_cable)
        if face is not None:
            faces.append(face)
    return faces


def _image_face(f: OrthoFace, plage: float) -> Image.Image:
    """RVB à gauche, profondeur à droite (clair = proche), fond sombre là où il n'y a pas de point.
    Le gris couvre au moins `plage` mètres : une face plane ne doit pas ressortir en dégradé."""
    vide = np.isnan(f.profondeur)
    rvb = f.rvb.copy()
    rvb[vide] = (24, 26, 34)
    etendue = max(float(np.nanmax(f.profondeur)), plage)
    gris = (255 * (1 - 0.85 * np.nan_to_num(f.profondeur) / etendue)).astype(np.uint8)
    prof = np.repeat(gris[..., None], 3, axis=2)
    prof[vide] = (24, 26, 34)
    separation = np.full((rvb.shape[0], 4, 3), 255, np.uint8)
    return Image.fromarray(np.hstack([rvb, separation, prof]))


def ecrire_faces(faces: list[OrthoFace], dossier, nuage: Nuage, r: Rasters, cfg: Config, largeur_max: int = 4000) -> None:
    """Par extrémité : face_NN.png (contrôle) et face_NN.npz (rasters + repère, pour la suite).
    Plus faces.geojson (trace au sol de chaque plan image, repère d'origine) et controle_faces.png (vue de dessus)."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    monde = lambda p: [round(float(v), 4) for v in p + nuage.origine]
    e = min(1.0, largeur_max / max(r.rvb.shape[:2]))
    plan = Image.fromarray(r.rvb).resize((max(1, int(r.rvb.shape[1] * e)), max(1, int(r.rvb.shape[0] * e))), Image.BILINEAR)
    dessin = ImageDraw.Draw(plan)
    px = lambda p: ((p[0] - r.x0) / r.pixel * e, (r.y1 - p[1]) / r.pixel * e)
    features = []
    for i, f in enumerate(faces, 1):
        nom = f"face_{i:02d}"
        _image_face(f, cfg.face_plage_gris).save(dossier / f"{nom}.png")
        np.savez_compressed(dossier / f"{nom}.npz", rvb=f.rvb, profondeur=f.profondeur, coin=f.coin, u=f.u,
                            visee=f.visee, pixel=f.pixel, bout=f.bout, z_sol=f.z_sol, origine=nuage.origine)
        a, b = f.coin, f.coin + f.profondeur.shape[1] * f.pixel * f.u
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [monde(a), monde(b)]},
            "properties": {"image": f"{nom}.png", "classe": CLASSES[f.classe], "id_cable": f.id_cable,
                           "orientation": f.orientation, "extremite": monde(f.bout),
                           "z_sol": round(f.z_sol + float(nuage.origine[2]), 4)},
        })
        dessin.line([px(a), px(b)], fill=(255, 220, 0), width=3)
        dessin.line([px(f.bout), px((a + b) / 2)], fill=(0, 220, 255), width=2)
        dessin.text(px(b), str(i), fill=(255, 220, 0))
    fc = {"type": "FeatureCollection", "features": features}
    if nuage.crs:
        fc["crs"] = {"type": "name", "properties": {"name": nuage.crs}}
    (dossier / "faces.geojson").write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    plan.save(dossier / "controle_faces.png")
