"""Export GeoJSON 3D (coordonnées d'origine) et images de contrôle."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from .config import CLASSES
from .retour3d import Ponctuel, Troncon

COULEURS = {1: (40, 120, 255), 2: (255, 60, 60), 3: (255, 200, 0), 4: (0, 220, 140)}


def vers_geojson(troncons: list[Troncon], ponctuels: list[Ponctuel], origine, crs=None) -> dict:
    """FeatureCollection 3D en coordonnées d'origine : LineString par tronçon (classe, origine observe/deduit,
    id_cable, longueur, confiance, rayon), Point par ponctuel (dessus de l'objet, z_min, emprise)."""
    arr = lambda p: [[round(float(v), 4) for v in q] for q in np.atleast_2d(p) + origine]
    features = []
    for t in troncons:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": arr(t.pts)},
            "properties": {
                "classe": CLASSES[t.classe], "origine": t.origine, "id_cable": t.id_cable,
                "longueur_m": round(t.longueur, 3), "confiance": round(t.confiance, 3),
                "rayon_estime_m": round(t.rayon, 4),
            },
        })
    for p in ponctuels:
        xmin, ymin, xmax, ymax = p.emprise
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": arr(p.xyz)[0]},
            "properties": {
                "classe": CLASSES[p.classe], "origine": "observe", "confiance": round(p.confiance, 3),
                "z_min": round(float(p.z_min + origine[2]), 4),
                "emprise": [round(float(v), 4) for v in
                            (xmin + origine[0], ymin + origine[1], xmax + origine[0], ymax + origine[1])],
            },
        })
    fc = {"type": "FeatureCollection", "features": features}
    if crs:
        fc["crs"] = {"type": "name", "properties": {"name": crs}}
    return fc


def ecrire_geojson(chemin, fc: dict) -> None:
    """Écrit le GeoJSON tel quel (coordonnées déjà en repère d'origine)."""
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


def image_controle(chemin, rvb: np.ndarray, classes: np.ndarray | None = None,
                   zone: np.ndarray | None = None, largeur_max=4000) -> None:
    """Ortho + masques en surimpression, pour vérifier une étape d'un coup d'œil."""
    img = rvb.astype(np.float32)
    if zone is not None:
        img[~zone] *= 0.45
    if classes is not None:
        for c, couleur in COULEURS.items():
            m = classes == c
            img[m] = 0.35 * img[m] + 0.65 * np.array(couleur, np.float32)
    im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    if max(im.size) > largeur_max:
        e = largeur_max / max(im.size)
        im = im.resize((max(1, int(im.width * e)), max(1, int(im.height * e))), Image.BILINEAR)
    im.save(chemin)
