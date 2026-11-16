"""GeoJSON de vérité terrain -> masque de classes aligné sur l'ortho."""
from __future__ import annotations

import json

import numpy as np
from rasterio.features import rasterize
from scipy.spatial import cKDTree
from shapely.geometry import shape

from .config import Config
from .ortho import Rasters


def lire_geojson(chemin, cfg: Config):
    """Retourne [(id_classe, géométrie shapely en coordonnées d'origine, propriétés)]."""
    donnees = json.loads(open(chemin, encoding="utf-8").read())
    objets, ignores = [], 0
    for f in donnees.get("features", []):
        valeur = (f.get("properties") or {}).get(cfg.champ_classe)
        if valeur not in cfg.valeurs_classes or f.get("geometry") is None:
            ignores += 1
            continue
        objets.append((cfg.valeurs_classes[valeur], shape(f["geometry"]), f["properties"]))
    if ignores:
        print(f"  {ignores} objet(s) GeoJSON ignoré(s) (classe inconnue dans '{cfg.champ_classe}').")
    return objets


def rasteriser(objets, r: Rasters, origine, cfg: Config) -> np.ndarray:
    """Masque (H, W) uint8. Les ponctuels sont dessinés en dernier (au-dessus des câbles)."""
    tampons = {1: cfg.tampon_cable_noir, 2: cfg.tampon_fourreau, 3: cfg.rayon_boite, 4: cfg.rayon_coffret}
    formes = [(g.buffer(tampons[c]), c) for c, g, _ in sorted(objets, key=lambda o: o[0])]
    if not formes:
        return np.zeros(r.forme, np.uint8)
    masque = rasterize(formes, out_shape=r.forme, transform=r.affine_monde(origine), fill=0, dtype="uint8")
    _retirer_parties_cachees(masque, objets, r, origine, cfg)
    return masque


def est_rouge(rvb: np.ndarray, cfg: Config) -> np.ndarray:
    """(N, 3) uint8 -> (N,) bool : couleur du fourreau rouge (seuils `rouge_visible`, teinte qui passe par 0)."""
    from skimage.color import rgb2hsv
    t_min, t_max, s_min, v_min = cfg.rouge_visible
    h = rgb2hsv(rvb.reshape(-1, 1, 3).astype(np.uint8))[:, 0]
    return ((h[:, 0] >= t_min) | (h[:, 0] <= t_max)) & (h[:, 1] >= s_min) & (h[:, 2] >= v_min)


def _retirer_parties_cachees(masque, objets, r: Rasters, origine, cfg: Config) -> None:
    """Un câble levé qui passe sous un obstacle n'est pas visible sur l'ortho : là où la
    surface est nettement au-dessus du Z du câble, l'étiquette est retirée (sinon le modèle
    apprendrait à « voir » un câble sur l'obstacle). Nécessite un GeoJSON avec Z.
    Exception : un pixel de fourreau de couleur rouge (`rouge_visible`) est visible, par exemple contre une
    paroi au-dessus du point levé ; il reste étiqueté, sinon le modèle apprendrait ce rouge comme du fond."""
    lignes = [(c, g) for c, g, _ in objets if c in (1, 2) and g.geom_type == "LineString"]
    if not lignes:
        return
    if not all(g.has_z for _, g in lignes):
        print("  GeoJSON sans Z : les câbles passant sous un obstacle restent étiquetés.")
        return
    echantillons = []
    for _, g in lignes:
        c = np.array(g.coords, dtype=float) - origine
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))])
        cible = np.arange(0, s[-1] + r.pixel, 2 * r.pixel)
        echantillons.append(np.column_stack([np.interp(cible, s, c[:, k]) for k in range(3)]))
    echantillons = np.vstack(echantillons)
    lig, col = np.nonzero((masque == 1) | (masque == 2))
    x, y = r.px_vers_xy(lig, col)
    _, plus_proche = cKDTree(echantillons[:, :2]).query(np.column_stack([x, y]))
    cache = r.hauteur[lig, col] - echantillons[plus_proche, 2] > cfg.ecart_obstacle
    cache &= ~((masque[lig, col] == 2) & est_rouge(r.rvb[lig, col], cfg))
    masque[lig[cache], col[cache]] = 0
