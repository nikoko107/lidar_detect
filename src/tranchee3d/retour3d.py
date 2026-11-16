"""Masques 2D -> géométries 3D mesurées sur le nuage brut.

Câbles : squelette du masque, puis en chaque point du squelette on mesure la
ligne de crête (génératrice supérieure) dans les points du nuage.
Ponctuels : composantes connexes, centroïde et emprise 3D.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .config import LINEAIRES, PONCTUELS, Config
from .nuage import Nuage
from .ortho import Rasters

HUIT = np.ones((3, 3), bool)


@dataclass
class Troncon:
    classe: int
    pts: np.ndarray              # (N, 3) repère local
    origine: str = "observe"     # "observe" ou "deduit"
    confiance: float = 1.0
    rayon: float = 0.0           # demi-largeur estimée (m)
    id_cable: int = -1

    @property
    def longueur(self) -> float:
        return float(np.linalg.norm(np.diff(self.pts, axis=0), axis=1).sum())


@dataclass
class Ponctuel:
    classe: int
    xyz: np.ndarray              # (3,) repère local, Z = dessus de l'objet
    z_min: float
    emprise: tuple               # (xmin, ymin, xmax, ymax) local
    confiance: float = 1.0
    proprietes: dict = field(default_factory=dict)


def _sans_petits_objets(masque: np.ndarray, aire_min: int) -> np.ndarray:
    etiq, n = ndi.label(masque, structure=HUIT)
    if n == 0:
        return masque
    aires = np.bincount(etiq.ravel())
    aires[0] = 0
    return (aires >= aire_min)[etiq]


# ---------------------------------------------------------------- squelette

def _ordonner(coords: np.ndarray) -> np.ndarray:
    """Ordonne les pixels d'une branche de squelette d'une extrémité à l'autre."""
    restants = {tuple(c) for c in coords}
    voisins = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def nb_voisins(p):
        return sum((p[0] + dl, p[1] + dc) in restants for dl, dc in voisins)

    depart = next((p for p in restants if nb_voisins(p) == 1), next(iter(restants)))
    chemin = [depart]
    restants.discard(depart)
    while True:
        p = chemin[-1]
        suivant = next(((p[0] + dl, p[1] + dc) for dl, dc in voisins
                        if (p[0] + dl, p[1] + dc) in restants), None)
        if suivant is None:
            break
        chemin.append(suivant)
        restants.discard(suivant)
    return np.array(chemin)


def chemins_squelette(masque: np.ndarray):
    """Découpe le squelette aux embranchements et renvoie des chemins de pixels ordonnés."""
    sk = skeletonize(masque)
    nb = ndi.convolve(sk.astype(np.uint8), HUIT.astype(np.uint8), mode="constant") - 1
    noeuds = sk & (nb > 2)
    branches = sk & ~ndi.binary_dilation(noeuds, structure=HUIT)
    etiq, _ = ndi.label(branches, structure=HUIT)
    for k, tranche in enumerate(ndi.find_objects(etiq), 1):
        coords = np.argwhere(etiq[tranche] == k)
        if len(coords) >= 3:
            yield _ordonner(coords) + [tranche[0].start, tranche[1].start]


# ---------------------------------------------------------------- mesure 3D

def _reechantillonner(xy: np.ndarray, pas: float) -> np.ndarray:
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if s[-1] < pas:
        return xy[[0, -1]]
    cible = np.linspace(0, s[-1], int(round(s[-1] / pas)) + 1)
    return np.column_stack([np.interp(cible, s, xy[:, 0]), np.interp(cible, s, xy[:, 1])])


def _lisser(pts: np.ndarray, fenetre: int) -> np.ndarray:
    n = len(pts)
    pts = pts.copy()
    if n >= 5:
        pts[:, 2] = ndi.median_filter(pts[:, 2], size=5, mode="nearest")   # points aberrants en Z
    f = min(fenetre, n if n % 2 else n - 1)
    if f >= 5:
        pts = savgol_filter(pts, f, 2, axis=0, mode="interp")
    return pts


def classer_points(classes: np.ndarray, r: Rasters, nuage: Nuage, cfg: Config):
    """Pixel de chaque point, masque des points de surface et classe de chaque point (0 si caché)."""
    H, W = r.forme
    lig, col = r.xy_vers_px(nuage.xyz[:, 0], nuage.xyz[:, 1])
    dedans = (lig >= 0) & (lig < H) & (col >= 0) & (col < W)
    lig, col = lig.clip(0, H - 1), col.clip(0, W - 1)
    # on ne garde que la surface vue du dessus (pas les points cachés sous un objet)
    surface = dedans & (nuage.xyz[:, 2] >= r.hauteur[lig, col] - cfg.tolerance_surface)
    return lig, col, surface, np.where(surface, classes[lig, col], 0)


def lisser_z(z: np.ndarray, cfg: Config) -> np.ndarray:
    """Profil en long : médiane glissante (écarte les points de paroi ou de terre) puis moyenne
    glissante sur `lissage_z` mètres. Un fourreau ne fait pas de vagues de quelques centimètres."""
    n = min(len(z), int(round(cfg.lissage_z / cfg.pas))) | 1
    if cfg.lissage_z <= 0 or n < 3:
        return z
    return ndi.uniform_filter1d(ndi.median_filter(z, size=n, mode="nearest"), n, mode="nearest")


def sans_aberrants_z(z: np.ndarray, cfg: Config) -> np.ndarray:
    """Mesures de crête prises sur un objet au-dessus du câble (planche, plaque) : écartées si elles
    s'éloignent de plus de `ecart_z_aberrant` de la médiane sur `lissage_z`, puis interpolées.
    Médiane en miroir : en bout de tronçon, recopier le premier point (souvent l'aberrant) la fausserait."""
    n = min(len(z), int(round(cfg.lissage_z / cfg.pas))) | 1
    if cfg.ecart_z_aberrant <= 0 or n < 3:
        return z
    garde = np.abs(z - ndi.median_filter(z, size=n, mode="reflect")) <= cfg.ecart_z_aberrant
    if garde.all() or not garde.any():
        return z
    i = np.arange(len(z))
    return np.interp(i, i[garde], z[garde])


def extraire(classes: np.ndarray, confiance: np.ndarray, r: Rasters, nuage: Nuage, cfg: Config):
    """Retourne (tronçons observés, ponctuels)."""
    lig, col, surface, classe_pt = classer_points(classes, r, nuage, cfg)

    troncons: list[Troncon] = []
    for c in LINEAIRES:
        masque = _sans_petits_objets(classes == c, cfg.surface_min_px)
        idx = np.flatnonzero(classe_pt == c)
        if not masque.any() or len(idx) == 0:
            continue
        pts_c = nuage.xyz[idx]
        arbre = cKDTree(pts_c[:, :2])
        demi_largeur = ndi.distance_transform_edt(masque) * r.pixel

        for chemin in chemins_squelette(masque):
            l, k = chemin[:, 0], chemin[:, 1]
            rayon = float(np.clip(np.median(demi_largeur[l, k]), cfg.rayon_min, cfg.rayon_max))
            xy = _reechantillonner(np.column_stack(r.px_vers_xy(l, k)), cfg.pas)
            mesures = []
            for p, voisins in zip(xy, arbre.query_ball_point(xy, max(rayon, 2 * r.pixel))):
                if len(voisins) < 3:
                    continue
                v = pts_c[voisins]
                z_crete = np.percentile(v[:, 2], 90)
                haut = v[v[:, 2] >= z_crete - cfg.part_crete * rayon]
                mesures.append([*haut[:, :2].mean(axis=0), z_crete])
            # les bouts du squelette sont peu fiables (masque arrondi) : on les rogne
            rogne = int(np.ceil(1.5 * rayon / cfg.pas))
            if len(mesures) > 2 * rogne + 4:
                mesures = mesures[rogne:-rogne]
            if len(mesures) < 4:
                continue
            mesures = np.array(mesures)
            mesures[:, 2] = sans_aberrants_z(mesures[:, 2], cfg)
            pts = _lisser(mesures, cfg.lissage)
            pts[:, 2] = lisser_z(pts[:, 2], cfg)
            if cfg.convention == "axe":
                pts[:, 2] -= rayon
            t = Troncon(c, pts, "observe", float(confiance[l, k].mean()), rayon)
            if t.longueur >= cfg.longueur_min:
                troncons.append(t)

    ponctuels: list[Ponctuel] = []
    for c in PONCTUELS:
        masque = ndi.binary_opening(classes == c, structure=HUIT, iterations=2)
        etiq, n = ndi.label(masque)
        etiq_pt = np.where(surface, etiq[lig, col], 0)
        for k in range(1, n + 1):
            if (etiq == k).sum() < cfg.surface_min_px:
                continue
            v = nuage.xyz[etiq_pt == k]
            if len(v) < 10:
                continue
            z_haut, z_bas = np.percentile(v[:, 2], [98, 2])
            ponctuels.append(Ponctuel(
                c, np.array([*v[:, :2].mean(axis=0), z_haut]), float(z_bas),
                (*v[:, :2].min(axis=0), *v[:, :2].max(axis=0)),
                float(confiance[etiq == k].mean()),
            ))
    return troncons, ponctuels
