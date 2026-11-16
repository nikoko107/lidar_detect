"""Détection de la tranchée sur le raster de hauteur (sans apprentissage)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize

from .config import Config
from .nuage import Nuage
from .ortho import Rasters


@dataclass
class Tranchee:
    zone: np.ndarray               # (H, W) bool : tranchée + marge (zone d'intérêt)
    fond: np.ndarray               # (H, W) bool : tranchée seule
    hauteur_relative: np.ndarray   # (H, W) float32 : hauteur - sol (NaN si vide)


def points_fond(nuage: Nuage, r: Rasters, t: Tranchee, cfg: Config) -> np.ndarray:
    """Points du fond de fouille et des `hauteur_fond` au-dessus, seulement. Fond = Z minimal par cellule de
    `grille_sol` (médiane 3x3 contre le bruit), minimum glissant sur `fenetre_fond` (une cellule de paroi prend
    le fond voisin), pris dans la tranchée détectée et prolongé au plus proche au-delà ; hors zone d'intérêt, rien. Écartés : terrain
    autour, parois hautes, lèvres de terre, surplombs, coffrets."""
    f = max(1, int(round(cfg.grille_sol / r.pixel)))
    lig, col = r.xy_vers_px(nuage.xyz[:, 0], nuage.xyz[:, 1])
    H, W = (r.forme[0] - 1) // f + 1, (r.forme[1] - 1) // f + 1
    lig, col = lig.clip(0, r.forme[0] - 1), col.clip(0, r.forme[1] - 1)
    cellule = lig // f * W + col // f
    fond = np.full(H * W, np.nan)
    ordre = np.argsort(nuage.xyz[:, 2], kind="stable")[::-1]     # du plus haut au plus bas : le plus bas reste
    fond[cellule[ordre]] = nuage.xyz[ordre, 2]
    fond = fond.reshape(H, W)
    dans = t.fond[::f, ::f]
    if not dans.any():
        raise ValueError("Pas de tranchée détectée : fond de fouille introuvable.")
    dans = ndi.binary_erosion(dans) if ndi.binary_erosion(dans).any() else dans   # cellules de bord : à demi terrain
    _, (il, ic) = ndi.distance_transform_edt(np.isnan(fond), return_indices=True)
    fond = ndi.median_filter(fond[il, ic], size=3, mode="nearest")
    taille = max(3, int(round(cfg.fenetre_fond / cfg.grille_sol)) | 1)
    fond = ndi.minimum_filter(fond, size=taille, mode="nearest")
    _, (il, ic) = ndi.distance_transform_edt(~dans, return_indices=True)
    fond = fond[il, ic].reshape(-1)
    # hors zone d'intérêt, un terrain plus bas que le fond (rue en pente) passerait le seuil
    return (nuage.xyz[:, 2] <= fond[cellule] + cfg.hauteur_fond) & t.zone[lig, col]


def detecter_tranchee(r: Rasters, cfg: Config) -> Tranchee:
    H, W = r.forme
    f = max(1, int(round(cfg.grille_sol / r.pixel)))

    # 1. hauteur à basse résolution (médiane par bloc), trous comblés
    Hc, Wc = H // f, W // f
    blocs = r.hauteur[: Hc * f, : Wc * f].reshape(Hc, f, Wc, f).transpose(0, 2, 1, 3).reshape(Hc, Wc, -1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hc = np.nanmedian(blocs, axis=2)
    vide = np.isnan(hc)
    if vide.all():
        raise ValueError("Raster de hauteur vide.")
    if vide.any():
        _, (il, ic) = ndi.distance_transform_edt(vide, return_indices=True)
        hc = hc[il, ic]

    # 2. sol = centile haut sur une grande fenêtre (insensible à la tranchée, plus étroite)
    taille = max(3, int(round(cfg.fenetre_sol / (f * r.pixel))) | 1)
    sol = ndi.percentile_filter(hc, cfg.centile_sol, size=taille, mode="nearest")
    sol = ndi.gaussian_filter(sol, taille / 4)

    # 3. tranchée = ce qui est nettement sous le sol
    fond = (sol - hc > cfg.profondeur_min) & ~vide
    fond = ndi.binary_opening(fond, iterations=1)
    etiq, n = ndi.label(fond)
    if n:
        aires = ndi.sum(fond, etiq, range(1, n + 1)) * (f * r.pixel) ** 2
        fond = np.isin(etiq, 1 + np.flatnonzero(aires >= cfg.surface_min_tranchee))

    # 4. marge autour de la tranchée
    n_marge = int(round(cfg.marge_tranchee / (f * r.pixel)))
    zone = ndi.binary_dilation(fond, iterations=max(1, n_marge)) if fond.any() else fond

    haut = lambda a, ordre: resize(a.astype(np.float32), (H, W), order=ordre, anti_aliasing=False)
    sol_plein = haut(sol, 1)
    return Tranchee(
        zone=haut(zone, 0) > 0.5,
        fond=haut(fond, 0) > 0.5,
        hauteur_relative=(r.hauteur - sol_plein).astype(np.float32),
    )


def tenseur_entree(r: Rasters, t: Tranchee) -> np.ndarray:
    """Entrée du modèle : (4, H, W) float32 = RVB normalisé + hauteur relative."""
    x = np.empty((4, *r.forme), np.float32)
    x[:3] = np.moveaxis(r.rvb, -1, 0) / 255.0
    h = np.nan_to_num(t.hauteur_relative, nan=0.0)
    x[3] = np.clip(h, -1.5, 1.5) / 1.5
    return x
