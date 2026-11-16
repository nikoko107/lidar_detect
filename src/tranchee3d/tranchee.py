"""Détection de la tranchée sur le raster de hauteur (sans apprentissage)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize

from .config import Config
from .ortho import Rasters


@dataclass
class Tranchee:
    zone: np.ndarray               # (H, W) bool : tranchée + marge (zone d'intérêt)
    fond: np.ndarray               # (H, W) bool : tranchée seule
    hauteur_relative: np.ndarray   # (H, W) float32 : hauteur - sol (NaN si vide)


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
