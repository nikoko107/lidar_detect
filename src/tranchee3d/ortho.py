"""Nuage -> ortho RVB + raster de hauteur, sur une grille commune."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .config import Config
from .nuage import Nuage


@dataclass
class Rasters:
    rvb: np.ndarray        # (H, W, 3) uint8
    hauteur: np.ndarray    # (H, W) float32, NaN si vide (repère local)
    x0: float              # bord gauche (local)
    y1: float              # bord haut (local)
    pixel: float
    source_ortho: str = "nuage"   # "nuage" ou "fournie"

    @property
    def forme(self):
        return self.hauteur.shape

    def xy_vers_px(self, x, y):
        """Coordonnées locales -> (ligne, colonne) entières."""
        col = np.floor((np.asarray(x) - self.x0) / self.pixel).astype(np.int64)
        lig = np.floor((self.y1 - np.asarray(y)) / self.pixel).astype(np.int64)
        return lig, col

    def px_vers_xy(self, lig, col):
        """(ligne, colonne) -> centre du pixel en coordonnées locales."""
        x = self.x0 + (np.asarray(col) + 0.5) * self.pixel
        y = self.y1 - (np.asarray(lig) + 0.5) * self.pixel
        return x, y

    def affine_monde(self, origine):
        """Transformation affine rasterio dans le repère d'origine."""
        from rasterio.transform import from_origin
        return from_origin(self.x0 + origine[0], self.y1 + origine[1], self.pixel, self.pixel)


def generer_rasters(nuage: Nuage, cfg: Config, cadre: Rasters | None = None) -> Rasters:
    """Projection verticale : le point le plus haut de chaque pixel l'emporte. `cadre` : reprendre sa grille."""
    p = cfg.pixel
    if cadre is not None:
        H, W = cadre.forme
        r = Rasters(np.zeros((H, W, 3), np.uint8), np.full((H, W), np.nan, np.float32), cadre.x0, cadre.y1, p)
    else:
        x0, y0 = nuage.xyz[:, :2].min(axis=0)
        x1, y1 = nuage.xyz[:, :2].max(axis=0)
        W = int(np.ceil((x1 - x0) / p)) + 1
        H = int(np.ceil((y1 - y0) / p)) + 1
        # bord haut = y0 + (H-1)p, pas y0 + Hp : sinon la ligne 0 est toujours vide et y0 tombe hors grille (ligne H)
        r = Rasters(np.zeros((H, W, 3), np.uint8), np.full((H, W), np.nan, np.float32), x0, y0 + (H - 1) * p, p)

    lig, col = r.xy_vers_px(nuage.xyz[:, 0], nuage.xyz[:, 1])
    lig, col = lig.clip(0, H - 1), col.clip(0, W - 1)      # points posés pile sur le bord
    ordre = np.argsort(nuage.xyz[:, 2], kind="stable")       # du plus bas au plus haut
    plat = (lig * W + col)[ordre]
    # en cas d'indices répétés, la dernière affectation (la plus haute) est conservée
    r.hauteur.reshape(-1)[plat] = nuage.xyz[ordre, 2]
    r.rvb.reshape(-1, 3)[plat] = nuage.rvb[ordre]

    _boucher(r, cfg.bouchage_px)
    return r


def _boucher(r: Rasters, dist_max: int) -> None:
    """Comble les petits trous par le plus proche voisin."""
    vide = np.isnan(r.hauteur)
    if dist_max <= 0 or not vide.any():
        return
    dist, (il, ic) = ndi.distance_transform_edt(vide, return_indices=True)
    a_combler = vide & (dist <= dist_max)
    r.hauteur[a_combler] = r.hauteur[il[a_combler], ic[a_combler]]
    r.rvb[a_combler] = r.rvb[il[a_combler], ic[a_combler]]


def appliquer_ortho_fournie(r: Rasters, chemin_ortho, origine) -> None:
    """Remplace les couleurs par celles d'une ortho GeoTIFF, rééchantillonnée sur la grille."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    H, W = r.forme
    with rasterio.open(chemin_ortho) as src:
        dst = np.zeros((3, H, W), np.uint8)
        for b in range(3):
            reproject(
                rasterio.band(src, b + 1), dst[b],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=r.affine_monde(origine), dst_crs=src.crs,
                resampling=Resampling.bilinear,
            )
    couvert = dst.any(axis=0)
    r.rvb[couvert] = np.moveaxis(dst, 0, -1)[couvert]
    r.source_ortho = "fournie"


def ecrire_geotiff(chemin, tableau: np.ndarray, r: Rasters, origine, crs=None) -> None:
    import rasterio

    if tableau.ndim == 2:
        tableau = tableau[None]
    else:
        tableau = np.moveaxis(tableau, -1, 0)
    with rasterio.open(
        chemin, "w", driver="GTiff", height=tableau.shape[1], width=tableau.shape[2],
        count=tableau.shape[0], dtype=tableau.dtype, transform=r.affine_monde(origine),
        crs=crs, compress="deflate",
    ) as dst:
        dst.write(tableau)
