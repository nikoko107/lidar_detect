"""Lecture / écriture des nuages LAS-LAZ.

Les coordonnées sont décalées en repère local (float64 -> petites valeurs) pour
ne pas perdre de précision ; `origine` permet de revenir au repère d'origine.
"""
from __future__ import annotations

from dataclasses import dataclass

import laspy
import numpy as np


@dataclass
class Nuage:
    xyz: np.ndarray       # (N, 3) float64, repère local
    rvb: np.ndarray       # (N, 3) uint8
    origine: np.ndarray   # (3,) float64 : monde = local + origine
    crs: str | None = None

    def __len__(self) -> int:
        return len(self.xyz)


def lire_nuage(chemin) -> Nuage:
    las = laspy.read(str(chemin))
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    origine = np.floor(xyz.min(axis=0))
    xyz -= origine
    try:
        rvb = np.column_stack([las.red, las.green, las.blue]).astype(np.float64)
    except AttributeError as e:
        raise ValueError("Le nuage n'a pas de couleurs RVB.") from e
    if rvb.max() > 255:           # couleurs codées sur 16 bits
        rvb /= 257.0
    crs = None
    try:
        c = las.header.parse_crs()
        crs = c.to_string() if c is not None else None
    except Exception:
        pass
    return Nuage(xyz, rvb.round().clip(0, 255).astype(np.uint8), origine, crs)


def ecrire_nuage(chemin, xyz_monde: np.ndarray, rvb: np.ndarray) -> None:
    entete = laspy.LasHeader(point_format=2, version="1.2")
    entete.offsets = np.floor(xyz_monde.min(axis=0))
    entete.scales = np.array([0.0001, 0.0001, 0.0001])
    las = laspy.LasData(entete)
    las.x, las.y, las.z = xyz_monde[:, 0], xyz_monde[:, 1], xyz_monde[:, 2]
    c16 = rvb.astype(np.uint16) * 257
    las.red, las.green, las.blue = c16[:, 0], c16[:, 1], c16[:, 2]
    las.write(str(chemin))
