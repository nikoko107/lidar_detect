"""Chantier synthétique, pour tester la chaîne sans données réelles.

Produit un nuage LAS coloré et le GeoJSON de vérité terrain : une tranchée
sinueuse, un câble noir, un fourreau rouge, une boîte de jonction, un coffret
hors sol, et une plaque posée en travers de la tranchée (zone masquée).
Ce n'est qu'un banc d'essai : il ne remplace pas des données réelles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .nuage import ecrire_nuage

ORIGINE = np.array([652_340.0, 6_861_870.0, 48.0])   # coordonnées de type Lambert-93


def generer(dossier, graine=0, longueur=14.0, espacement=0.004):
    rng = np.random.default_rng(graine)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    largeur = 3.2
    nx, ny = int(longueur / espacement), int(largeur / espacement)
    X = (np.arange(nx) + 0.5) * espacement
    Y = (np.arange(ny) + 0.5) * espacement
    xx, yy = np.meshgrid(X, Y)

    bruit = lambda sigma: ndi.gaussian_filter(rng.standard_normal(xx.shape), sigma)
    norme = lambda a: a / (np.abs(a).max() + 1e-9)

    # --- sol et tranchée
    axe = largeur / 2 + rng.uniform(0.1, 0.25) * np.sin(X / rng.uniform(2.0, 3.5) + rng.uniform(0, 6))
    demi, profondeur = rng.uniform(0.22, 0.30), rng.uniform(0.7, 0.9)
    ecart = np.abs(yy - axe[None, :])
    z = 0.03 * norme(bruit(60)) + 0.004 * norme(bruit(2))
    creux = np.clip((demi + 0.06 - ecart) / 0.06, 0, 1)            # parois légèrement inclinées
    z -= profondeur * creux
    teinte = rng.uniform(0.8, 1.15)
    sol = np.array([118, 104, 84]) * teinte
    rvb = sol[None, None] * (1 + 0.22 * norme(bruit(3))[..., None] + 0.10 * norme(bruit(25))[..., None])
    rvb = rvb * (1 - 0.30 * creux[..., None])                      # fond de tranchée plus sombre
    rvb += rng.normal(0, 5, rvb.shape)

    def poser_cable(decalage, rayon, couleur, amplitude):
        cy = axe + decalage + amplitude * np.sin(X / rng.uniform(0.8, 1.6) + rng.uniform(0, 6))
        zf = -profondeur + 0.01 * np.sin(X / 1.3)
        d = np.abs(yy - cy[None, :])
        dessus = d < rayon
        zc = zf[None, :] + rayon + np.sqrt(np.clip(rayon**2 - d**2, 0, None))
        z[dessus] = np.maximum(z[dessus], zc[dessus])
        ombre = 1 - 0.45 * (d[dessus] / rayon) ** 2                # flancs plus sombres
        rvb[dessus] = np.array(couleur)[None] * ombre[:, None] + rng.normal(0, 4, (dessus.sum(), 3))
        return np.column_stack([X, cy, zf + 2 * rayon])            # génératrice supérieure

    noir = poser_cable(-0.09, 0.016, (34, 34, 38), 0.02)
    rouge = poser_cable(+0.09, 0.030, (190, 38, 34), 0.015)

    def poser_bloc(cx, cy, lx, ly, z_haut, couleur):
        m = (np.abs(xx - cx) < lx / 2) & (np.abs(yy - cy) < ly / 2)
        z[m] = z_haut
        rvb[m] = np.array(couleur)[None] * (1 + 0.06 * norme(bruit(4))[m][:, None]) + rng.normal(0, 3, (m.sum(), 3))

    # --- boîte de jonction sur le câble noir
    xb = rng.uniform(0.25, 0.4) * longueur
    ib = int(xb / espacement)
    poser_bloc(xb, noir[ib, 1], 0.20, 0.11, noir[ib, 2] + 0.05, (58, 60, 66))
    boite = [xb, noir[ib, 1], noir[ib, 2] + 0.05]

    # --- plaque en travers de la tranchée : les câbles passent dessous
    xp = rng.uniform(0.6, 0.75) * longueur
    ip = int(xp / espacement)
    poser_bloc(xp, axe[ip], rng.uniform(0.6, 0.9), 2 * demi + 0.7, 0.03, (150, 122, 80))

    # --- coffret hors sol, à côté de la tranchée
    xc = rng.uniform(0.1, 0.9) * longueur
    yc = axe[int(xc / espacement)] + rng.choice([-1, 1]) * (demi + 0.45)
    poser_bloc(xc, yc, 0.28, 0.20, 0.95, (176, 178, 172))
    coffret = [xc, yc, 0.95]

    # --- nuage : un point par cellule, avec un peu de bruit de position
    n = xx.size
    pts = np.column_stack([
        xx.ravel() + rng.uniform(-0.4, 0.4, n) * espacement,
        yy.ravel() + rng.uniform(-0.4, 0.4, n) * espacement,
        z.ravel() + rng.normal(0, 0.0015, n),
    ])
    garde = rng.random(n) > 0.05
    ecrire_nuage(dossier / "nuage.las", pts[garde] + ORIGINE, rvb.reshape(-1, 3).clip(0, 255)[garde].astype(np.uint8))

    ligne = lambda p: [[round(float(v), 4) for v in q] for q in p[::25] + ORIGINE]
    point = lambda p: [round(float(v), 4) for v in np.array(p) + ORIGINE]
    features = [
        {"type": "Feature", "properties": {"classe": "cable_noir"},
         "geometry": {"type": "LineString", "coordinates": ligne(noir)}},
        {"type": "Feature", "properties": {"classe": "fourreau_rouge"},
         "geometry": {"type": "LineString", "coordinates": ligne(rouge)}},
        {"type": "Feature", "properties": {"classe": "boite_jonction"},
         "geometry": {"type": "Point", "coordinates": point(boite)}},
        {"type": "Feature", "properties": {"classe": "coffret"},
         "geometry": {"type": "Point", "coordinates": point(coffret)}},
    ]
    with open(dossier / "verite.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    return dossier
