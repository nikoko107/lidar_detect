"""Visualiseur 3D : une page HTML autonome (aucune connexion nécessaire).

Affiche le nuage (sous-échantillonné), les câbles observés et déduits, les
boîtes et coffrets, et la vérité terrain si elle est fournie.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from .nuage import Nuage
from .reseau_existant import par_classe

ACTIFS = Path(__file__).parent / "assets"


def _b64(tableau: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(tableau).tobytes()).decode("ascii")


def construire(chemin_html, nuage: Nuage, prediction: dict, verite: dict | None = None,
               max_points=1_500_000, titre="Tranchée 3D", graine=0, existant: dict | None = None) -> Path:
    """prediction / verite / existant (réseau open data + raccords) : dictionnaires GeoJSON en coordonnées d'origine."""
    centre = (nuage.xyz.min(axis=0) + nuage.xyz.max(axis=0)) / 2
    decalage = nuage.origine + centre                      # monde -> repère du visualiseur
    local = lambda coords: (np.atleast_2d(np.asarray(coords, dtype=float))[:, :3] - decalage).round(4)

    n = len(nuage)
    choix = np.arange(n) if n <= max_points else np.sort(
        np.random.default_rng(graine).choice(n, max_points, replace=False))

    troncons, ponctuels = [], []
    for f in prediction.get("features", []):
        g, p = f["geometry"], f["properties"]
        if g["type"] == "LineString":
            troncons.append({"classe": p["classe"], "origine": p.get("origine", "observe"),
                             "longueur": p.get("longueur_m", 0.0), "pts": local(g["coordinates"]).tolist()})
        elif g["type"] == "Point":
            xyz = local(g["coordinates"])[0]
            objet = {"classe": p["classe"], "xyz": xyz.tolist()}
            if "emprise" in p and "z_min" in p:
                x0, y0, x1, y1 = p["emprise"]
                objet["boite"] = [*(np.array([x0, y0, p["z_min"]]) - decalage).round(4).tolist(),
                                  *(np.array([x1, y1, g["coordinates"][2]]) - decalage).round(4).tolist()]
            ponctuels.append(objet)

    lignes_v, points_v = [], []
    z_defaut = float(nuage.origine[2] + centre[2])
    for f in (verite or {}).get("features", []):
        g = shape(f["geometry"])
        avec_z = lambda c: [(*q[:2], q[2] if len(q) > 2 else z_defaut) for q in c]
        if g.geom_type == "LineString":
            lignes_v.append(local(avec_z(g.coords)).tolist())
        elif g.geom_type == "MultiLineString":
            lignes_v += [local(avec_z(l.coords)).tolist() for l in g.geoms]
        elif g.geom_type == "Point":
            points_v.append(local(avec_z(g.coords))[0].tolist())

    donnees = {
        "nuage": {"positions": _b64((nuage.xyz[choix] - centre).astype(np.float32)),
                  "couleurs": _b64(nuage.rvb[choix]), "affiches": int(len(choix)), "total": int(n)},
        "troncons": troncons, "ponctuels": ponctuels,
        "verite": {"lignes": lignes_v, "points": points_v},
        "existant": par_classe(existant, decalage),
    }
    proteger = lambda s: s.replace("</", "<\\/")           # ne pas fermer la balise <script>
    html = (ACTIFS / "visualiseur.html").read_text(encoding="utf-8")
    html = html.replace("__TITRE__", titre)
    html = html.replace("__BUNDLE__", proteger((ACTIFS / "three_bundle.min.js").read_text(encoding="utf-8")))
    html = html.replace("__DONNEES__", proteger(json.dumps(donnees, separators=(",", ":"))))
    chemin_html = Path(chemin_html)
    chemin_html.write_text(html, encoding="utf-8")
    return chemin_html
