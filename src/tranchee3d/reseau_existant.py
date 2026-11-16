"""Réseau BT existant (open data Enedis, 2D, précision 5–10 m) et raccord probable des câbles détectés.

Aide à la lecture du chantier, jamais évaluée : le souterrain est posé au Z des câbles détectés,
l'aérien à sol + `hauteur_aerien` ; un raccord vers l'aérien monte par un poteau ajouté.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, box, shape
from shapely.ops import nearest_points, transform

from .config import Config
from .nuage import Nuage
from .raccordement import bouts_libres
from .retour3d import Troncon

JEUX = {"bt_souterrain": "reseau-souterrain-bt", "bt_aerien": "reseau-bt"}
SOURCE = "Enedis, open data, Licence Ouverte 2.0"   # mention de source exigée par la licence


def telecharger(nuage: Nuage, cfg: Config) -> dict:
    """Lignes BT autour du nuage (emprise + marge_reseau), GeoJSON 2D dans le CRS du nuage (coordonnées d'origine)."""
    import pyproj
    if not nuage.crs:
        raise ValueError("CRS du nuage inconnu")
    mini, maxi = nuage.xyz.min(axis=0)[:2] + nuage.origine[:2], nuage.xyz.max(axis=0)[:2] + nuage.origine[:2]
    vers_wgs = pyproj.Transformer.from_crs(nuage.crs, "EPSG:4326", always_xy=True).transform
    depuis_wgs = pyproj.Transformer.from_crs("EPSG:4326", nuage.crs, always_xy=True).transform
    lon, lat = vers_wgs(*(mini + maxi) / 2)
    rayon = np.linalg.norm(maxi - mini) / 2 + cfg.marge_reseau
    emprise = box(*(mini - cfg.marge_reseau), *(maxi + cfg.marge_reseau))
    features = []
    for classe, jeu in JEUX.items():
        offset, total = 0, 1
        while offset < total:
            requete = urllib.parse.urlencode({
                "where": f"within_distance(geometry, geom'POINT({lon} {lat})', {rayon:.0f}m)",
                "limit": 100, "offset": offset})
            with urllib.request.urlopen(f"{cfg.url_opendata}/{jeu}/records?{requete}", timeout=cfg.delai_opendata) as r:
                page = json.load(r)
            total, offset = page["total_count"], offset + 100
            for enr in page["results"]:
                g = transform(depuis_wgs, shape(json.loads(enr["geometry"]))).intersection(emprise)
                for ligne in getattr(g, "geoms", [g]):
                    if ligne.geom_type == "LineString" and not ligne.is_empty:
                        features.append({"type": "Feature", "geometry": ligne.__geo_interface__,
                                         "properties": {"classe": classe}})
    return {"type": "FeatureCollection", "features": features}


def raccorder_reseau(reseau: dict, troncons: list[Troncon], nuage: Nuage, cfg: Config) -> dict:
    """Réseau existant en 3D + un raccord probable par câble (bout libre le plus proche, à moins de
    distance_raccord_max en plan) ; FeatureCollection 3D en coordonnées d'origine."""
    sol = float(np.median(nuage.xyz[:, 2]))
    obs = [t.pts[:, 2] for t in troncons if t.origine == "observe"]
    z_sout = float(np.median(np.concatenate(obs))) if obs else sol
    z_classe = {"bt_souterrain": z_sout, "bt_aerien": sol + cfg.hauteur_aerien}
    o = nuage.origine
    lignes = [(f["properties"]["classe"], LineString(np.asarray(f["geometry"]["coordinates"])[:, :2] - o[:2]))
              for f in reseau["features"]]

    sortie = []
    ajouter = lambda classe, pts, **p: sortie.append({
        "type": "Feature", "properties": {"classe": classe, "origine": "opendata", **p},
        "geometry": {"type": "LineString", "coordinates": (np.asarray(pts) + o).round(4).tolist()}})
    for classe, l in lignes:
        xy = np.array(l.coords)
        ajouter(classe, np.column_stack([xy, np.full(len(xy), z_classe[classe])]), source=SOURCE)

    par_cable = {}
    for t, i in bouts_libres(troncons, cfg.trou_observe):
        par_cable.setdefault(t.id_cable, []).append(t.pts[i])
    for cable, bouts in par_cable.items():
        candidats = [(l.distance(Point(b[:2])), classe, l, b) for classe, l in lignes for b in bouts]
        if not candidats:
            break
        d, classe, l, b = min(candidats, key=lambda c: c[0])
        if d > cfg.distance_raccord_max:
            continue
        cible = np.array(nearest_points(l, Point(b[:2]))[0].coords[0])
        pts = [b, [*cible, b[2]]]
        if classe == "bt_aerien":
            pts.append([*cible, z_classe[classe]])
            ajouter("poteau_ajoute", [[*cible, sol], [*cible, z_classe[classe]]], id_cable=cable)
        ajouter("raccord_probable", pts, type_reseau=classe, id_cable=cable, distance_m=round(d, 2))
    fc = {"type": "FeatureCollection", "features": sortie}
    if nuage.crs:
        fc["crs"] = {"type": "name", "properties": {"name": nuage.crs}}
    return fc


def reseau_existant(nuage: Nuage, troncons: list[Troncon], dossier: Path, cfg: Config) -> dict | None:
    """Télécharge (ou relit le cache opendata_bt.geojson) puis raccorde ; None si désactivé ou indisponible."""
    if not cfg.reseau_existant:
        return None
    cache = Path(dossier) / "opendata_bt.geojson"
    try:
        if cache.exists():
            reseau = json.loads(cache.read_text(encoding="utf-8"))
        else:
            reseau = telecharger(nuage, cfg)
            cache.write_text(json.dumps(reseau), encoding="utf-8")
    except Exception as err:     # hors ligne, API en panne, CRS absent : le résultat reste valable sans cette couche
        print(f"  avertissement : réseau existant indisponible ({type(err).__name__} : {err})")
        return None
    return raccorder_reseau(reseau, troncons, nuage, cfg)


def par_classe(fc: dict | None, decalage) -> dict[str, list]:
    """Lignes 3D par classe, décalées dans le repère d'affichage (atelier, visualiseur)."""
    lignes = {}
    for f in (fc or {}).get("features", []):
        pts = (np.asarray(f["geometry"]["coordinates"]) - decalage).round(4).tolist()
        lignes.setdefault(f["properties"]["classe"], []).append(pts)
    return lignes
