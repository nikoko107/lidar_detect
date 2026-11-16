"""Import d'un chantier Enedis (dossier extrait d'une archive de récolement).

Entrée : nuage/*.laz (un ou plusieurs), recolement/RPD_*_Reco.geojson, project-metadata.json.
Sortie : nuage.laz (nuages fusionnés, CRS écrit dans l'en-tête) et verite.geojson
(une propriété de classe, comme l'attend etiquettes.py).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import laspy
import numpy as np
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

# fichier RPD -> classe ; le câble n'est gardé que hors fourreau (câble noir nu)
FICHIERS = {"Fourreau": "fourreau_rouge", "Jonction": "boite_jonction", "Coffret": "coffret"}


def _features(dossier: Path, nom: str) -> list[dict]:
    p = dossier / "recolement" / f"RPD_{nom}_Reco.geojson"
    return json.loads(p.read_text(encoding="utf-8"))["features"] if p.exists() else []


def importer(dossier, sortie, ecart_fourreau: float) -> Path:
    dossier, sortie = Path(dossier), Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    meta = json.loads((dossier / "project-metadata.json").read_text(encoding="utf-8"))
    crs = meta.get("projection") or None

    features = []
    for nom, classe in FICHIERS.items():
        for f in _features(dossier, nom):
            if f.get("geometry"):
                features.append({"type": "Feature", "geometry": f["geometry"],
                                 "properties": {**f["properties"], "classe": classe}})
    fourreaux = unary_union([shape(f["geometry"]) for f in _features(dossier, "Fourreau")])
    for f in _features(dossier, "CableElectrique"):
        nu = shape(f["geometry"]).difference(fourreaux.buffer(ecart_fourreau))
        for g in getattr(nu, "geoms", [nu]):
            if g.geom_type == "LineString" and g.length > 0:
                features.append({"type": "Feature", "geometry": mapping(g),
                                 "properties": {**f["properties"], "classe": "cable_noir"}})
    fc = {"type": "FeatureCollection", "features": features}
    if crs:
        fc["crs"] = {"type": "name", "properties": {"name": crs}}
    (sortie / "verite.geojson").write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")

    # orthos fournies (ortho/ de l'archive) : recopiées pour la préparation et l'atelier
    from .preparation import orthos_de
    orthos = orthos_de(dossier)
    if orthos:
        (sortie / "ortho").mkdir(exist_ok=True)
        for o in orthos:
            shutil.copy2(o, sortie / "ortho" / o.name)

    nuages = sorted((dossier / "nuage").glob("*.laz"))
    if not nuages:
        raise FileNotFoundError(f"Aucun .laz dans {dossier / 'nuage'}")
    las = [laspy.read(str(p)) for p in nuages]
    xyz = np.vstack([np.column_stack([l.x, l.y, l.z]) for l in las])
    rvb = np.vstack([np.column_stack([l.red, l.green, l.blue]) for l in las]).astype(np.uint16)
    entete = laspy.LasHeader(point_format=2, version="1.2")
    entete.offsets = np.floor(xyz.min(axis=0))
    entete.scales = las[0].header.scales     # précision d'origine, pas de gain artificiel
    if crs:
        import pyproj
        entete.add_crs(pyproj.CRS(crs))
    out = laspy.LasData(entete)
    out.x, out.y, out.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out.red, out.green, out.blue = rvb[:, 0], rvb[:, 1], rvb[:, 2]
    out.write(str(sortie / "nuage.laz"))
    compte = {c: sum(f["properties"]["classe"] == c for f in features) for c in [*FICHIERS.values(), "cable_noir"]}
    print(f"  {len(nuages)} nuage(s), {len(xyz):,} points ; objets : {compte}".replace(",", " "))
    return sortie
