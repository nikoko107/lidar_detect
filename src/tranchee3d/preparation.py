"""Préparation d'un chantier pour l'entraînement : nuage (+ ortho fournie) + GeoJSON -> chantier.npz.

`preparer_bruts` enchaîne import et préparation pour tout un dossier brut Enedis, en sautant les
chantiers déjà préparés avec les mêmes sources et les mêmes réglages.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Config

# réglages qui changent le contenu de chantier.npz : s'ils changent, le chantier est re-préparé
REGLAGES_PREPARATION = [
    "pixel", "bouchage_px", "grille_sol", "fenetre_sol", "centile_sol", "profondeur_min", "surface_min_tranchee",
    "marge_tranchee", "ecart_fourreau", "champ_classe", "valeurs_classes", "tampon_cable_noir", "tampon_fourreau",
    "rayon_boite", "rayon_coffret", "ecart_obstacle", "rouge_visible",
]
EXTENSIONS_ORTHO = ("*.tif", "*.tiff", "*.jp2")


def orthos_de(dossier) -> list[Path]:
    """Orthos fournies d'un chantier (sous-dossier ortho/)."""
    d = Path(dossier) / "ortho"
    return sorted(p for ext in EXTENSIONS_ORTHO for p in d.glob(ext))


def rasters_et_tranchee(chemin_nuage, orthos, cfg: Config):
    """orthos : chemin, liste de chemins ou None. Une ortho fournie remplace les couleurs là où elle couvre."""
    from .nuage import lire_nuage
    from .ortho import appliquer_ortho_fournie, generer_rasters
    from .tranchee import detecter_tranchee

    nuage = lire_nuage(chemin_nuage)
    print(f"  {len(nuage):,} points lus".replace(",", " "))
    orthos = [orthos] if isinstance(orthos, (str, Path)) else orthos or []
    r = generer_rasters(nuage, cfg)
    for o in orthos:
        appliquer_ortho_fournie(r, o, nuage.origine)
    t = detecter_tranchee(r, cfg)
    part = 100 * t.zone.mean()
    print(f"  ortho {r.forme[1]} x {r.forme[0]} px ({r.source_ortho}), zone d'intérêt : {part:.0f} % de l'image")
    return nuage, r, t


def preparer(chemin_nuage, chemin_geojson, sortie, orthos, cfg: Config) -> Path:
    from .etiquettes import lire_geojson, rasteriser
    from .export import image_controle
    from .tranchee import tenseur_entree

    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    nuage, r, t = rasters_et_tranchee(chemin_nuage, orthos, cfg)
    masque = rasteriser(lire_geojson(chemin_geojson, cfg), r, nuage.origine, cfg)
    hors_zone = int(((masque > 0) & ~t.zone).sum())
    if hors_zone:
        print(f"  attention : {hors_zone} px étiquetés hors de la zone d'intérêt (marge trop faible ?)")
    np.savez_compressed(sortie / "chantier.npz", entree=tenseur_entree(r, t).astype(np.float16),
                        masque=masque, zone=t.zone)
    image_controle(sortie / "controle_tranchee.png", r.rvb, zone=t.zone)
    image_controle(sortie / "controle_etiquettes.png", r.rvb, classes=masque, zone=t.zone)
    print(f"  -> {sortie}/chantier.npz + images de contrôle")
    return sortie / "chantier.npz"


def chantiers_bruts(bruts) -> list[Path]:
    """Sous-dossiers d'un dossier brut Enedis qui contiennent un nuage (nuage/*.laz)."""
    return sorted(d for d in Path(bruts).iterdir() if d.is_dir() and any((d / "nuage").glob("*.laz")))


def _signature(chantier: Path, cfg: Config) -> dict:
    sources = [p for sous in ("nuage", "recolement", "ortho") for p in (chantier / sous).glob("*") if p.is_file()]
    sources.append(chantier / "project-metadata.json")
    return {"reglages": {k: getattr(cfg, k) for k in REGLAGES_PREPARATION},
            "sources": {p.relative_to(chantier).as_posix(): p.stat().st_mtime for p in sources if p.exists()}}


def preparer_bruts(bruts, donnees, prepare, cfg: Config) -> list[str]:
    """Importe et prépare chaque chantier du dossier brut ; retourne leurs noms."""
    from .rpd import importer

    noms = []
    for chantier in chantiers_bruts(bruts):
        n = chantier.name
        noms.append(n)
        cible, fiche = Path(prepare) / n, Path(prepare) / n / "preparation.json"
        signature = _signature(chantier, cfg)
        if (cible / "chantier.npz").exists() and fiche.exists() and json.loads(fiche.read_text(encoding="utf-8")) == signature:
            print(f"Chantier {n} : déjà préparé")
            continue
        print(f"Chantier {n} : import et préparation")
        importe = importer(chantier, Path(donnees) / n, cfg.ecart_fourreau)
        orthos = orthos_de(importe)
        if orthos:
            print(f"  ortho fournie : {', '.join(o.name for o in orthos)}")
        preparer(importe / "nuage.laz", importe / "verite.geojson", cible, orthos, cfg)
        fiche.write_text(json.dumps(signature), encoding="utf-8")
    if not noms:
        raise ValueError(f"Aucun chantier (sous-dossier avec nuage/*.laz) dans {bruts}")
    return noms
