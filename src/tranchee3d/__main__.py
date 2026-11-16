"""Ligne de commande : python -m tranchee3d <commande> --help"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import Config
from .preparation import rasters_et_tranchee as _rasters_et_tranchee


def cmd_synthetique(a, cfg):
    """Chantiers de test générés (sans données réelles)."""
    from .synthetique import generer
    for i in range(a.nombre):
        d = generer(Path(a.sortie) / f"chantier_{a.graine + i:02d}", graine=a.graine + i, longueur=a.longueur)
        print(f"Chantier synthétique : {d}")


def cmd_importer(a, cfg):
    """Dossiers bruts Enedis -> donnees/<chantier>/{nuage.laz, verite.geojson}."""
    from .rpd import importer
    for d in a.dossiers:
        print(f"Import de {d}")
        importer(d, Path(a.sortie) / Path(d).name, cfg.ecart_fourreau)


def cmd_preparer(a, cfg):
    """Nuage (+ ortho) + GeoJSON -> fichier d'entraînement et images de contrôle."""
    from .preparation import preparer
    preparer(a.nuage, a.geojson, a.sortie, a.ortho, cfg)


def cmd_entrainer(a, cfg):
    """Entraînement par chantier : --validation choisit l'epoch, --test mesure une fois à la fin, --bruts prépare avant."""
    from .modele import entrainer
    fichiers = sorted(Path(a.donnees).glob("*/chantier.npz"))
    if a.bruts:   # dossier brut Enedis : import + préparation (cache), puis ses seuls chantiers
        from .preparation import preparer_bruts
        noms = set(preparer_bruts(a.bruts, a.importes, a.donnees, cfg))
        fichiers = sorted(f for f in Path(a.donnees).glob("*/chantier.npz") if f.parent.name in noms)
    if not fichiers:
        raise SystemExit(f"Aucun chantier préparé dans {a.donnees}")
    inconnus = (set(a.validation or []) | set(a.test or [])) - {f.parent.name for f in fichiers}
    if inconnus:
        raise SystemExit(f"Chantiers inconnus dans {a.donnees} : {sorted(inconnus)}")
    val = [f for f in fichiers if f.parent.name in set(a.validation or [])]
    test = [f for f in fichiers if f.parent.name in set(a.test or [])]
    train = [f for f in fichiers if f not in val and f not in test]
    if not val:
        print("Aucun chantier de validation : la dernière epoch est gardée.")
    entrainer(train, val, a.sortie, cfg, graine=a.graine, reprendre=a.reprendre, fichiers_test=test)


def cmd_predire(a, cfg):
    """Nuage (+ ortho) -> GeoJSON 3D + visualiseur."""
    from .export import ecrire_geojson, image_controle, vers_geojson
    from .inference import predire
    from .raccordement import raccorder, raccorder_ouvrages
    from .reseau_existant import reseau_existant
    from .retour3d import extraire
    from .tranchee import tenseur_entree
    from .visualiseur import construire

    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nuage, r, t = _rasters_et_tranchee(a.nuage, a.ortho, cfg)
    t1 = time.time()
    classes, confiance = predire(tenseur_entree(r, t), t.zone, a.modele, cfg)
    t2 = time.time()
    troncons, ponctuels = extraire(classes, confiance, r, nuage, cfg)
    troncons = raccorder_ouvrages(raccorder(troncons, cfg), ponctuels, cfg)
    t3 = time.time()

    fc = vers_geojson(troncons, ponctuels, nuage.origine, nuage.crs)
    ecrire_geojson(sortie / "resultat.geojson", fc)
    image_controle(sortie / "controle_prediction.png", r.rvb, classes=classes, zone=t.zone)
    existant = reseau_existant(nuage, troncons, sortie, cfg)
    if existant is not None:
        ecrire_geojson(sortie / "reseau_existant.geojson", existant)
    verite = json.load(open(a.verite, encoding="utf-8")) if a.verite else None
    construire(sortie / "visualiseur.html", nuage, fc, verite, max_points=a.max_points,
               titre=Path(a.nuage).parent.name or "Tranchée 3D", existant=existant)

    obs = sum(x.longueur for x in troncons if x.origine == "observe")
    ded = sum(x.longueur for x in troncons if x.origine == "deduit")
    print(f"  câbles : {obs:.2f} m observés, {ded:.2f} m déduits ; ponctuels : {len(ponctuels)}")
    print(f"  temps : rasters {t1 - t0:.1f} s, inférence CPU {t2 - t1:.1f} s, 3D {t3 - t2:.1f} s")
    print(f"  -> {sortie}/resultat.geojson, visualiseur.html, controle_prediction.png")


def cmd_evaluer(a, cfg):
    """Compare un resultat.geojson à la vérité : écarts 3D, longueur retrouvée, ponctuels (JSON sur stdout)."""
    from .evaluation import evaluer
    print(json.dumps(evaluer(a.prediction, a.verite, cfg), indent=2, ensure_ascii=False))


def cmd_visualiser(a, cfg):
    """Page HTML 3D autonome : nuage + prédiction + vérité."""
    from .nuage import lire_nuage
    from .visualiseur import construire
    charger = lambda c: json.load(open(c, encoding="utf-8")) if c else None
    vide = {"type": "FeatureCollection", "features": []}
    chemin = construire(a.sortie, lire_nuage(a.nuage), charger(a.prediction) or vide, charger(a.verite),
                        max_points=a.max_points)
    print(f"Visualiseur : {chemin}")


def cmd_atelier(a, cfg):
    """Serveur web local : atelier de débogage (/) et page d'entraînement (/entrainement)."""
    from .serveur import servir
    servir(a.donnees, a.modeles, a.sortie, cfg, port=a.port, max_points=a.max_points, prepare=a.prepare, hote=a.hote)


def main(argv=None):
    """Point d'entrée : `python -m tranchee3d [--config x.toml] <commande> ...`."""
    p = argparse.ArgumentParser(prog="tranchee3d", description=__doc__)
    p.add_argument("--config", help="fichier TOML de paramètres (voir config.exemple.toml)")
    s = p.add_subparsers(dest="commande", required=True)

    c = s.add_parser("synthetique", help="génère des chantiers de test")
    c.add_argument("sortie"); c.add_argument("--nombre", type=int, default=1)
    c.add_argument("--graine", type=int, default=0); c.add_argument("--longueur", type=float, default=14.0)
    c.set_defaults(f=cmd_synthetique)

    c = s.add_parser("importer", help="chantiers Enedis (nuage/ + recolement/) -> nuage.laz + verite.geojson")
    c.add_argument("dossiers", nargs="+"); c.add_argument("sortie")
    c.set_defaults(f=cmd_importer)

    c = s.add_parser("preparer",help="prépare un chantier annoté pour l'entraînement")
    c.add_argument("nuage"); c.add_argument("geojson"); c.add_argument("sortie")
    c.add_argument("--ortho", help="ortho GeoTIFF fournie (sinon générée depuis le nuage)")
    c.set_defaults(f=cmd_preparer)

    c = s.add_parser("entrainer", help="entraîne le modèle et l'exporte en ONNX")
    c.add_argument("donnees", help="dossier contenant un sous-dossier par chantier préparé")
    c.add_argument("sortie"); c.add_argument("--validation", nargs="*", help="noms des chantiers de validation")
    c.add_argument("--graine", type=int, default=0)
    c.add_argument("--bruts", help="dossier brut Enedis : chaque chantier est importé et préparé dans <donnees> avant")
    c.add_argument("--importes", default="donnees", help="avec --bruts : où écrire nuage.laz et verite.geojson")
    c.add_argument("--test", nargs="*", help="chantiers de test : mesurés une fois, à la fin, avec le modèle retenu")
    c.add_argument("--reprendre", help="poids .pt de départ (poursuite d'un entraînement)")
    c.set_defaults(f=cmd_entrainer)

    c = s.add_parser("predire", help="nuage -> GeoJSON 3D + visualiseur")
    c.add_argument("nuage"); c.add_argument("modele", help="fichier .onnx"); c.add_argument("sortie")
    c.add_argument("--ortho"); c.add_argument("--verite", help="GeoJSON de référence, affiché dans le visualiseur")
    c.add_argument("--max-points", type=int, default=1_500_000)
    c.set_defaults(f=cmd_predire)

    c = s.add_parser("evaluer", help="compare un résultat à la vérité terrain")
    c.add_argument("prediction"); c.add_argument("verite")
    c.set_defaults(f=cmd_evaluer)

    c = s.add_parser("visualiser", help="construit le visualiseur à partir de fichiers existants")
    c.add_argument("nuage"); c.add_argument("sortie")
    c.add_argument("--prediction"); c.add_argument("--verite")
    c.add_argument("--max-points", type=int, default=1_500_000)
    c.set_defaults(f=cmd_visualiser)

    c = s.add_parser("atelier", help="page web locale : chaque étape de la chaîne dans un onglet")
    c.add_argument("--donnees", default="donnees"); c.add_argument("--modeles", default="modeles")
    c.add_argument("--prepare", default="prepare", help="chantiers préparés, pour l'entraînement")
    c.add_argument("--sortie", default="sortie/atelier"); c.add_argument("--port", type=int, default=8765)
    c.add_argument("--hote", default="127.0.0.1", help="adresse d'écoute")
    c.add_argument("--max-points", type=int, default=1_500_000)
    c.set_defaults(f=cmd_atelier)

    a = p.parse_args(argv)
    a.f(a, Config.charger(a.config))


if __name__ == "__main__":
    main()
