"""Tests rapides des briques géométriques, sur des données synthétiques (aucun fichier)."""
import json

import numpy as np
import pytest

from tranchee3d.config import Config
from tranchee3d.nuage import Nuage
from tranchee3d.ortho import generer_rasters
from tranchee3d.raccordement import raccorder
from tranchee3d.retour3d import Troncon
from tranchee3d.tranchee import detecter_tranchee

CFG = Config()


def _nuage_tranchee(largeur=0.4, profondeur=0.5, pas=0.01):
    """Sol plat 4 m x 4 m à Z = 10, tranchée de `largeur` m le long de Y, creusée de `profondeur`."""
    g = np.arange(0, 4, pas)
    x, y = np.meshgrid(g, g)
    x, y = x.ravel(), y.ravel()
    z = np.where(np.abs(x - 2) < largeur / 2, 10 - profondeur, 10.0)
    return Nuage(np.column_stack([x, y, z]), np.full((len(x), 3), 120, np.uint8), np.zeros(3))


def test_ortho_garde_le_point_le_plus_haut():
    n = Nuage(np.array([[0.5, 0.5, 1.0], [0.5, 0.5, 3.0], [0.5, 0.5, 2.0]]),
              np.array([[10, 0, 0], [30, 0, 0], [20, 0, 0]], np.uint8), np.zeros(3))
    r = generer_rasters(n, CFG)
    lig, col = r.xy_vers_px(0.5, 0.5)
    assert r.hauteur[lig, col] == 3.0 and r.rvb[lig, col, 0] == 30


def test_tranchee_detectee_et_profondeur_relative():
    n = _nuage_tranchee()
    r = generer_rasters(n, CFG)
    t = detecter_tranchee(r, CFG)
    lig, col = r.xy_vers_px(2.0, 2.0)          # milieu de la tranchée
    assert t.fond[lig, col] and t.zone[lig, col]
    assert t.hauteur_relative[lig, col] == pytest.approx(-0.5, abs=0.03)
    lig, col = r.xy_vers_px(0.5, 2.0)          # sol, loin de la tranchée
    assert not t.fond[lig, col] and abs(t.hauteur_relative[lig, col]) < 0.03


def _troncon(x0, x1, z=0.0, classe=2):
    xs = np.arange(x0, x1, CFG.pas) if x1 > x0 else np.arange(x0, x1, -CFG.pas)
    return Troncon(classe, np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, z)]), "observe", 0.9, 0.02)


def test_raccordement_pont_deduit_sur_un_trou_d_un_metre():
    res = raccorder([_troncon(0, 2), _troncon(3, 5)], CFG)
    origines = [t.origine for t in res]
    assert origines.count("deduit") == 1 and origines.count("observe") == 2
    assert len({t.id_cable for t in res}) == 1                 # un seul câble
    pont = next(t for t in res if t.origine == "deduit")
    assert pont.longueur == pytest.approx(1.0, abs=0.05) and abs(pont.pts[:, 2]).max() < 1e-6


def test_raccordement_trou_court_fusionne_en_observe():
    res = raccorder([_troncon(0, 2), _troncon(2.03, 4)], CFG)
    assert len(res) == 1 and res[0].origine == "observe" and res[0].longueur > 3.9


def test_raccordement_refuse_au_dela_du_trou_max_et_des_classes_differentes():
    assert all(t.origine == "observe" for t in raccorder([_troncon(0, 2), _troncon(9, 11)], CFG))
    assert all(t.origine == "observe" for t in raccorder([_troncon(0, 2), _troncon(3, 5, classe=1)], CFG))


def test_evaluation_parfaite(tmp_path):
    from tranchee3d.evaluation import evaluer
    ligne = {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0, 1], [5, 0, 1.2]]},
             "properties": {"classe": "fourreau_rouge", "origine": "observe"}}
    for nom in ("p", "v"):
        (tmp_path / f"{nom}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [ligne]}))
    b = evaluer(tmp_path / "p.geojson", tmp_path / "v.geojson", CFG)["fourreau_rouge"]
    assert b["dimension"] == "3D" and b["part_verite_retrouvee"] == 1.0 and b["observe"]["ecart_median_m"] == 0.0


def test_config_refuse_un_parametre_inconnu(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("pixel = 0.002\ninconnu = 1\n")
    with pytest.raises(ValueError, match="inconnu"):
        Config.charger(f)
    f.write_text("pixel = 0.002\n")
    assert Config.charger(f).pixel == 0.002


def test_repartition_par_chantier_reproductible():
    from tranchee3d.entrainements import repartir
    noms = [f"c{i}" for i in range(10)]
    r = repartir(noms, 15, 15, graine=0)
    assert len(r["validation"]) == 2 and len(r["test"]) == 2 and len(r["train"]) == 6
    assert set(r["train"]) | set(r["validation"]) | set(r["test"]) == set(noms)   # partition complète, sans doublon
    assert r == repartir(noms, 15, 15, graine=0) and r != repartir(noms, 15, 15, graine=1)
    assert repartir(noms, 0, 0, 0)["train"] == sorted(noms)                        # modèle final : tout en entraînement
    with pytest.raises(ValueError, match="aucun chantier"):
        repartir(noms, 50, 50, 0)


def test_dossiers_de_donnees_bruts_puis_prepares(tmp_path):
    from tranchee3d.entrainements import dossiers_de_donnees
    for c in ("a", "b"):
        (tmp_path / "bruts" / c / "nuage").mkdir(parents=True)
        (tmp_path / "bruts" / c / "nuage" / "x.laz").touch()
    (tmp_path / "prepare" / "a").mkdir(parents=True)
    (tmp_path / "prepare" / "a" / "chantier.npz").touch()
    d = dossiers_de_donnees(tmp_path)
    assert list(d) == ["bruts", "prepare"]                                          # bruts d'abord
    assert d["bruts"] == {"type": "brut", "chantiers": ["a", "b"]}
    assert d["prepare"] == {"type": "prepare", "chantiers": ["a"]}


def test_prolongement_jusqu_au_coffret():
    from tranchee3d.raccordement import raccorder_ouvrages
    from tranchee3d.retour3d import Ponctuel
    cable = _troncon(0, 2)
    cable.id_cable = 0
    coffret = Ponctuel(4, np.array([3.0, 0.0, cable.pts[-1, 2] + 0.7]), 0.0, (2.8, -0.2, 3.2, 0.2))
    loin = Ponctuel(3, np.array([-5.0, 0.0, 0.0]), 0.0, (-5.1, -0.1, -4.9, 0.1))
    res = raccorder_ouvrages([cable], [coffret, loin], CFG)
    assert len(res) == 2
    ajout = res[1]
    assert ajout.origine == "deduit" and ajout.classe == cable.classe and ajout.id_cable == 0
    assert np.allclose(ajout.pts[0], cable.pts[-1]) and np.allclose(ajout.pts[-1], [3.0, 0.0, CFG.hauteur_entree_ouvrage])


def test_fourreau_rouge_visible_garde_malgre_obstacle():
    # surface 30 cm au-dessus du fourreau levé : moitié gauche rouge (fourreau vu contre une paroi), moitié droite grise
    from shapely.geometry import LineString
    from tranchee3d.etiquettes import _retirer_parties_cachees
    from tranchee3d.ortho import Rasters
    rvb = np.full((20, 100, 3), 128, np.uint8)
    rvb[:, :50] = (200, 30, 40)
    r = Rasters(rvb, np.full((20, 100), 0.3, np.float32), 0.0, 0.2, 0.01)
    masque = np.zeros((20, 100), np.uint8)
    masque[8:12] = 2
    _retirer_parties_cachees(masque, [(2, LineString([(0, 0.1, 0), (1, 0.1, 0)]), {})], r, np.zeros(3), CFG)
    assert (masque[8:12, :50] == 2).all() and (masque[8:12, 50:] == 0).all()
