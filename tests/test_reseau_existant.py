import numpy as np

from tranchee3d.config import Config
from tranchee3d.nuage import Nuage
from tranchee3d.reseau_existant import raccorder_reseau
from tranchee3d.retour3d import Troncon

ORIGINE = np.array([700000.0, 6600000.0, 100.0])


def _cas(classe_reseau, y_reseau):
    """Câble de x=0 à x=10 à Z local -0,8 ; réseau 2D parallèle à y_reseau, au-delà du bout x=10."""
    xyz = np.random.default_rng(0).uniform([-5, -5, -0.2], [20, 5, 0.2], (1000, 3))   # sol à Z local ~0
    nuage = Nuage(xyz, np.zeros_like(xyz, np.uint8), ORIGINE, "EPSG:2154")
    cable = Troncon(2, np.column_stack([np.linspace(0, 10, 50), np.zeros(50), np.full(50, -0.8)]), id_cable=0)
    ligne = np.array([[16.0, y_reseau], [30.0, y_reseau]]) + ORIGINE[:2]
    reseau = {"features": [{"geometry": {"type": "LineString", "coordinates": ligne.tolist()},
                            "properties": {"classe": classe_reseau}}]}
    fc = raccorder_reseau(reseau, [cable], nuage, Config())
    return {f["properties"]["classe"]: np.array(f["geometry"]["coordinates"]) - ORIGINE for f in fc["features"]}


def test_raccord_souterrain_a_plat_au_z_du_cable():
    r = _cas("bt_souterrain", 0.0)
    assert np.allclose(r["raccord_probable"], [[10, 0, -0.8], [16, 0, -0.8]], atol=1e-3)
    assert np.allclose(r["bt_souterrain"][:, 2], -0.8)


def test_raccord_aerien_par_poteau_ajoute():
    cfg = Config()
    r = _cas("bt_aerien", 0.0)
    raccord, poteau = r["raccord_probable"], r["poteau_ajoute"]
    assert np.allclose(raccord[:2], [[10, 0, -0.8], [16, 0, -0.8]], atol=1e-3)
    assert np.allclose(raccord[2, :2], [16, 0]) and abs(raccord[2, 2] - cfg.hauteur_aerien) < 0.2
    assert np.allclose(poteau[:, :2], [[16, 0], [16, 0]]) and abs(poteau[1, 2] - poteau[0, 2] - cfg.hauteur_aerien) < 1e-6


def test_reseau_trop_loin_pas_de_raccord():
    r = _cas("bt_souterrain", 40.0)
    assert "raccord_probable" not in r and "bt_souterrain" in r
