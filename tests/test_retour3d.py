import numpy as np

from tranchee3d.config import Config
from tranchee3d.retour3d import sans_aberrants_z


def test_objet_au_dessus_en_bout_de_troncon_ecarte():
    # 3 mesures prises sur une planche 50 cm au-dessus, en début de tronçon, puis le fourreau à plat
    z = np.r_[[106.55, 106.60, 106.50], np.full(60, 106.05)]
    assert np.allclose(sans_aberrants_z(z, Config()), 106.05)


def test_vraie_remontee_en_bout_conservee():
    # fourreau qui remonte de 30 cm sur le dernier mètre (remontée vers un coffret)
    z = np.r_[np.full(60, 106.0), np.linspace(106.0, 106.3, 50)]
    assert np.array_equal(sans_aberrants_z(z, Config()), z)
