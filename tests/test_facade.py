import numpy as np

from tranchee3d.config import Config
from tranchee3d.facade import ortho_de_face
from tranchee3d.nuage import Nuage


def _chantier(angle_deg=25.0, graine=0):
    """Sol plat + coffret 0,40 x 0,20 x 0,80 m à 0,9 m devant le bout du câble, tourné de `angle_deg`."""
    rng = np.random.default_rng(graine)
    sol = np.column_stack([rng.uniform(0, 4, 200_000), rng.uniform(0, 4, 200_000), rng.normal(0, 0.003, 200_000)])
    a = np.radians(angle_deg)
    largeur, normale = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    centre = np.array([2.0, 2.4])
    n = 30_000
    s, z = rng.uniform(-0.2, 0.2, n), rng.uniform(0, 0.8, n)
    avant = np.column_stack([centre - 0.1 * normale + s[:, None] * largeur, z])        # face tournée vers le câble
    p = rng.uniform(-0.1, 0.1, n // 2)
    cote = np.column_stack([centre + 0.2 * largeur + p[:, None] * normale, z[: n // 2]])
    xyz = np.vstack([sol, avant, cote])
    rvb = np.vstack([np.full((len(sol), 3), 110), np.full((n, 3), 200), np.full((n // 2, 3), 150)]).astype(np.uint8)
    return Nuage(xyz, rvb, np.zeros(3)), largeur, normale


def test_ortho_tournee_face_au_coffret():
    nuage, largeur, normale = _chantier()
    cfg = Config()
    f = ortho_de_face(nuage, np.array([2.0, 1.5, -0.6]), np.array([0.0, 1.0, 0.0]), 0.0, cfg)
    assert f.orientation == "plan"
    assert np.degrees(np.arccos(np.clip(f.visee[:2] @ normale, -1, 1))) < 2      # visée perpendiculaire à la face
    # la face claire (200) apparaît comme un rectangle de 0,40 x 0,80 m
    clair = (f.rvb[..., 0] > 180) & np.isfinite(f.profondeur)
    lig, col = np.nonzero(clair)
    assert abs((np.ptp(col) + 1) * f.pixel - 0.40) < 0.03
    assert abs((np.ptp(lig) + 1) * f.pixel - (0.80 - cfg.face_z_min)) < 0.03
    # retour en 3D : les pixels de la face retombent sur le plan de la face, au centimètre
    xyz = f.px_vers_xyz(lig, col)
    ecart = (xyz[:, :2] - (np.array([2.0, 2.4]) - 0.1 * normale)) @ normale
    assert np.abs(ecart).max() < 0.01


def test_sans_plan_vue_dans_l_axe_du_cable():
    nuage, _, _ = _chantier()
    nuage = Nuage(nuage.xyz[nuage.rvb[:, 0] == 110], nuage.rvb[nuage.rvb[:, 0] == 110], np.zeros(3))   # sol seul
    rng = np.random.default_rng(1)                     # un buisson : volume sans plan
    buisson = np.column_stack([rng.normal(2.0, 0.15, 3000), rng.normal(2.4, 0.15, 3000), rng.uniform(0.1, 0.9, 3000)])
    nuage = Nuage(np.vstack([nuage.xyz, buisson]), np.vstack([nuage.rvb, np.full((3000, 3), 90, np.uint8)]), np.zeros(3))
    f = ortho_de_face(nuage, np.array([2.0, 1.5, -0.6]), np.array([0.0, 1.0, 0.0]), 0.0, Config(face_plan_points_min=2000))
    assert f.orientation == "cable"
    assert np.allclose(f.visee, [0, 1, 0])


def test_rien_au_dessus_du_sol_pas_d_image():
    nuage, _, _ = _chantier()
    sol = nuage.rvb[:, 0] == 110
    assert ortho_de_face(Nuage(nuage.xyz[sol], nuage.rvb[sol], np.zeros(3)), np.array([2.0, 1.5, -0.6]),
                         np.array([0.0, 1.0, 0.0]), 0.0, Config()) is None
