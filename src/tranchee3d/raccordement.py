"""Raccordement des tronçons de câble.

Deux passes :
1. trous très courts (coupure de segmentation, embranchement de squelette) :
   les tronçons sont fusionnés et restent « observés » ;
2. trous plus longs (zone masquée) : un tronçon « déduit » est interpolé entre
   les deux extrémités, par une courbe qui respecte leurs directions.

L'appariement est glouton par coût croissant (distance + désalignement).
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .retour3d import Ponctuel, Troncon


def _tangente_sortante(pts: np.ndarray, bout: int, portee: int) -> np.ndarray:
    k = min(len(pts) - 1, max(1, portee))
    v = pts[0] - pts[k] if bout == 0 else pts[-1] - pts[-1 - k]
    return v / (np.linalg.norm(v) + 1e-12)


def _apparier(troncons, d_max, cfg: Config, avec_angles: bool):
    """Liste de paires ((i, bout_i), (j, bout_j)) retenues, sans réutiliser une extrémité."""
    portee = int(round(cfg.portee_tangente / cfg.pas))
    bouts = [(i, b, t.pts[0 if b == 0 else -1], _tangente_sortante(t.pts, b, portee))
             for i, t in enumerate(troncons) for b in (0, 1)]
    candidats = []
    for a in range(len(bouts)):
        ia, ba, pa, ta = bouts[a]
        for b in range(a + 1, len(bouts)):
            ib, bb, pb, tb = bouts[b]
            if ia == ib or troncons[ia].classe != troncons[ib].classe:
                continue
            v = pb - pa
            d = float(np.linalg.norm(v))
            if d > d_max:
                continue
            cout = d / d_max
            if avec_angles and d > 1e-6:
                u = v / d
                a1 = np.degrees(np.arccos(np.clip(ta @ u, -1, 1)))     # A regarde-t-il vers B ?
                a2 = np.degrees(np.arccos(np.clip(tb @ -u, -1, 1)))    # B regarde-t-il vers A ?
                pente = abs(v[2]) / max(np.linalg.norm(v[:2]), 1e-6)
                # trou court : la proximité suffit (coude, remontée) ; l'angle ne sert qu'à classer
                if d > cfg.trou_libre and (a1 > cfg.angle_max or a2 > cfg.angle_max or pente > cfg.pente_max):
                    continue
                cout += (a1 + a2) / (2 * cfg.angle_max)
            candidats.append((cout, (ia, ba), (ib, bb)))
    pris, paires = set(), []
    for _, ea, eb in sorted(candidats, key=lambda c: c[0]):
        if ea not in pris and eb not in pris:
            pris.update((ea, eb))
            paires.append((ea, eb))
    return paires


def _chainer(n: int, paires):
    """Regroupe les tronçons en chaînes ordonnées : [[(indice, inversé), ...], ...]."""
    lien = {}
    for ea, eb in paires:
        lien[ea], lien[eb] = eb, ea
    vus, chaines = set(), []
    # on part d'abord des tronçons ayant une extrémité libre, puis des boucles éventuelles
    departs = [(i, b) for i in range(n) for b in (0, 1) if (i, b) not in lien]
    departs += [(i, 0) for i in range(n)]
    for i, bout_entree in departs:
        if i in vus:
            continue
        chaine = []
        while i not in vus:
            vus.add(i)
            chaine.append((i, bout_entree == 1))
            suivant = lien.get((i, 1 - bout_entree))
            if suivant is None:
                break
            i, bout_entree = suivant
        chaines.append(chaine)
    return chaines


def _pont(p0, t0, p1, t1_entrant, pas) -> np.ndarray:
    """Courbe de Hermite entre deux extrémités, tangente aux deux tronçons."""
    d = np.linalg.norm(p1 - p0)
    n = max(2, int(round(d / pas)))
    s = np.linspace(0, 1, n + 1)[:, None]
    m0, m1 = t0 * d, t1_entrant * d
    return ((2 * s**3 - 3 * s**2 + 1) * p0 + (s**3 - 2 * s**2 + s) * m0
            + (-2 * s**3 + 3 * s**2) * p1 + (s**3 - s**2) * m1)


def sans_eperons(troncons: list[Troncon], cfg: Config) -> list[Troncon]:
    """Retire les petits tronçons qui doublent un tronçon plus long de même classe (branches
    du squelette) : sinon ils prennent les extrémités et bloquent le raccordement."""
    import shapely
    gardes = []
    for t in sorted(troncons, key=lambda t: -t.longueur):
        points = shapely.points(t.pts[:, :2])
        if t.longueur < cfg.longueur_eperon and any(
                g.classe == t.classe
                and shapely.distance(shapely.linestrings(g.pts[:, :2]), points).max() < cfg.ecart_eperon
                for g in gardes):
            continue
        gardes.append(t)
    return gardes


def raccorder(troncons: list[Troncon], cfg: Config) -> list[Troncon]:
    """Retourne les tronçons observés fusionnés + les tronçons déduits, avec un id_cable commun."""
    if not troncons:
        return []
    troncons = sans_eperons(troncons, cfg)
    oriente = lambda t, inverse: t.pts[::-1] if inverse else t.pts

    # passe 1 : fusion des trous courts
    fusionnes = []
    for chaine in _chainer(len(troncons), _apparier(troncons, cfg.trou_observe, cfg, False)):
        membres = [troncons[i] for i, _ in chaine]
        pts = np.vstack([oriente(troncons[i], inv) for i, inv in chaine])
        longueurs = [m.longueur for m in membres]
        fusionnes.append(Troncon(
            membres[0].classe, pts, "observe",
            float(np.average([m.confiance for m in membres], weights=longueurs)),
            float(np.average([m.rayon for m in membres], weights=longueurs)),
        ))
    fusionnes = [t for t in fusionnes if t.longueur >= cfg.longueur_min]

    # passe 2 : ponts « déduits » à travers les zones masquées
    portee = int(round(cfg.portee_tangente / cfg.pas))
    resultat = []
    chaines = _chainer(len(fusionnes), _apparier(fusionnes, cfg.trou_max, cfg, True))
    for id_cable, chaine in enumerate(chaines):
        precedent = None
        for i, inverse in chaine:
            t = fusionnes[i]
            pts = oriente(t, inverse)
            if precedent is not None:
                t0 = _tangente_sortante(precedent, 1, portee)
                t1 = -_tangente_sortante(pts, 0, portee)
                resultat.append(Troncon(t.classe, _pont(precedent[-1], t0, pts[0], t1, cfg.pas),
                                        "deduit", 0.0, t.rayon, id_cable))
            resultat.append(Troncon(t.classe, pts, "observe", t.confiance, t.rayon, id_cable))
            precedent = pts
    return resultat


def bouts_libres(troncons: list[Troncon], tolerance: float) -> list[tuple[Troncon, int]]:
    """Extrémités (tronçon, 0 ou -1) qui ne touchent aucune autre extrémité du même câble, en plan."""
    bouts = [(t, b) for t in troncons for b in (0, -1)]
    xy = np.array([t.pts[b, :2] for t, b in bouts]).reshape(-1, 2)
    meme = np.array([t.id_cable for t, _ in bouts])
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    d[meme[:, None] != meme[None]] = np.inf
    np.fill_diagonal(d, np.inf)
    return [bouts[i] for i in np.flatnonzero(d.min(axis=1, initial=np.inf) > tolerance)]


def raccorder_ouvrages(troncons: list[Troncon], ponctuels: list[Ponctuel], cfg: Config) -> list[Troncon]:
    """Prolonge chaque bout libre jusqu'à la boîte ou au coffret le plus proche (à moins de distance_ouvrage_max
    en plan) : courbe 3D partant dans la direction du câble, jusqu'à l'entrée de l'objet (z_min + hauteur_entree_ouvrage,
    sans dépasser son dessus), comme le point levé au récolement.
    Le prolongement garde la classe et l'id_cable du câble (continuité du fourreau) et est « deduit »."""
    if not ponctuels:
        return troncons
    portee = int(round(cfg.portee_tangente / cfg.pas))
    objets = np.array([[*p.xyz[:2], min(p.z_min + cfg.hauteur_entree_ouvrage, p.xyz[2])] for p in ponctuels])
    ajouts = []
    for t, b in bouts_libres(troncons, cfg.trou_observe):
        pts = t.pts if b == -1 else t.pts[::-1]
        bout = pts[-1]
        d = np.linalg.norm(objets[:, :2] - bout[:2], axis=1)
        k = int(d.argmin())
        if not cfg.trou_observe < d[k] <= cfg.distance_ouvrage_max:
            continue
        arrivee = (objets[k] - bout) / np.linalg.norm(objets[k] - bout)
        chemin = _pont(bout, _tangente_sortante(pts, 1, portee), objets[k], arrivee, cfg.pas)
        ajouts.append(Troncon(t.classe, chemin, "deduit", 0.0, t.rayon, t.id_cable))
    return troncons + ajouts
