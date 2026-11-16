"""Comparaison d'un GeoJSON prédit à la vérité terrain."""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import shape

from .config import CLASSES, Config


def _densifier(geom, pas) -> np.ndarray:
    lignes = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
    morceaux = []
    for ligne in lignes:
        c = np.array(ligne.coords, dtype=float)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))])
        cible = np.linspace(0, s[-1], max(2, int(s[-1] / pas) + 1))
        morceaux.append(np.column_stack([np.interp(cible, s, c[:, k]) for k in range(c.shape[1])]))
    return np.vstack(morceaux)


def _charger(chemin, champ, pas):
    lignes, points = {}, {}
    for f in json.load(open(chemin, encoding="utf-8"))["features"]:
        classe = f["properties"].get(champ)
        g = shape(f["geometry"])
        if g.geom_type in ("LineString", "MultiLineString"):
            lignes.setdefault(classe, []).append((_densifier(g, pas), f["properties"]))
        elif g.geom_type == "Point":
            points.setdefault(classe, []).append(np.array(g.coords[0], dtype=float))
    return lignes, points


def evaluer(chemin_prediction, chemin_verite, cfg: Config) -> dict:
    """Écarts en mètres. La comparaison est 3D si la vérité terrain porte un Z, sinon 2D."""
    pred_l, pred_p = _charger(chemin_prediction, "classe", cfg.pas_eval)
    vrai_l, vrai_p = _charger(chemin_verite, cfg.champ_classe, cfg.pas_eval)
    tol, bilan = cfg.tolerance_eval, {}

    for classe in CLASSES[1:3]:
        if classe not in vrai_l:
            continue
        v = np.vstack([p for p, _ in vrai_l[classe]])
        dim = 3 if v.shape[1] >= 3 else 2
        res = {"dimension": f"{dim}D", "longueur_verite_m": round(len(v) * cfg.pas_eval, 2)}
        if classe in pred_l:
            arbre_v = cKDTree(v[:, :dim])
            tout = np.vstack([p for p, _ in pred_l[classe]])
            res["part_verite_retrouvee"] = round(float(
                (cKDTree(tout[:, :dim]).query(v[:, :dim])[0] <= tol).mean()), 3)
            for origine in ("observe", "deduit"):
                sel = [p for p, pr in pred_l[classe] if pr.get("origine") == origine]
                if sel:
                    d = arbre_v.query(np.vstack(sel)[:, :dim])[0]
                    res[origine] = {
                        "longueur_m": round(len(d) * cfg.pas_eval, 2),
                        "ecart_median_m": round(float(np.median(d)), 4),
                        "ecart_p95_m": round(float(np.percentile(d, 95)), 4),
                    }
        else:
            res["part_verite_retrouvee"] = 0.0
        bilan[classe] = res

    for classe in CLASSES[3:]:
        v, p = vrai_p.get(classe, []), pred_p.get(classe, [])
        if not v and not p:
            continue
        trouves, ecarts, libres = 0, [], list(range(len(p)))
        for pv in v:
            if not libres:
                break
            d = [np.linalg.norm(p[j][:2] - pv[:2]) for j in libres]
            j = int(np.argmin(d))
            if d[j] <= max(tol, cfg.tolerance_ponctuel):
                trouves += 1
                ecarts.append(d[j])
                libres.pop(j)
        bilan[classe] = {
            "verite": len(v), "predits": len(p), "trouves": trouves,
            "rappel": round(trouves / len(v), 3) if v else None,
            "precision": round(trouves / len(p), 3) if p else None,
            "ecart_plan_median_m": round(float(np.median(ecarts)), 4) if ecarts else None,
        }
    return bilan
