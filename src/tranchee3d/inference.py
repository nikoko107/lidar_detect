"""Inférence CPU avec ONNX Runtime, par tuiles avec recouvrement."""
from __future__ import annotations

import numpy as np

from .config import CLASSES, Config


def predire(entree: np.ndarray, zone: np.ndarray, chemin_onnx, cfg: Config, lot=4):
    """entree (4, H, W) -> (classes uint8 (H, W), confiance float32 (H, W))."""
    import json
    from pathlib import Path
    import onnxruntime as ort

    contrat = Path(chemin_onnx).with_suffix(".json")
    if contrat.exists():
        attendu = {"hauteur_fond": 0.0, **json.loads(contrat.read_text(encoding="utf-8"))}   # anciens modèles : nuage entier
        ecarts = {k: (attendu[k], getattr(cfg, k)) for k in ("pixel", "fenetre_sol", "centile_sol", "hauteur_fond")
                  if k in attendu and attendu[k] != getattr(cfg, k)}
        if ecarts:
            print(f"  attention : le modèle a été entraîné avec {ecarts} (valeur modèle, valeur actuelle) ; "
                  "l'ortho ne correspond plus à ce qu'il a appris.")
    session = ort.InferenceSession(str(chemin_onnx), providers=["CPUExecutionProvider"])
    t = session.get_inputs()[0].shape[2]
    t = t if isinstance(t, int) else cfg.tuile
    pas = t - cfg.recouvrement
    _, H, W = entree.shape

    # complète pour que la grille de tuiles couvre toute l'image
    Hp = max(t, int(np.ceil((H - t) / pas)) * pas + t) if H > t else t
    Wp = max(t, int(np.ceil((W - t) / pas)) * pas + t) if W > t else t
    x = np.zeros((4, Hp, Wp), np.float32)
    x[:, :H, :W] = entree
    z = np.zeros((Hp, Wp), bool)
    z[:H, :W] = zone

    # fenêtre de pondération : les bords de tuile comptent moins
    f = np.hanning(t + 2)[1:-1].astype(np.float32)
    poids = np.outer(f, f) + 1e-3
    proba = np.zeros((len(CLASSES), Hp, Wp), np.float32)
    somme = np.zeros((Hp, Wp), np.float32)

    coins = [(l, c) for l in range(0, Hp - t + 1, pas) for c in range(0, Wp - t + 1, pas)
             if z[l:l + t, c:c + t].any()]
    for i in range(0, len(coins), lot):
        groupe = coins[i:i + lot]
        logits = session.run(None, {"entree": np.stack([x[:, l:l + t, c:c + t] for l, c in groupe])})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        for (l, c), pi in zip(groupe, p):
            proba[:, l:l + t, c:c + t] += pi * poids
            somme[l:l + t, c:c + t] += poids

    vu = somme > 0
    proba[:, vu] /= somme[vu]
    proba[0, ~vu] = 1.0
    classes = proba.argmax(0).astype(np.uint8)[:H, :W]
    confiance = proba.max(0)[:H, :W]
    classes[~zone] = 0
    return classes, confiance
