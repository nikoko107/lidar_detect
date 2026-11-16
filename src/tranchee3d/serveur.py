"""Serveur HTTP local de l'atelier (/) et de la page d'entraînement (/entrainement).

Ne contient que les routes : l'exécution d'un nuage est dans `atelier.py`, les entraînements dans
`entrainements.py`. N'écoute que sur 127.0.0.1 par défaut (`--hote` pour changer).
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .atelier import ETAPES, Atelier
from .config import Config
from .entrainements import Entrainements, cafeine_disponible

ACTIFS = Path(__file__).parent / "assets"
PAGES = {"/": "atelier.html", "/entrainement": "entrainement.html"}
TYPES = {".png": "image/png", ".js": "text/javascript", ".html": "text/html; charset=utf-8"}


def _gestionnaire(atelier: Atelier, entrainements: Entrainements):
    """Fabrique la classe de requêtes liée aux deux objets (BaseHTTPRequestHandler est instancié par requête)."""

    class Gestionnaire(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass    # pas de journal d'accès : une requête toutes les 3 s pour le suivi

        # --------------------------------------------------------- réponses
        def _envoyer(self, corps: bytes, type_="application/json", code=200):
            self.send_response(code)
            self.send_header("Content-Type", type_)
            self.send_header("Content-Length", str(len(corps)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(corps)

        def _json(self, objet, code=200):
            self._envoyer(json.dumps(objet, ensure_ascii=False).encode(), code=code)

        def _corps_json(self) -> dict:
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        # --------------------------------------------------------- GET
        def do_GET(self):
            url = urlparse(self.path)
            chemin = unquote(url.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if chemin in PAGES:
                    self._envoyer((ACTIFS / PAGES[chemin]).read_bytes(), TYPES[".html"])
                elif chemin == "/three_bundle.min.js":
                    self._envoyer((ACTIFS / "three_bundle.min.js").read_bytes(), TYPES[".js"])
                # atelier
                elif chemin == "/api/liste":
                    self._json(atelier.liste())
                elif chemin == "/api/etat":
                    self._json({**atelier.etat, "ordre": ETAPES})
                elif chemin.startswith("/fichiers/"):
                    self._fichier(chemin.removeprefix("/fichiers/"))
                # entraînement
                elif chemin == "/api/entrainement/options":
                    self._json(entrainements.options())
                elif chemin == "/api/entrainement/liste":
                    self._json({"runs": entrainements.liste(), "cafeine": entrainements.cafeine_active(),
                                "cafeine_disponible": cafeine_disponible(),
                                "en_cours": entrainements.nom if entrainements.en_cours() else None})
                elif chemin == "/api/entrainement/repartition":
                    self._json(entrainements.repartition(q.get("dossier", ""), float(q.get("validation", 0)),
                                                         float(q.get("test", 0)), int(q.get("graine", 0))))
                elif chemin == "/api/entrainement/suivi":
                    try:
                        self._json(entrainements.suivi(q.get("nom", "")))
                    except ValueError as e:
                        self._json({"erreur": str(e)}, 404)
                else:
                    self._json({"erreur": "introuvable"}, 404)
            except ValueError as e:
                self._json({"erreur": str(e)}, 400)

        def _fichier(self, relatif: str):
            """Fichier d'étape du chantier en cours d'affichage, et rien d'autre : le chemin résolu doit
            rester directement dans ce dossier (pas de `..`, pas de sous-dossier)."""
            if not atelier.etat["dossier"]:
                return self._json({"erreur": "introuvable"}, 404)
            base = (atelier.sortie / atelier.etat["dossier"]).resolve()
            f = (base / relatif).resolve()
            if f.parent != base or not f.is_file():
                return self._json({"erreur": "introuvable"}, 404)
            self._envoyer(f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

        # --------------------------------------------------------- POST
        def do_POST(self):
            chemin = urlparse(self.path).path
            try:
                if chemin == "/api/televerser":
                    taille = int(self.headers.get("Content-Length", 0))
                    self._json({"chemin": atelier.televerser(unquote(self.headers.get("X-Nom", "")), self.rfile, taille)})
                elif chemin == "/api/lancer":
                    d = self._corps_json()
                    atelier.lancer(d["nuage"], d.get("modele") or None, d.get("ortho") or None)
                    self._json({"ok": True})
                elif chemin == "/api/entrainement/lancer":
                    self._json({"nom": entrainements.lancer(self._corps_json())})
                elif chemin == "/api/entrainement/cafeine":
                    entrainements.basculer_cafeine(bool(self._corps_json().get("actif")))
                    self._json({"cafeine": entrainements.cafeine_active()})
                elif chemin == "/api/entrainement/arreter":
                    entrainements.arreter()
                    self._json({"ok": True})
                else:
                    self._json({"erreur": "introuvable"}, 404)
            except (ValueError, KeyError, TypeError) as e:
                self._json({"erreur": str(e)}, 400)

    return Gestionnaire


def servir(donnees, modeles, sortie, cfg: Config, port=8765, max_points=1_500_000, prepare="prepare",
           hote="127.0.0.1") -> None:
    """Serveur multi-fils (une exécution longue ne bloque pas les requêtes d'état).
    Un entraînement lancé avant un redémarrage est retrouvé par son processus.pid."""
    atelier = Atelier(Path(donnees), Path(modeles), Path(sortie), cfg, max_points)
    entrainements = Entrainements(Path(prepare), Path(modeles))
    serveur = ThreadingHTTPServer((hote, port), _gestionnaire(atelier, entrainements))
    print(f"Atelier : http://127.0.0.1:{port}  ·  entraînement : http://127.0.0.1:{port}/entrainement  (Ctrl+C pour arrêter)")
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        pass
