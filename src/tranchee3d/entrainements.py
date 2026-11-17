"""Gestion des entraînements depuis la page web : répartition des chantiers, lancement de
`entrainer` dans un processus séparé, suivi par son historique.jsonl, anti-veille macOS.
Gestion des processus par `psutil` : identique sous macOS, Linux et Windows (pas de `ps`, `pgrep`, `kill`).

Le processus est indépendant du serveur : il survit à un redémarrage et est retrouvé par le fichier
modeles/<nom>/processus.pid. Aucune dépendance HTTP ici (voir `serveur.py`).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import psutil

from .config import Config

# paramètres de Config proposés dans le formulaire (les autres gardent leur défaut) et valeurs permises
PARAMETRES_ENTRAINEMENT = [
    "epochs", "iterations_par_epoch", "lot", "taux_apprentissage", "tuile", "encodeur", "part_tuiles_objet",
    "poids_distance", "critere", "bruit_rvb", "bruit_hauteur", "decalage_hauteur", "echelle", "proba_trou",
    "taille_trou", "teinte_fourreau", "saturation_fourreau",
]
CHOIX = {"encodeur": ["resnet18", "resnet34"], "critere": ["distance", "iou"]}
NOM_VALIDE = re.compile(r"[A-Za-z0-9_-]{1,60}")   # nom d'entraînement = nom de dossier sous modeles/
LIGNES_JOURNAL = 15                               # fin du journal renvoyée à la page


def cafeine_disponible() -> bool:
    """`caffeinate` n'existe que sur macOS."""
    return shutil.which("caffeinate") is not None


def _entrainement_vivant(pid: int) -> bool:
    """Le pid existe et c'est bien un entraînement tranchee3d (un pid peut être réattribué)."""
    if pid <= 0:
        return False
    try:
        p = psutil.Process(pid)
        if p.status() == psutil.STATUS_ZOMBIE:
            return False
        commande = " ".join(p.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return "tranchee3d" in commande and "entrainer" in commande


def _arreter_pid(pid: int, delai: float = 10.0, popen: subprocess.Popen | None = None) -> None:
    """Arrêt propre (SIGTERM sous Unix, TerminateProcess sous Windows) du processus et de ses enfants
    (chargeurs de données torch), puis arrêt forcé de ceux qui n'ont pas fini après `delai` secondes.
    Si le processus est notre propre fils (`popen`), c'est Popen qui l'attend, pour garder son code retour."""
    try:
        parent = psutil.Process(pid)
        enfants = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for p in enfants + ([] if popen else [parent]):
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    if popen:
        popen.terminate()
    _, restants = psutil.wait_procs(enfants if popen else enfants + [parent], timeout=delai)
    for p in restants:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    if popen:
        try:
            popen.wait(timeout=delai)
        except subprocess.TimeoutExpired:
            popen.kill()
            popen.wait()


def dossiers_de_donnees(racine: Path = Path(".")) -> dict[str, dict]:
    """Dossiers du projet utilisables pour l'entraînement, bruts d'abord :
    - brut : <dossier>/<chantier>/nuage/*.laz (archives Enedis, importées et préparées au lancement) ;
    - préparé : <dossier>/<chantier>/chantier.npz.
    Cherchés sur deux niveaux de profondeur depuis `racine`."""
    from .preparation import chantiers_bruts
    trouves = {}
    for motif in ("*/*/nuage", "*/*/*/nuage"):
        for r in {p.parent.parent for p in racine.glob(motif)}:
            if noms := [c.name for c in chantiers_bruts(r)]:
                trouves[r.relative_to(racine).as_posix() if racine != Path(".") else str(r)] = {"type": "brut", "chantiers": noms}
    for motif in ("*/*/chantier.npz", "*/*/*/chantier.npz"):
        for r in {p.parent.parent for p in racine.glob(motif)}:
            cle = r.relative_to(racine).as_posix() if racine != Path(".") else str(r)
            trouves.setdefault(cle, {"type": "prepare", "chantiers": sorted(p.parent.name for p in r.glob("*/chantier.npz"))})
    return dict(sorted(trouves.items(), key=lambda kv: (kv[1]["type"] != "brut", kv[0])))


def repartir(chantiers: list[str], pct_validation: float, pct_test: float, graine: int) -> dict:
    """Répartition par chantier entier (jamais par tuile), tirée au hasard, reproductible avec la graine.
    Un pourcentage non nul donne au moins un chantier ; il doit en rester pour l'entraînement."""
    if not (0 <= pct_validation <= 100 and 0 <= pct_test <= 100):
        raise ValueError("Pourcentages entre 0 et 100.")
    noms = list(chantiers)
    np.random.default_rng(int(graine)).shuffle(noms)
    nb = lambda pct: 0 if pct <= 0 else max(1, round(len(noms) * pct / 100))
    nv, nt = nb(pct_validation), nb(pct_test)
    if nv + nt >= len(noms):
        raise ValueError(f"Il ne reste aucun chantier pour l'entraînement ({len(noms)} chantiers).")
    return {"train": sorted(noms[nv + nt:]), "validation": sorted(noms[:nv]), "test": sorted(noms[nv:nv + nt])}


class Entrainements:
    """Un entraînement à la fois, lancé comme sous-processus `python -m tranchee3d entrainer …`."""

    def __init__(self, prepare: Path, modeles: Path):
        self.prepare, self.modeles = prepare, modeles
        self.processus, self.pid, self.nom = None, None, None
        # un entraînement lancé avant un redémarrage du serveur continue : on le retrouve par son pid
        for f in modeles.glob("**/processus.pid"):
            pid = int(f.read_text(encoding="utf-8").strip() or 0)
            if _entrainement_vivant(pid):
                self.pid, self.nom = pid, f.parent.relative_to(modeles).as_posix()

    # ------------------------------------------------------------- état
    def en_cours(self) -> bool:
        """Vrai si le processus d'entraînement suivi tourne encore."""
        if self.processus is not None:
            return self.processus.poll() is None
        return self.pid is not None and _entrainement_vivant(self.pid)

    def options(self) -> dict:
        """Ce que le formulaire propose : dossiers de données, poids de reprise, paramètres et défauts."""
        defauts = {f.name: getattr(Config(), f.name) for f in fields(Config) if f.name in PARAMETRES_ENTRAINEMENT}
        dossiers = dossiers_de_donnees()
        return {"dossiers": [{"chemin": k, "type": v["type"], "chantiers": len(v["chantiers"])} for k, v in dossiers.items()],
                "dossier_defaut": next(iter(dossiers), None),
                "poids": [p.as_posix() for p in sorted(self.modeles.glob("**/meilleur.pt"))],
                "parametres": [{"nom": n, "defaut": defauts[n], "choix": CHOIX.get(n)} for n in PARAMETRES_ENTRAINEMENT]}

    def repartition(self, dossier: str, pct_validation: float, pct_test: float, graine: int) -> dict:
        """Aperçu de la répartition pour un dossier de données connu (même tirage qu'au lancement)."""
        dossiers = dossiers_de_donnees()
        if dossier not in dossiers:
            raise ValueError("Dossier de données inconnu.")
        return repartir(dossiers[dossier]["chantiers"], pct_validation, pct_test, graine)

    # ------------------------------------------------------------- lancement
    def _config_toml(self, parametres: dict) -> str:
        """Valide les paramètres du formulaire (type du défaut, choix permis, pas de négatif) -> TOML."""
        lignes = []
        for p in self.options()["parametres"]:
            v = type(p["defaut"])(parametres.get(p["nom"], p["defaut"]))   # mauvais type -> ValueError
            if p["choix"] and v not in p["choix"]:
                raise ValueError(f"{p['nom']} : valeur non permise.")
            if isinstance(v, (int, float)) and v < 0:
                raise ValueError(f"{p['nom']} : valeur négative.")
            lignes.append(f'{p["nom"]} = {json.dumps(v)}')
        return "\n".join(lignes) + "\n"

    def lancer(self, d: dict) -> str:
        """Valide le formulaire, écrit modeles/<nom>/config.toml et lance `entrainer`. Retourne le nom.
        Un dossier brut est passé en --bruts : import et préparation dans `prepare` avant l'entraînement."""
        if self.en_cours():
            raise ValueError(f"L'entraînement « {self.nom} » est déjà en cours.")
        nom = d.get("nom") or time.strftime("run_%Y%m%d_%H%M")
        if not NOM_VALIDE.fullmatch(nom):
            raise ValueError("Nom : lettres, chiffres, - et _ seulement.")
        dossier = self.modeles / nom
        if (dossier / "historique.jsonl").exists():
            raise ValueError(f"Le dossier {dossier} contient déjà un entraînement.")
        donnees = d.get("dossier", "")
        graine = int(d.get("graine", 0))
        rep = self.repartition(donnees, float(d.get("pct_validation", 0)), float(d.get("pct_test", 0)), graine)
        reprendre = d.get("reprendre") or None
        if reprendre and reprendre not in self.options()["poids"]:
            raise ValueError("Poids de départ inconnus.")
        toml = self._config_toml(d.get("parametres", {}))

        dossier.mkdir(parents=True, exist_ok=True)
        (dossier / "config.toml").write_text(toml, encoding="utf-8")
        brut = dossiers_de_donnees()[donnees]["type"] == "brut"
        commande = [sys.executable, "-m", "tranchee3d", "--config", str(dossier / "config.toml"), "entrainer",
                    str(self.prepare) if brut else donnees, str(dossier), "--graine", str(graine)]
        if brut:
            commande += ["--bruts", donnees]
        if rep["validation"]:
            commande += ["--validation", *rep["validation"]]
        if rep["test"]:
            commande += ["--test", *rep["test"]]
        if reprendre:
            commande += ["--reprendre", reprendre]
        # le fils garde sa propre copie du descripteur : on peut fermer le nôtre juste après Popen
        with open(dossier / "entrainement.log", "w", encoding="utf-8") as journal:
            self.processus = subprocess.Popen(commande, stdout=journal, stderr=subprocess.STDOUT,
                                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.pid, self.nom = self.processus.pid, nom
        (dossier / "processus.pid").write_text(str(self.pid), encoding="utf-8")
        return nom

    def arreter(self) -> None:
        """Arrête le processus d'entraînement et ses enfants (meilleur.pt déjà écrit reste utilisable)."""
        if self.en_cours():
            _arreter_pid(self.pid, popen=self.processus)

    # ------------------------------------------------------------- anti-veille (macOS)
    def _pid_cafeine(self) -> int | None:
        """caffeinate attaché à l'entraînement en cours, même lancé avant un redémarrage du serveur."""
        if not cafeine_disponible() or not self.en_cours():
            return None
        for p in psutil.process_iter(["name", "cmdline"]):
            commande = p.info["cmdline"] or []
            if p.info["name"] == "caffeinate" and commande[-2:] == ["-w", str(self.pid)]:
                return p.pid
        return None

    def cafeine_active(self) -> bool:
        """Un caffeinate est-il attaché à l'entraînement en cours ?"""
        return self._pid_cafeine() is not None

    def basculer_cafeine(self, actif: bool) -> None:
        """Empêche la mise en veille tant que l'entraînement tourne (`caffeinate -w pid` s'arrête avec lui)."""
        if not actif:
            if pid := self._pid_cafeine():
                _arreter_pid(pid)
            return
        if not self.en_cours():
            raise ValueError("Aucun entraînement en cours.")
        if not cafeine_disponible():
            raise ValueError("Anti-veille indisponible : caffeinate n'existe que sur macOS.")
        if not self.cafeine_active():
            subprocess.Popen(["caffeinate", "-i", "-m", "-s", "-w", str(self.pid)])

    # ------------------------------------------------------------- suivi
    def liste(self) -> list[dict]:
        """Entraînements connus (dossiers avec historique ou pid), le plus récent d'abord."""
        dossiers = {p.parent for motif in ("**/historique.jsonl", "**/processus.pid") for p in self.modeles.glob(motif)}
        runs = []
        for d in sorted(dossiers, key=lambda d: d.stat().st_mtime, reverse=True):
            nom = d.relative_to(self.modeles).as_posix() if d != self.modeles else "."
            runs.append({"nom": nom, "en_cours": self.en_cours() and nom == self.nom})
        return runs

    def suivi(self, nom: str) -> dict:
        """Historique JSONL d'un entraînement et fin de son journal, pour les graphiques de la page."""
        dossier = (self.modeles / nom).resolve()
        if not dossier.is_relative_to(self.modeles.resolve()) or not dossier.is_dir():
            raise ValueError("Entraînement inconnu.")
        h = dossier / "historique.jsonl"   # absent pendant l'import et la préparation des données
        lignes = [json.loads(l) for l in h.read_text(encoding="utf-8").splitlines() if l] if h.exists() else []
        journal = dossier / "entrainement.log"
        fin = journal.read_text(encoding="utf-8", errors="replace").splitlines()[-LIGNES_JOURNAL:] if journal.exists() else []
        en_cours = self.en_cours() and nom == self.nom
        code = None if en_cours or self.nom != nom or self.processus is None else self.processus.returncode
        return {"nom": nom, "en_cours": en_cours, "code_retour": code, "lignes": lignes, "journal": fin}
