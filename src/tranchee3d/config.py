"""Paramètres de la chaîne. Tout seuil réglable est ici, pas dans le code."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

CLASSES = ["fond", "cable_noir", "fourreau_rouge", "boite_jonction", "coffret"]
LINEAIRES = (1, 2)
PONCTUELS = (3, 4)


@dataclass
class Config:
    """Tous les réglages de la chaîne, surchargés par un TOML (`--config`). Aucune constante ailleurs.
    Les valeurs qui changent le contenu de chantier.npz sont listées dans preparation.REGLAGES_PREPARATION."""
    # --- Rasters ---
    pixel: float = 0.004            # taille de pixel de l'ortho (m)
    bouchage_px: int = 3            # trous comblés jusqu'à cette distance (px)

    # --- Tranchée ---
    grille_sol: float = 0.05        # résolution de travail pour le sol (m)
    fenetre_sol: float = 2.0        # fenêtre d'estimation du sol (m), > largeur de tranchée x 2
    centile_sol: float = 60.0       # centile des hauteurs retenu comme sol
    profondeur_min: float = 0.15    # profondeur minimale pour être « tranchée » (m)
    surface_min_tranchee: float = 0.5   # m²
    marge_tranchee: float = 0.6     # bande conservée autour (m), pour les coffrets

    # --- Étiquettes (GeoJSON -> masques) ---
    ecart_fourreau: float = 0.05    # import RPD : câble à moins de cette distance d'un fourreau = sous fourreau (m)
    champ_classe: str = "classe"    # propriété GeoJSON portant la classe
    valeurs_classes: dict = field(default_factory=lambda: {
        "cable_noir": 1, "fourreau_rouge": 2, "boite_jonction": 3, "coffret": 4})
    tampon_cable_noir: float = 0.015    # demi-largeur rastérisée (m)
    tampon_fourreau: float = 0.03
    rayon_boite: float = 0.10
    rayon_coffret: float = 0.18
    ecart_obstacle: float = 0.10        # surface au-dessus du câble levé => câble caché, étiquette retirée (m)
    rouge_visible: list = field(default_factory=lambda: [0.93, 0.05, 0.35, 0.25])  # teinte min, teinte max (passe par 0), saturation min, valeur min (0-1) :
                                        # pixel de fourreau de cette couleur = visible, étiquette gardée malgré ecart_obstacle

    # --- Modèle ---
    tuile: int = 384
    recouvrement: int = 64
    encodeur: str = "resnet18"
    poids_encodeur: str = "imagenet"    # "" pour partir de zéro
    epochs: int = 30
    iterations_par_epoch: int = 200
    lot: int = 8
    taux_apprentissage: float = 1e-3
    part_tuiles_objet: float = 0.7      # part des tuiles tirées sur un objet

    # --- Bruit d'entraînement (0 = désactivé) ---
    bruit_rvb: float = 0.03             # écart-type du bruit gaussien sur les couleurs (0-1)
    bruit_hauteur: float = 0.01         # écart-type du bruit sur la hauteur relative (m)
    decalage_hauteur: float = 0.05      # décalage aléatoire du sol estimé, ± (m)
    echelle: float = 0.15               # changement d'échelle aléatoire, ± (part)
    proba_trou: float = 0.3             # probabilité d'un trou (pixels vides) par tuile
    taille_trou: float = 0.25           # côté maximal du trou (part de la tuile)
    teinte_fourreau: float = 0.03       # variation ± de la teinte du fourreau (part du cercle : 0,03 ≈ ±11°)
    saturation_fourreau: float = 0.25   # variation ± relative de la saturation du fourreau

    # --- Perte de distance (boundary loss) et choix du meilleur modèle ---
    poids_distance: float = 3.0         # 0 = désactivée ; plus grand = écarts au tracé plus pénalisés
    plafond_distance: float = 0.20      # au-delà, la pénalité ne croît plus (m)
    exposant_distance: float = 1.0      # > 1 durcit les écarts lointains, < 1 les écarts proches
    attente_distance: float = 0.3       # part de l'entraînement sans perte de distance (CE + Dice seuls)
    montee_distance: float = 0.3        # part suivante pendant laquelle le poids monte de 0 à sa valeur
    distance_sur_ponctuels: bool = False  # True : la perte de distance s'applique aussi aux boîtes et coffrets
    critere: str = "distance"           # "distance" (écart moyen au tracé) ou "iou"

    # --- Retour en 3D ---
    pas: float = 0.02               # pas d'échantillonnage le long des câbles (m)
    tolerance_surface: float = 0.03  # points gardés sous la surface vue du dessus (m)
    surface_min_px: int = 150       # taille minimale d'un masque
    longueur_min: float = 0.15      # tronçon minimal (m)
    convention: str = "crete"       # "crete" (génératrice supérieure) ou "axe"
    rayon_min: float = 0.006       # bornes du rayon de câble estimé depuis la largeur du masque (m)
    rayon_max: float = 0.08
    part_crete: float = 0.35       # points gardés pour le XY de la crête : à moins de part_crete x rayon sous le Z de crête
    lissage: int = 9                # fenêtre de lissage (nb d'échantillons, impair)
    lissage_z: float = 1.0          # fenêtre de lissage du Z le long du câble (m), contre les vaguelettes ; 0 = désactivé
    ecart_z_aberrant: float = 0.10  # mesure de crête à plus de ça de la médiane locale (objet au-dessus) : écartée (m) ; 0 = désactivé

    # --- Raccordement ---
    trou_observe: float = 0.06      # en dessous, deux tronçons sont fusionnés comme observés (m)
    trou_max: float = 6.0           # au-delà, on ne relie pas (m)
    trou_libre: float = 0.5         # en dessous, liaison sans contrôle d'angle ni de pente (m)
    longueur_eperon: float = 0.5    # tronçon plus court que ça qui double un autre tronçon : supprimé (m)
    ecart_eperon: float = 0.10      # « double » = entièrement à moins de cette distance en plan (m)
    angle_max: float = 35.0         # désalignement maximal (degrés)
    portee_tangente: float = 0.30   # longueur servant à estimer la direction d'une extrémité (m)
    pente_max: float = 0.5          # |dz| / distance maximal
    distance_ouvrage_max: float = 2.0  # bout de câble prolongé jusqu'à une boîte / un coffret à moins de ça, en plan (m)
    hauteur_entree_ouvrage: float = 0.40  # entrée du câble dans l'ouvrage, au-dessus de son point le plus bas (m, médiane du récolement)

    # --- Réseau existant (open data Enedis, aide à la lecture, non évalué) ---
    reseau_existant: bool = True        # False : pas de téléchargement ni de raccord
    marge_reseau: float = 30.0          # rayon de recherche autour de l'emprise du nuage (m)
    distance_raccord_max: float = 15.0  # au-delà, pas de raccord probable (open data précis à 5-10 m) (m)
    hauteur_aerien: float = 8.0         # hauteur du réseau aérien au-dessus du sol (m)
    url_opendata: str = "https://opendata.enedis.fr/api/explore/v2.1/catalog/datasets"
    delai_opendata: float = 20.0        # délai maximal d'une requête (s)

    # --- Évaluation ---
    tolerance_eval: float = 0.05    # m : un point de vérité est « retrouvé » à moins de ça d'une prédiction
    tolerance_ponctuel: float = 0.25  # m : appariement boîtes / coffrets en plan
    pas_eval: float = 0.01          # m : densification des lignes avant comparaison

    @classmethod
    def charger(cls, chemin: str | Path | None) -> "Config":
        """Config par défaut, ou surchargée par le TOML (clés inconnues refusées : protège des fautes de frappe)."""
        if chemin is None:
            return cls()
        donnees = tomllib.loads(Path(chemin).read_text(encoding="utf-8"))
        connus = {f.name for f in fields(cls)}
        inconnus = set(donnees) - connus
        if inconnus:
            raise ValueError(f"Paramètres inconnus dans {chemin} : {sorted(inconnus)}")
        return cls(**donnees)
