# tranchee3d

Détection 3D de câbles, boîtes de jonction et coffrets dans un nuage de points
colorisé de tranchée ouverte. Détection en 2D sur une ortho, mesure en 3D sur
le nuage. Inférence sur CPU.

## Installation (macOS, Linux, Windows)

Python 3.11 ou plus récent. Environnement virtuel créé **hors du disque FAT** (qui gère mal les liens
symboliques).

macOS / Linux :

```bash
python3 -m venv ~/.venvs/tranchee3d
source ~/.venvs/tranchee3d/bin/activate
python -m pip install -e ".[dev]" pyproj
```

Windows (PowerShell) :

```powershell
python -m venv D:\lidar\.venv
& "D:\lidar\.venv\Scripts\Activate.ps1"
python -m pip install -e ".[dev]" pyproj
```

`pyproj` sert à `importer` (CRS des données Enedis). L'entraînement utilise MPS (Mac Apple Silicon)
ou CUDA s'ils sont disponibles, sinon le CPU ; l'inférence reste toujours sur CPU (ONNX Runtime).
L'atelier se lance avec `python -m tranchee3d atelier` (http://127.0.0.1:8765). L'anti-veille de la
page d'entraînement (`caffeinate`) n'existe que sur macOS.

Les données (`bruts/`, `donnees/`, `prepare*/`, `modeles/`, `sortie/`) ne sont pas versionnées
(voir `.gitignore`).

## Essai sans données réelles

```bash
# 1. quatre chantiers synthétiques (nuage.las + verite.geojson)
python -m tranchee3d synthetique donnees --nombre 4

# 2. préparation de chaque chantier (ortho, tranchée, masques, images de contrôle)
for c in donnees/chantier_*; do
  python -m tranchee3d preparer $c/nuage.las $c/verite.geojson prepare/$(basename $c)
done

# 3. entraînement (MPS si disponible) et export ONNX
python -m tranchee3d entrainer prepare modeles --validation chantier_03

# 4. prédiction CPU sur un chantier : GeoJSON 3D + visualiseur
python -m tranchee3d predire donnees/chantier_03/nuage.las modeles/v1/modele.onnx sortie \
    --verite donnees/chantier_03/verite.geojson

# 5. écarts à la vérité terrain
python -m tranchee3d evaluer sortie/resultat.geojson donnees/chantier_03/verite.geojson

open sortie/visualiseur.html
```

## Avec vos données

Chantiers Enedis (dossier extrait avec `nuage/*.laz`, `recolement/RPD_*_Reco.geojson`,
`project-metadata.json`) : `python -m tranchee3d importer bruts/* donnees` écrit pour chacun
`donnees/<chantier>/nuage.laz` (nuages fusionnés, CRS dans l'en-tête) et `verite.geojson`.

- `preparer` et `predire` acceptent `--ortho chemin.tif` quand une ortho est fournie ;
  sinon l'ortho est générée depuis le nuage.
- Le GeoJSON d'entraînement doit porter la classe dans une propriété. Son nom et ses
  valeurs se règlent dans la configuration (`champ_classe`, `valeurs_classes`).
- Tous les seuils sont dans `config.exemple.toml` : copiez-le, modifiez-le, puis
  `python -m tranchee3d --config ma_config.toml <commande> ...`.

## Perte d'entraînement

Entropie croisée + Dice + **perte de distance** (boundary loss) : la probabilité prédite
est multipliée par la distance signée au tracé d'entraînement, donc une détection loin du
tracé coûte d'autant plus cher. Réglages dans la configuration :

- `poids_distance` (3.0 ; 0 la désactive), `plafond_distance` (0,20 m), `exposant_distance` ;
- `attente_distance` puis `montee_distance` : la perte n'entre en jeu qu'après 30 % de
  l'entraînement, puis monte progressivement. Appliquée dès le départ, elle empêche les
  classes rares d'émerger ;
- `distance_sur_ponctuels` (faux par défaut) : elle ne s'applique qu'aux câbles. Activée
  sur les boîtes, elle a fait disparaître leur détection dans les essais ;
- `critere = "distance"` : le modèle retenu est celui dont l'écart moyen au tracé (en cm,
  affiché à chaque epoch) est le plus faible, plutôt que le meilleur IoU.

## Présentation

[`docs/presentation.html`](docs/presentation.html) : présentation fonctionnelle et technique (à ouvrir dans un navigateur, fichier autonome). Contient des images des chantiers Enedis : à ne pas diffuser sans accord.

## Guide développeur

[`docs/developpeurs.html`](docs/developpeurs.html) (à ouvrir dans un navigateur) : chaque étape de la chaîne
en détail, avec la fonction, l'algorithme ou le modèle, les paramètres et le code source.

## Entraînement

Guide complet (préparation, interface web avec graphiques, paramètres, lecture des métriques) :
[`docs/ENTRAINEMENT.md`](docs/ENTRAINEMENT.md). Page web : `python -m tranchee3d atelier` puis
http://127.0.0.1:8765/entrainement.

## Bruit d'entraînement

En plus des rotations, miroirs et variations de couleur, chaque tuile reçoit (réglages dans la
configuration, 0 = désactivé) : bruit gaussien sur les couleurs (`bruit_rvb`), bruit et décalage
du sol sur la hauteur relative (`bruit_hauteur`, `decalage_hauteur`), changement d'échelle
(`echelle`), trou de pixels vides non étiquetés (`proba_trou`, `taille_trou`).

Poursuivre un entraînement depuis des poids existants :
`python -m tranchee3d --config config_reprise.toml entrainer prepare modeles/v2_bruit --validation ... --reprendre modeles/v1/meilleur.pt`

## Atelier de débogage

`python -m tranchee3d atelier` puis http://127.0.0.1:8765 : choix ou téléversement d'un nuage,
exécution de la chaîne, un onglet par étape (nuage, ortho, tranchée, détection 2D, nuage nettoyé,
dessins auto, séparation réseaux, sous masque, raccord ouvrages, résultat + évaluation + réseau BT existant). Fichiers dans `sortie/atelier/<chantier>/`.

## Modules

| Module | Rôle |
|---|---|
| `nuage.py` | lecture LAS/LAZ, repère local |
| `ortho.py` | nuage → ortho RVB + raster de hauteur ; prise en compte d'une ortho fournie |
| `tranchee.py` | détection de la tranchée, hauteur relative au sol |
| `etiquettes.py` | GeoJSON → masques ; retrait des parties cachées sous un obstacle |
| `modele.py` | entraînement U-Net (CE + Dice + perte de distance), export ONNX |
| `inference.py` | inférence ONNX Runtime par tuiles |
| `retour3d.py` | masques → polylignes 3D (ligne de crête) et ponctuels 3D |
| `raccordement.py` | fusion des coupures, tronçons déduits sous masque, prolongement jusqu'aux boîtes et coffrets |
| `facade.py` | orthos de face (RVB + profondeur) devant les extrémités libres des câbles, pour chercher les coffrets : `sortie/faces/` (`face_NN.png`, `face_NN.npz`, `faces.geojson`, `controle_faces.png`). Aucune détection pour l'instant |
| `reseau_existant.py` | réseau BT open data Enedis autour du chantier, raccord probable (`reseau_existant.geojson`) |
| `export.py` | GeoJSON 3D, images de contrôle |
| `evaluation.py` | écarts à la vérité terrain |
| `visualiseur.py` | page HTML 3D autonome |
| `synthetique.py` | chantiers de test |
| `rpd.py` | import des chantiers Enedis (récolement RPD) |
| `preparation.py` | import + préparation d'un dossier brut, avec cache (`preparation.json`) |
| `atelier.py` | exécution d'un nuage étape par étape, un fichier par étape (`assets/atelier.html`) |
| `entrainements.py` | lancement et suivi des entraînements en sous-processus, anti-veille |
| `serveur.py` | routes HTTP de l'atelier et de la page d'entraînement |

## Limites connues

- Deux câbles superposés : seul celui du dessus est mesuré.
- Réseau existant (open data Enedis, 2D, précis à 5-10 m) : le souterrain est posé au Z des câbles détectés,
  l'aérien à sol + `hauteur_aerien` (sol = médiane du Z du nuage, tirée un peu vers le bas par la tranchée) ;
  le raccord probable est indicatif, jamais évalué.
- Un câble plaqué contre une paroi verticale est mal vu du dessus.
- Un tronçon « déduit » est une interpolation, pas une mesure.
  Le prolongement vers un ouvrage vise `hauteur_entree_ouvrage` (0,40 m) au-dessus de son point le plus bas ;
  le récolement varie de 0,15 à 0,85 m, et une jonction peut être levée sous l'objet détecté (OSR31662830 : 1,3 m).
- L'appariement des extrémités est glouton (coût croissant), pas une affectation optimale.
  Les petits tronçons qui doublent un tronçon plus long sont retirés avant (`longueur_eperon`,
  `ecart_eperon`) ; sous `trou_libre`, deux extrémités sont reliées sans contrôle d'angle.
- Le Z des câbles est lissé sur `lissage_z` (1 m) : un vrai changement de pente plus court est adouci.
- Les données synthétiques sont beaucoup plus faciles que des données réelles :
  elles valident le fonctionnement du code, pas la performance attendue.
- Une mesure de crête prise sur un objet au-dessus du câble (planche, plaque) est écartée si elle
  s'éloigne de plus de `ecart_z_aberrant` (10 cm) de la médiane locale ; un objet qui couvre plus
  de la moitié de `lissage_z` n'est plus détecté ainsi.
- Les plages des augmentations radiométriques (contraste, gain, décalage) sont encore en dur dans
  `modele._lot_aleatoire` ; le bruit, l'échelle, les trous et la couleur du fourreau sont dans `Config`.
- `modele.onnx` est accompagné d'un `modele.json` (pixel, tuile, encodeur, paramètres du sol) ;
  `predire` avertit si la config courante diffère, mais n'empêche rien.
- Les mesures publiées sur les 2 chantiers de validation ont aussi servi à choisir l'epoch :
  légèrement optimistes. Utiliser des chantiers de test (`--test`, ou le % test de la page) pour
  une mesure indépendante.

## Raccordement, ouvrages et réseau existant

Après la détection, trois étapes relient ce qui a été vu (onglets 8 à 10 de l'atelier) :

1. **Sous masque** (`raccorder`) : deux extrémités alignées à moins de `trou_max` (6 m) sont reliées
   par un tronçon `deduit`. Exemple : sur `OSR31663028`, une plaque cache 4,08 m de fourreau ; avec
   l'ancien seuil de 4 m la liaison était refusée (61 % du tracé retrouvé), à 6 m elle est faite
   (68 %). Sur les 10 autres chantiers, le passage de 4 à 6 m ne change rien.
2. **Ouvrages** (`raccorder_ouvrages`) : chaque bout libre à moins de `distance_ouvrage_max` (2 m en plan)
   d'une boîte ou d'un coffret détecté est prolongé jusqu'à lui, dans la continuité du fourreau (même
   classe, même `id_cable`, `origine = deduit`). Le point d'arrivée est à `hauteur_entree_ouvrage`
   (0,40 m) au-dessus du point le plus bas de l'objet, là où le récolement lève le câble (médiane des
   20 ouvrages des 11 chantiers). Gain de 1 à 4 points de tracé retrouvé sur 10 chantiers sur 11.
3. **Réseau existant** (`reseau_existant`, à l'étape Résultat) : lignes BT aériennes (`reseau-bt`)
   et souterraines (`reseau-souterrain-bt`) téléchargées sur l'API open data Enedis autour du nuage
   (`marge_reseau`, 30 m), puis un raccord probable par câble, depuis son bout libre le plus proche,
   s'il est à moins de `distance_raccord_max` (15 m). Souterrain : raccord à plat, au Z du câble.
   Aérien : un poteau est ajouté sous la ligne et le raccord y monte (`hauteur_aerien`, 8 m).
   Exemples : `64643029`, ligne aérienne à 2 cm en plan de la remontée → poteau ajouté ;
   `OSR31661291`, ligne souterraine à 15 cm du bout du câble.
   Sortie à part, `reseau_existant.geojson` (classes `bt_souterrain`, `bt_aerien`, `poteau_ajoute`,
   `raccord_probable`) : `resultat.geojson` et l'évaluation ne changent pas. Le téléchargement est
   gardé dans `opendata_bt.geojson` ; hors ligne sans cache, un avertissement s'affiche et le reste est produit.

```toml
trou_max = 6.0                 # m
distance_ouvrage_max = 2.0     # m, en plan
hauteur_entree_ouvrage = 0.40  # m au-dessus du bas de l'ouvrage
reseau_existant = true         # false : ni téléchargement ni raccord
distance_raccord_max = 15.0    # m
hauteur_aerien = 8.0           # m au-dessus du sol
```

## Licences

| Élément | Licence | Remarque |
|---|---|---|
| segmentation-models-pytorch 0.6 (U-Net) | MIT | |
| Poids ImageNet de l'encodeur ResNet-18 (`smp-hub/resnet18.imagenet` = torchvision `resnet18-5c106cde.pth`) | « other », non précisée | Point de départ de **tous** les modèles de `modeles/`. torchvision indique que ces poids « peuvent avoir leurs propres conditions, issues du jeu d'entraînement » ; les images ImageNet sont réservées à la recherche non commerciale, et la portée de cette restriction sur les poids n'est pas tranchée. À valider par le juridique avant un usage commercial, sinon réentraîner avec `poids_encodeur = ""`. |
| Modèles entraînés (`modeles/`) | propriétaire | Dérivés des poids ci-dessus et des données Enedis : diffusion soumise à l'accord d'Enedis. |
| torch 2.15 / torchvision / timm | BSD-3 (+ composants tiers permissifs) / BSD-3 / Apache-2.0 | Entraînement seulement. |
| onnx / onnxruntime | Apache-2.0 / MIT | Inférence. |
| numpy, scipy, scikit-image, shapely, rasterio, laspy | BSD | |
| lazrs, pyproj, Pillow | MIT / MIT / MIT-CMU | |
| three.js r187 (`assets/three_bundle.min.js`) | MIT | Mention de copyright conservée dans le bundle. |
| Réseau BT open data Enedis | Licence Ouverte 2.0 (Etalab) | Réutilisation libre, y compris commerciale, en citant la source et la date de mise à jour (juillet 2026, deux par an). La mention est portée par `reseau_existant.geojson` (`source`) et par les deux visualiseurs. |

Aucun composant AGPL/GPL (pas d'Ultralytics/YOLO).

## Tests

```bash
pytest tests        # ~1 s, sans données : ortho, tranchée, raccordement, ouvrages, réseau existant, évaluation, config, crête
```
