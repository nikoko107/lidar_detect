# Entraîner le modèle de détection

Le modèle est un U-Net qui segmente l'ortho (RVB + hauteur relative) en cinq classes : fond,
câble noir, fourreau rouge, boîte de jonction, coffret. Il s'entraîne sur MPS (GPU du Mac) et
s'exporte en ONNX pour l'inférence sur CPU.

## 1. Les données

**Le plus simple** : déposez les chantiers Enedis extraits dans un dossier du projet (par
exemple `bruts/`), un sous-dossier par chantier, **archive complète** :

```
bruts/<chantier>/nuage/*.laz                       un ou plusieurs nuages
bruts/<chantier>/recolement/RPD_*_Reco.geojson     vérité terrain
bruts/<chantier>/ortho/*.tif                       ortho fournie, si elle existe
bruts/<chantier>/project-metadata.json             système de coordonnées
```

puis choisissez ce dossier dans la page d'entraînement (section 2) : l'import et la préparation
se font au lancement. Un chantier déjà préparé n'est refait que si ses fichiers ou les réglages
de préparation (pixel, tampons, détection de tranchée…) ont changé (fiche
`prepare/<chantier>/preparation.json`). Pendant cette phase, la page affiche « Préparation » et
le journal.

**Ortho fournie** : si `<chantier>/ortho/` contient une ortho (`.tif`, `.tiff`, `.jp2`, géoréférencée),
elle remplace les couleurs de l'ortho générée depuis le nuage là où elle couvre ; le raster de
hauteur vient toujours du nuage. L'atelier l'utilise aussi automatiquement pour la détection.

Les étapes manuelles ci-dessous font la même chose en ligne de commande.

### Import et préparation à la main

Toutes les commandes se lancent depuis `tranchee3d/`, avec l'environnement Python :

```bash
cd /Volumes/FAT/lidar/tranchee3d
source ~/.venvs/tranchee3d/bin/activate
```

1. **Importer** les chantiers Enedis extraits dans `bruts/` (nuage + récolement RPD) :

   ```bash
   python -m tranchee3d importer bruts/* donnees
   ```

   Chaque chantier donne `donnees/<chantier>/nuage.laz` et `verite.geojson`.

2. **Préparer** chaque chantier (ortho, tranchée, masques d'étiquettes) :

   ```bash
   for d in donnees/*/; do
     n=$(basename $d)
     python -m tranchee3d preparer $d/nuage.laz $d/verite.geojson prepare/$n
   done
   ```

   Vérifiez `prepare/<chantier>/controle_etiquettes.png` : le tracé coloré doit tomber sur le
   fourreau visible. Un chantier mal étiqueté apprend au modèle à se tromper.

## 2. Lancer l'entraînement depuis l'interface web

```bash
python -m tranchee3d atelier
```

Ouvrez **http://127.0.0.1:8765/entrainement** (ou le lien « Entraînement → » de l'atelier).

**Colonne de gauche : nouvel entraînement**

- **Nom** : dossier de sortie `modeles/<nom>/`. Laissé vide : `run_AAAAMMJJ_HHMM`.
- **Données** : un dossier **brut** (`<chantier>/nuage/*.laz`, importé et préparé au lancement,
  voir section 1) ou un dossier **préparé** (`<chantier>/chantier.npz`). La page liste les dossiers
  du projet de l'un ou l'autre type.
- **Répartition** : pourcentages de **validation** et de **test** ; le reste va à l'entraînement.
  La répartition se fait par chantier entier, jamais par tuile, tirée au hasard selon la
  **graine** (même graine = même répartition). L'aperçu donne le nombre de chantiers de chaque
  groupe (11 chantiers à 15 % / 15 % → 7 · 2 · 2). Tout pourcentage non nul donne au moins un chantier.
  - **Validation** : mesurée à chaque epoch, sert à choisir l'epoch gardée.
  - **Test** : mesuré une seule fois, à la fin, avec le modèle retenu. Ces chantiers n'ont servi
    ni à apprendre ni à choisir l'epoch : c'est la mesure la plus honnête (tuile « Test »).
  - **Validation à 0 %** : plus de mesure pour choisir, la **dernière epoch** est gardée et les
    graphiques d'écart et d'IoU restent vides. Avec test aussi à 0 % : modèle final sur tous les
    chantiers. Fixez alors le nombre d'epochs d'après un essai validé.
  - Pour comparer deux essais, gardez la même graine et les mêmes pourcentages. Ne faites pas
    tourner les chantiers de validation d'une epoch à l'autre : un chantier déjà vu en entraînement
    donne un score trompeusement bon.
- **Partir de** : « Zéro » repart des poids ImageNet ; un `meilleur.pt` poursuit un entraînement
  existant (baissez alors le taux d'apprentissage, voir plus bas).
- **Paramètres** : voir le tableau ci-dessous. Valeurs par défaut pré-remplies.
- **Lancer** / **Arrêter**. Un seul entraînement à la fois. Fermer la page ou redémarrer le
  serveur n'arrête pas l'entraînement : le serveur le retrouve au redémarrage.
- **☕ Anti-veille** (en haut à droite) : lance `caffeinate` attaché au processus d'entraînement.
  Le Mac ne se met plus en veille tant que l'entraînement tourne ; la protection s'arrête seule
  à la fin. Refermer l'écran d'un portable sur batterie met quand même le Mac en veille.

**Colonne de droite : suivi**, rafraîchi toutes les 3 secondes. Le menu en haut choisit
l'entraînement affiché (en cours ou passé).

- **Progression**, **meilleur écart moyen** (epoch retenue), **écart du fourreau rouge**,
  **temps par epoch** et fin estimée.
- **Perte d'entraînement** : doit descendre.
- **Écart au tracé (cm), validation**, par classe : distance moyenne entre la détection et le
  tracé de vérité terrain. C'est le critère principal. Plafonné à 20 cm (classe jamais trouvée).
- **IoU par classe, validation** : recouvrement entre masque prédit et masque de vérité.
- La ligne pointillée « retenu » marque l'epoch dont les poids sont gardés. Survolez un
  graphique pour lire les valeurs d'une epoch ; le tableau donne toutes les epochs.
- **Journal** : fin de la sortie texte, utile en cas d'erreur.

## 3. Lancer l'entraînement en ligne de commande

```bash
# depuis zéro, 2 chantiers de validation et 1 de test (noms de dossiers dans prepare/)
python -m tranchee3d entrainer prepare modeles/mon_essai --validation 64643074 OSR31662148 --test 64643029

# avec des paramètres modifiés : fichier TOML (ne mettre que les lignes changées)
python -m tranchee3d --config ma_config.toml entrainer prepare modeles/mon_essai --validation 64643074 OSR31662148

# modèle final sur tous les chantiers (sans --validation : dernière epoch gardée)
python -m tranchee3d --config ma_config.toml entrainer prepare modeles/final

# directement depuis un dossier brut (import + préparation dans prepare/ si nécessaire)
python -m tranchee3d entrainer prepare modeles/mon_essai --bruts bruts --validation 64643074 --test OSR31662148

# poursuite depuis des poids existants
python -m tranchee3d --config config_reprise.toml entrainer prepare modeles/v2 \
    --validation 64643074 OSR31662148 --reprendre modeles/v1/meilleur.pt
```

Exemple de `ma_config.toml` :

```toml
epochs = 40
taux_apprentissage = 0.0005
proba_trou = 0.4
```

La liste complète des paramètres et leurs valeurs par défaut est dans `src/tranchee3d/config.py`.

## 4. Paramètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `epochs` | 30 | Nombre de passes ; une validation à la fin de chacune. |
| `iterations_par_epoch` | 200 | Lots tirés par epoch. |
| `lot` | 8 | Tuiles par lot. Plus grand = plus de mémoire GPU. |
| `taux_apprentissage` | 0,001 | Maximum du cycle OneCycle. Pour une reprise : ~0,0003. |
| `tuile` | 384 | Côté des tuiles en pixels (384 px ≈ 1,5 m à 4 mm/pixel). |
| `encodeur` | resnet18 | Réseau de base. Une reprise exige le même encodeur. |
| `part_tuiles_objet` | 0,7 | Part des tuiles centrées sur un câble ou un ponctuel (classes rares). |
| `poids_distance` | 3,0 | Poids de la perte de distance au tracé ; 0 la désactive. |
| `critere` | distance | Choix des poids gardés : `distance` (écart moyen au tracé) ou `iou`. |
| `bruit_rvb` | 0,03 | Bruit gaussien sur les couleurs (0 à 1). |
| `bruit_hauteur` | 0,01 m | Bruit gaussien sur la hauteur relative. |
| `decalage_hauteur` | 0,05 m | Erreur aléatoire ± sur le sol estimé. |
| `echelle` | 0,15 | Changement d'échelle aléatoire ± 15 % (autres densités de nuage). |
| `proba_trou` | 0,3 | Probabilité, par tuile, d'un trou de pixels vides (non étiqueté). |
| `taille_trou` | 0,25 | Côté maximal du trou, en part de la tuile. |
| `teinte_fourreau` | 0,03 | Variation ± de la teinte des fourreaux (0,03 ≈ ±11° : rouge → orangé ou rosé). |
| `saturation_fourreau` | 0,25 | Variation ± de la saturation des fourreaux (rouge vif à délavé). |

S'y ajoutent les augmentations fixes : rotations de 90°, miroirs, luminosité, contraste et
dominante de couleur.

## 5. Lire les résultats

- **Sur-apprentissage** : la perte d'entraînement continue de baisser mais l'écart en validation
  stagne ou remonte. Le modèle apprend les chantiers par cœur. Remèdes : plus de bruit
  (`proba_trou`, `echelle`), moins d'epochs, ou plus de chantiers.
- **Classe à 20 cm en permanence** : le modèle ne la détecte pas du tout sur la validation
  (cas actuel des boîtes de jonction : trop peu d'exemples, surtout des remontées
  aéro-souterraines).
- **Câble noir absent des graphiques** : aucun câble noir nu dans les données actuelles, la
  classe n'est ni entraînée ni mesurée.
- Les écarts affichés ici sont mesurés **en 2D sur l'ortho**. La précision 3D finale se mesure
  après coup (section 7).

## 6. Fichiers produits

Dans `modeles/<nom>/` :

| Fichier | Contenu |
|---|---|
| `meilleur.pt` | Poids PyTorch de l'epoch retenue (pour une reprise). |
| `modele.onnx` | Modèle exporté, utilisé par `predire` et l'atelier (inférence CPU). Écrit à la fin seulement. |
| `historique.jsonl` | Métriques de chaque epoch (lu par la page web). |
| `config.toml` | Paramètres choisis dans la page web. |
| `entrainement.log` | Sortie texte complète. |

Un entraînement arrêté avant la fin garde `meilleur.pt` mais n'a pas de `modele.onnx`.

## 7. Après l'entraînement : mesurer en 3D

```bash
n=64643074
python -m tranchee3d predire donnees/$n/nuage.laz modeles/<nom>/modele.onnx sortie/$n --verite donnees/$n/verite.geojson
python -m tranchee3d evaluer sortie/$n/resultat.geojson donnees/$n/verite.geojson
```

Ne mesurez que sur les chantiers de validation : sur un chantier d'entraînement, le score est
trompeusement bon. Pour inspecter chaque étape (ortho, détection, nuage nettoyé, tracés,
raccordement), ouvrez l'atelier http://127.0.0.1:8765 et choisissez le nouveau `modele.onnx`.

Durée indicative sur le Mac M5 : 1,5 à 2 min par epoch avec 9 chantiers, soit ~45 min à
1 h pour 30 epochs.
