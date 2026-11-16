# Détection 3D de câbles et d'ouvrages en tranchée ouverte

## Objectif

À partir d'un nuage de points colorisé et géoréférencé d'une tranchée de travaux, produire automatiquement en 3D :

- les **câbles électriques** (linéaires), de deux types : câble noir nu, câble sous fourreau rouge ;
- les **boîtes de jonction** (ponctuels) ;
- les **coffrets électriques** (ponctuels, hors sol).

## Contraintes

- **Précision centimétrique** sur les parties visibles.
- **Sortie en 3D** (X, Y, Z dans le système de coordonnées du nuage).
- Les câbles qui passent **sous une zone masquée** doivent être reliés et identifiés comme tels.
- **Inférence sur CPU uniquement**.
- Machine de développement : Mac M5, 32 Go de RAM. Entraînement autorisé sur MPS.
- Langage : Python.

## Données

- Nuages de points colorisés (RVB), très denses et très précis. Un chantier fait quelques dizaines de mètres.
- Orthophoto haute définition, **disponible seulement pour certains chantiers**.
- Vérité terrain : GeoJSON des objets (câbles en lignes, boîtes et coffrets en points).

### À confirmer avant de coder (demander à l'utilisateur)

- Format exact des nuages (LAS/LAZ ?) et système de coordonnées.
- Densité réelle du nuage (espacement moyen entre points).
- Le GeoJSON contient-il le Z ? Quels attributs distinguent câble noir et fourreau rouge ?
- Convention de levé : axe du câble ou génératrice supérieure ?
- Nombre de chantiers annotés disponibles.
- Fréquence des câbles superposés ou plaqués contre une paroi verticale.

## Chaîne de traitement

Principe : **détecter en 2D sur une ortho, mesurer en 3D sur le nuage**.

1. **Entrée uniforme**
   - Lire le nuage.
   - Utiliser l'ortho fournie si elle existe, sinon générer une ortho depuis le nuage (projection verticale, pixel de 2 à 5 mm, point le plus haut par pixel).
   - Générer dans tous les cas un raster de hauteur aligné sur l'ortho.
2. **Détection de la tranchée**
   - Estimer le sol environnant sur le raster de hauteur ; la tranchée est la zone en contrebas.
   - Recadrer sur la tranchée avec une marge. Garder une bande hors sol autour pour les coffrets.
3. **Segmentation 2D**
   - Entrée : RVB + hauteur relative (4 canaux). Découpage en tuiles avec recouvrement.
   - Classes : fond, câble noir, fourreau rouge, boîte de jonction, coffret.
   - Étiquettes : GeoJSON rastérisé avec un tampon réglable (lignes) et un disque (points).
   - Modèle : U-Net léger via `segmentation-models-pytorch`.
   - Perte : entropie croisée + Dice + perte de distance au tracé (boundary loss, câbles seulement par défaut). Meilleur modèle choisi sur l'écart moyen au tracé.
   - Entraîner sur les deux types d'ortho (fournie et générée).
   - Séparer entraînement et validation **par chantier**, jamais par tuile.
4. **Export et inférence CPU**
   - Export ONNX, inférence avec ONNX Runtime (CPU).
5. **Retour en 3D**
   - Récupérer les points bruts du nuage sous chaque masque.
   - Séparer par altitude les câbles superposés.
   - Câbles : suivre la ligne de crête (génératrice supérieure) le long du squelette du masque, à pleine résolution. Produire une polyligne 3D lissée.
   - Boîtes et coffrets : regroupement (DBSCAN), puis centroïde 3D et emprise.
6. **Raccordement sous masque**
   - Apparier les extrémités de tronçons par affectation globale (coût : direction, altitude, couleur, classe).
   - Ne pas relier au-delà d'une longueur de trou maximale (paramètre).
   - Relier par une courbe 3D lissée et marquer le tronçon comme déduit.
7. **Export**
   - GeoJSON 3D ou GeoPackage.
   - Attributs par tronçon : `classe`, `origine` (`observe` / `deduit`), `confiance`.

## Points de vigilance

- Câble noir sur terre sombre : cas le plus difficile, à évaluer séparément.
- Un câble contre une paroi verticale est presque invisible vu du dessus.
- Un décalage de quelques centimètres entre GeoJSON et nuage bruite les étiquettes et plafonne la précision mesurable.
- Boîtes et coffrets sont rares : surveiller le déséquilibre de classes. Une boîte peut aussi se repérer comme un élargissement local le long d'un câble déjà tracé.
- La précision finale est plafonnée par le géoréférencement du nuage.
- Un tronçon déduit est une interpolation, pas une mesure.

## Pile technique

- Nuage : `laspy` (avec `lazrs`), `numpy`
- Raster et vecteur : `rasterio`, `geopandas`, `shapely`
- Modèle : `torch` (MPS pour l'entraînement), `segmentation-models-pytorch`, `onnx`, `onnxruntime`
- Post-traitement : `scikit-image`, `scikit-learn`, `scipy`
- Visualisation facultative : `open3d`, `matplotlib`
- Ne pas utiliser Ultralytics/YOLO (licence AGPL).

## Organisation du travail

Avancer par étapes indépendantes. Chaque étape est un module avec une commande en ligne, et produit une sortie **vérifiable visuellement** (image ou fichier ouvrable dans QGIS / CloudCompare). Ne pas passer à l'étape suivante sans validation de l'utilisateur.

1. `ortho` : nuage → ortho + raster de hauteur
2. `tranchee` : détection et recadrage
3. `etiquettes` : GeoJSON → masques ; puis `entrainement`
4. `inference` : ONNX sur CPU → masques
5. `retour3d` : masques → polylignes et points 3D
6. `raccordement` : liaison sous masque
7. `export` et `evaluation`

Structure :

```
src/tranchee3d/   modules de la chaîne
donnees/          données (jamais versionnées)
modeles/          poids et fichiers ONNX
sortie/           résultats et contrôles visuels
```

## État du code

Une première version de toute la chaîne existe dans `src/tranchee3d/` (voir `README.md`
pour les commandes). Elle a été exécutée de bout en bout **uniquement sur des chantiers
synthétiques** (`synthetique.py`), bien plus faciles que la réalité. À faire en priorité :

1. Faire tourner `preparer` sur un vrai chantier et vérifier les images de contrôle
   (tranchée, étiquettes). Ajuster `pixel`, `fenetre_sol`, `profondeur_min`, les tampons.
2. Adapter `champ_classe` et `valeurs_classes` au GeoJSON réel.
3. Entraîner sur les chantiers réels, valider par chantier.
4. Traiter les limites connues listées dans le README (câbles superposés, parois,
   appariement glouton à remplacer par une affectation globale si besoin).

Ne pas réécrire ce qui fonctionne : modifier module par module, en revalidant visuellement.

## Évaluation

- Segmentation : IoU par classe, par chantier.
- Câbles : distance 3D entre polyligne prédite et vérité terrain (médiane et 95e centile), taux de longueur détectée.
- Ponctuels : précision et rappel avec tolérance de distance.
- Rapporter séparément les tronçons observés et déduits, et le câble noir et le fourreau rouge.

## Règles

- Tous les seuils (taille de pixel, tampon, longueur de trou maximale) sont des paramètres dans un fichier de configuration, pas des constantes dans le code.
- Conserver le système de coordonnées d'origine de bout en bout ; attention à la perte de précision en float32 sur de grandes coordonnées (travailler en coordonnées locales décalées).
- Commencer simple : valider la chaîne complète sur un seul chantier avant d'optimiser.
