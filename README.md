# Atelier 1 — Deep Learning avec PyTorch

**TP : Régression et classification multiclasses avec des réseaux DNN/MLP**  
**Environnement :** Google Colab (PyTorch, GPU CUDA), pandas, NumPy, scikit-learn, skorch, Matplotlib, Seaborn  
**État d’avancement :** Parties 1 (régression) et 2 (classification multiclasses) réalisées ; résultats expérimentaux documentés ci-dessous.

## Objectifs

Ce TP vise à mettre en œuvre des réseaux de neurones entièrement connectés (*Multi-Layer Perceptron*, MLP) avec PyTorch pour deux tâches :

1. **Régression :** prédiction d'une valeur continue à partir des données boursières NYSE.
2. **Classification multiclasses :** prédiction du type de défaillance d’une machine industrielle.

Les étapes demandées comprennent l'analyse exploratoire des données (EDA), le prétraitement, la construction du modèle, la recherche d'hyperparamètres avec `GridSearchCV`, la visualisation des courbes d'apprentissage, la régularisation et l'évaluation des résultats.

## Partie 1 — Régression : NYSE

### 1. Jeu de données et objectif

- **Source :** [NYSE — Kaggle](https://www.kaggle.com/datasets/dgawlik/nyse)
- **Fichier utilisé :** `prices-split-adjusted.csv`
- **Sous-ensemble choisi :** action Apple (`AAPL`)
- **Période analysée :** de janvier 2010 à décembre 2016
- **Données exploitables après création de la cible :** **1 761 observations**

**Objectif de prédiction :** estimer le prix de clôture ajusté à la **séance de négociation suivante** à partir des données disponibles en fin de séance courante.

| Entrées du modèle (`X`) | Variable cible (`y`) |
|---|---|
| `open`, `high`, `low`, `close`, `volume` du jour *t* | `close_next_day` = `close` du jour *t+1* |

La cible a été construite avec `shift(-1)` après tri chronologique, puis la dernière ligne sans cible a été retirée. Le modèle suppose que les données complètes de la séance *t* sont connues lors de la prédiction ; il ne s'agit donc **pas** d'une prédiction faite avant l'ouverture du marché le jour *t*.

### 2. Analyse exploratoire (EDA)

Les analyses réalisées dans le notebook couvrent : la structure du dataset, les types de colonnes, les valeurs manquantes et doublons, les statistiques descriptives, les distributions des prix, les corrélations et l'évolution historique du prix de clôture d'Apple.

**Observations principales :**

- Les distributions de `open`, `high`, `low` et `close` sont fortement asymétriques vers la droite lorsqu'on regroupe les différentes entreprises.
- Les prix `open`, `high`, `low` et `close` sont très corrélés entre eux dans le dataset global (coefficients affichés à **1,00** après arrondi à deux décimales).
- La corrélation linéaire du `volume` avec les variables de prix est faible et négative, autour de **−0,13** sur ce dataset global.
- Le prix de clôture ajusté d'Apple montre une tendance générale à la hausse entre 2010 et 2016, accompagnée de fluctuations et de corrections. Il est donc essentiel de préserver l'ordre temporel lors de l'évaluation.

> Les corrélations descriptives calculées sur l'ensemble des entreprises ne démontrent pas une capacité à prédire le prix du lendemain pour Apple.

### 3. Prétraitement et séparation temporelle

Après filtrage sur `AAPL`, création de la cible et suppression de sa valeur manquante, les 1 761 observations ont été séparées **sans mélange** :

| Ensemble | Nombre de lignes | Proportion approximative | Période |
|---|---:|---:|---|
| Entraînement | 1 232 | 70 % | 04/01/2010 → 21/11/2014 |
| Validation | 264 | 15 % | 24/11/2014 → 10/12/2015 |
| Test | 265 | 15 % | 11/12/2015 → 29/12/2016 |

Les variables d'entrée ont été transformées à l'aide de `StandardScaler`, ajusté **uniquement sur Train** puis appliqué à Validation et Test. La cible a été standardisée séparément avec un second scaler, également ajusté sur Train. Les prédictions ont été reconverties en **USD** pour calculer les métriques finales.

Pour la recherche d'hyperparamètres, les scalers d'entrée ont été intégrés dans un pipeline `scikit-learn`, de manière à être recalculés dans chaque fold temporel ; la cible a été standardisée dans le wrapper `ScaledTargetNet` pendant chaque entraînement.

### 4. Premier modèle MLP (baseline)

Architecture PyTorch :

```text
Entrée : 5 variables
   ↓
Dense(64) + ReLU
   ↓
Dense(32) + ReLU
   ↓
Dense(1) — sortie continue
```

**Paramètres :** 2 497 paramètres entraînables, `MSELoss`, optimiseur **Adam**, `learning_rate = 0.001`, `batch_size = 32`, **100 époques**. Entraînement sur GPU CUDA dans Google Colab.

**Évolution de l'entraînement :**

| Mesure | Résultat |
|---|---:|
| Dernière Train Loss (MSE standardisée) | 0.002948 |
| Dernière Validation Loss (MSE standardisée) | 0.011261 |
| Meilleure Validation Loss observée | 0.009750 |
| Époque correspondante | 96 / 100 |

La perte diminue fortement au début, puis la perte de validation oscille autour d'une valeur plus élevée que celle d'entraînement. Cet écart suggère une généralisation plus difficile vers la période de validation, sans suffire à démontrer à lui seul un surapprentissage.

### 5. Optimisation avec GridSearchCV

Le modèle PyTorch a été rendu compatible avec scikit-learn grâce à **skorch**. La recherche a utilisé `GridSearchCV` avec **`TimeSeriesSplit(n_splits=3)`**, sur l'ensemble d'entraînement uniquement.

| Hyperparamètre | Valeurs comparées |
|---|---|
| Architectures | (64, 32) ; (128, 64) |
| Learning rate | 0.001 ; 0.0005 |
| Optimiseur | Adam ; SGD |
| Époques | 30 ; 60 |
| Batch size | 32 (fixe) |

Soit **16 configurations et 48 entraînements** (3 folds par configuration). Critère de sélection : **MSE moyenne en validation croisée**, exprimée en USD² après inversion de la standardisation de la cible.

**Meilleure configuration trouvée :** Adam, learning rate **0.0005**, **60 époques**, couches cachées **64 et 32**.

**RMSE moyen issu du score de validation croisée :** **1.6847 USD**, calculé par `sqrt(-grid_search.best_score_)`. Il correspond à la racine de la MSE moyenne des folds, et non à la moyenne arithmétique de leurs RMSE.

Cette configuration n'a pas amélioré de façon significative les performances sur la période de validation réservée.

### 6. Régularisation

Un modèle MLP de même taille (5 → 64 → 32 → 1) a été entraîné avec les techniques suivantes :

- **Dropout :** 0.2 après chaque couche cachée.
- **Weight decay (pénalisation des poids) :** 0.0001 avec Adam.
- **Early stopping :** patience de 15 époques, `min_delta = 0.0001`, maximum 150 époques ; restauration des poids associés à la meilleure perte de validation.
- **Learning rate :** 0.0005.

L'effet mesuré correspond à la **combinaison** de ces techniques ; les expériences ne permettent pas d'attribuer une amélioration à l'une d'elles isolément.

### 7. Résultats quantitatifs

Les erreurs sont évaluées **dans l'unité d'origine (USD)**. Pour référence, le modèle naïf prédit simplement que le prix de clôture de la séance suivante sera égal à celui de la séance courante : `close(t+1) ≈ close(t)`.

#### Validation

| Modèle | MAE (USD) ↓ | MSE (USD²) ↓ | RMSE (USD) ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| MLP initial | 1.7126 | 4.8568 | 2.2038 | 0.9110 |
| MLP optimisé | 1.7122 | 4.8587 | 2.2043 | 0.9109 |
| **MLP régularisé** | **1.6255** | **4.4714** | **2.1146** | **0.9180** |
| Modèle naïf | 1.4899 | 3.8281 | **1.9566** | **0.9298** |

La régularisation réduit le RMSE de validation d'environ **4,0 %** par rapport au premier MLP, mais le modèle naïf demeure meilleur.

#### Test

| Modèle | MAE (USD) ↓ | MSE (USD²) ↓ | RMSE (USD) ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| **MLP initial** | **1.1337** | **2.4211** | **1.5560** | **0.9569** |
| MLP régularisé | 1.1831 | 2.6688 | 1.6337 | 0.9525 |
| **Modèle naïf** | **1.0635** | **2.2629** | **1.5043** | **0.9597** |

Sur Test, le MLP initial dépasse légèrement le MLP régularisé, mais **aucun des deux ne bat le modèle naïf**. Le résultat illustre qu'un R² élevé pour les niveaux de prix ne suffit pas à prouver l'utilité prédictive du modèle : le prix de la veille constitue une référence très forte.

> **Limite méthodologique :** les performances Test du premier modèle avaient déjà été consultées durant l'exploration du TP. Il faut donc éviter de présenter cet ensemble comme totalement inédit par rapport au déroulement des choix expérimentaux. Les hyperparamètres de GridSearchCV ont néanmoins été recherchés sur Train avec des folds temporels, et la comparaison des modèles a été réalisée sur Validation.

### 8. Visualisations du notebook

Le notebook comporte notamment :

1. Histogrammes des variables de prix (`open`, `high`, `low`, `close`).
2. Matrice de corrélation incluant `volume`.
3. Évolution du prix de clôture ajusté d'Apple.
4. Courbes **Train Loss / Validation Loss** en fonction des époques pour le MLP initial et le modèle régularisé.
5. Analyse des résidus du modèle initial sur Validation.
6. Courbes de prédiction réelle / MLP initial / MLP optimisé sur Validation.
7. Comparaison finale des prix réels, MLP initial, MLP régularisé et modèle naïf sur Test.

**À propos de `Accuracy / Epochs` :** l'accuracy de classification ne convient pas à une cible numérique continue. Pour la régression, on privilégie MSE/MAE/RMSE et R². Un graphique RMSE/Epochs peut éventuellement compléter les courbes de perte, mais il ne faut pas présenter une « accuracy » de classification pour cette tâche.

### 9. Interprétation et limites

- Le MLP reproduit bien les **niveaux** de prix (R² Test > 0.95), mais ce résultat doit être comparé à la persistance du prix d'une séance à l'autre.
- L'optimisation a trouvé une bonne configuration parmi celles testées, **sans gain notable** sur la période de validation distincte.
- La régularisation améliore la validation mais **pas** le test : son effet dépend de la période et elle ne garantit pas de meilleures performances hors échantillon.
- Le modèle naïf reste la meilleure référence sur les métriques Test rapportées.
- L'étude porte sur une seule action (`AAPL`) et une période historique limitée ; aucune performance sur d'autres actions, régimes de marché ou données récentes n'est démontrée.
- Ces résultats sont pédagogiques et ne constituent pas une stratégie d'investissement validée.

### 10. Synthèse des apprentissages — Partie 1

Ce travail m'a permis de :

- Préparer une série temporelle de régression sans mélanger passé et futur.
- Utiliser `StandardScaler` sans introduire de fuite d'information dans la séparation des données.
- Construire et entraîner un réseau MLP avec PyTorch (`nn.Linear`, `ReLU`, `MSELoss`, Adam, `DataLoader`, GPU).
- Analyser les courbes de perte et les métriques de régression.
- Intégrer un modèle PyTorch à `GridSearchCV` avec `skorch` et validation croisée chronologique.
- Tester Dropout, weight decay et early stopping, puis comparer leurs résultats à ceux du modèle initial.
- Comprendre qu'un modèle profond peut obtenir de bonnes métriques tout en restant moins efficace qu'une référence naïve simple.

## Partie 2 — Classification multiclasses : maintenance prédictive

### 1. Dataset et objectif

- **Source :** [Machine Predictive Maintenance Classification — Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
- **Dimensions :** **10 000 lignes × 10 colonnes**.
- **Cible :** `Failure Type` (six catégories, dont `No Failure`).
- **Objectif :** prédire le type de panne à partir des paramètres d'une machine industrielle.

Variables utilisées : `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.

Les colonnes `UDI` et `Product ID` (identifiants) ainsi que `Target` (indicateur binaire de panne qui révélerait une partie de la cible) sont **exclues des entrées**.

### 2. Exploration et qualité des données

L'EDA a montré **0 valeur manquante, 0 ligne dupliquée, 0 doublon dans les identifiants**. La variable `Type` comporte trois modalités (`L`, `M`, `H`).

Les histogrammes révèlent notamment une vitesse de rotation asymétrique vers la droite, un couple approximativement centré autour de 40 Nm et une usure d'outil assez répartie jusqu'à environ 200 min. La matrice de corrélation indique :

- Température air / processus : **+0.88** (corrélation positive forte).
- Vitesse de rotation / couple : **−0.88** (corrélation négative forte).
- Usure de l'outil / autres variables numériques : **proche de 0** en corrélation linéaire.

Le boxplot du couple par type de panne montre des niveaux et dispersions différents, notamment un couple plutôt élevé pour `Overstrain Failure` ; les classes rares ont cependant des effectifs insuffisants pour conclure fortement.

**Répartition initiale des classes :**

| Classe | Effectif | Part |
|---|---:|---:|
| No Failure | 9 652 | 96,52 % |
| Heat Dissipation Failure | 112 | 1,12 % |
| Power Failure | 95 | 0,95 % |
| Overstrain Failure | 78 | 0,78 % |
| Tool Wear Failure | 45 | 0,45 % |
| Random Failures | 18 | 0,18 % |
| **Total** | **10 000** | **100 %** |

La classe majoritaire représente **96,52 %** des observations : une accuracy élevée peut donc être trompeuse. La métrique prioritaire choisie est le **F1-score macro**, avec analyse du recall par classe.

### 3. Séparation, encodage et standardisation

La séparation `train_test_split(..., stratify=y, random_state=42)` préserve approximativement les proportions de chaque classe :

| Classe | Train | Validation | Test |
|---|---:|---:|---:|
| Heat Dissipation Failure | 78 | 17 | 17 |
| No Failure | 6 756 | 1 448 | 1 448 |
| Overstrain Failure | 55 | 11 | 12 |
| Power Failure | 67 | 14 | 14 |
| Random Failures | 13 | 3 | 2 |
| Tool Wear Failure | 31 | 7 | 7 |
| **Total** | **7 000** | **1 500** | **1 500** |

La cible est encodée de 0 à 5 avec `LabelEncoder`. Les cinq variables numériques sont standardisées par `StandardScaler` et `Type` est transformée en trois colonnes par **One-Hot Encoding**. Le `ColumnTransformer` est ajusté **uniquement sur Train** : dimensions finales **(7000, 8)**, **(1500, 8)**, **(1500, 8)** ; les variables numériques du Train ont une moyenne voisine de 0 et un écart-type voisin de 1.

### 4. Augmentation de données avec SMOTENC

Pour traiter le mélange de variables numériques et catégorielles, l'expérience utilise **SMOTENC**, plutôt qu'un SMOTE appliqué directement à des colonnes One-Hot. L'augmentation s'effectue **uniquement sur Train** avec `random_state=42`, `k_neighbors=3`, et un objectif partiel de **500 observations pour chacune des cinq classes de panne** ; la classe `No Failure` demeure inchangée.

| Classe | Train avant | Train après SMOTENC |
|---|---:|---:|
| No Failure | 6 756 | 6 756 |
| Heat Dissipation Failure | 78 | 500 |
| Power Failure | 67 | 500 |
| Overstrain Failure | 55 | 500 |
| Tool Wear Failure | 31 | 500 |
| Random Failures | 13 | 500 |
| **Total** | **7 000** | **9 256** |

Les données augmentées sont transformées en **8 entrées `float32`** ; les labels sont de type `int64` pour `CrossEntropyLoss`. **Validation et Test ne sont ni augmentés ni rééquilibrés.** Un SMOTENC partant de seulement **13 cas réels** de `Random Failures` ne garantit pas une diversité de panne représentative.

### 5. Premier MLP PyTorch (baseline)

```text
8 entrées → Linear(64) → ReLU → Linear(32) → ReLU → Linear(6)
```

- **Paramètres entraînables :** 2 854.
- **Loss :** `CrossEntropyLoss` (sortie en *logits*, sans Softmax explicite).
- **Optimiseur :** Adam, `learning_rate=0.001`.
- **Batch size :** 64 ; **100 époques** ; entraînement possible sur GPU CUDA.
- **Entraînement :** 9 256 lignes après SMOTENC ; **validation :** 1 500 lignes non équilibrées.

Les courbes `Loss / Epochs` et `Accuracy / Epochs` montrent une baisse continue de la perte Train, tandis que la perte Validation baisse initialement puis remonte : **signe d'un surapprentissage possible**. En fin d'entraînement, l'accuracy Validation est **0.9587** et le F1 macro **0.5318** ; le maximum du F1 macro observé au cours de l'entraînement est **0.5475**, à l'époque 97, sans restauration de ces poids pour la baseline finale.

### 6. Optimisation avec GridSearchCV

Le réseau PyTorch a été intégré dans un pipeline avec `skorch`, `imblearn.pipeline.Pipeline`, `ColumnTransformer`, `SMOTENC` et une conversion explicite en `float32`. La recherche repart des **7 000 observations Train originales** : à chaque fold, le prétraitement et les exemples synthétiques sont calculés sur **le sous-ensemble d'entraînement du fold uniquement**, afin d'éviter une fuite de données.

- **Validation croisée :** `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`.
- **Critère :** `scoring='f1_macro'`.
- **Recherche :** architectures `(64, 32)` / `(128, 64)` ; learning rates `0.001` / `0.0005` ; Adam / SGD ; 30 / 60 époques ; batch size fixe de 64.
- **Volume :** 16 configurations × 3 folds = **48 entraînements de validation croisée**, puis réentraînement de la configuration retenue.

**Meilleurs hyperparamètres :** deux couches cachées **128 et 64**, **Adam**, `lr=0.0005`, **60 époques**. **Meilleur F1 macro CV : 0.5751**, moyenne des scores des folds (différent du score sur Validation séparée).

### 7. Régularisation et Early Stopping

Architecture régularisée : `8 → 128 → 64 → 6`, **9 798 paramètres**. Les techniques ont été testées **ensemble** :

- `Dropout(p=0.2)` après chacune des deux couches cachées ;
- Adam avec `weight_decay=0.0001` et `lr=0.0005` ;
- `Early Stopping` sur le **F1 macro Validation**, patience **15** époques, `min_delta=0.001`, plafond **150** époques, avec restauration des meilleurs poids.

**Résultat :** arrêt à l'époque **62**, meilleurs poids de l'époque **47**, meilleur **F1 macro Validation = 0.5093**. Les courbes Train/Validation Loss, Accuracy et F1 macro ont été produites. Cette régularisation combinée **n'a pas dépassé** le MLP optimisé ; elle ne permet pas d'isoler l'effet de chaque technique.

### 8. Résultats quantitatifs

#### Validation — 1 500 observations

| Modèle | Accuracy | Precision macro | Recall macro | F1 macro | F1 weighted |
|---|---:|---:|---:|---:|---:|
| MLP initial | 0.9587 | 0.5012 | 0.5812 | 0.5318 | 0.9657 |
| **MLP optimisé** | **0.9640** | **0.5379** | **0.5994** | **0.5565** | **0.9693** |
| MLP régularisé | 0.9520 | 0.4647 | 0.5854 | 0.5093 | 0.9610 |
| Majoritaire (`No Failure`) | 0.9653 | 0.1609 | 0.1667 | 0.1637 | 0.9483 |

Sur Validation, l'optimisation augmente le F1 macro de **0.5318 à 0.5565**. Malgré une accuracy légèrement plus élevée, le classificateur majoritaire a un F1 macro très inférieur.

#### Test — 1 500 observations

| Modèle | Accuracy | Precision macro | Recall macro | F1 macro | F1 weighted |
|---|---:|---:|---:|---:|---:|
| MLP initial | **0.9687** | **0.5720** | 0.6270 | 0.5924 | **0.9745** |
| **MLP optimisé** | 0.9673 | 0.5696 | **0.6408** | **0.5968** | 0.9737 |
| MLP régularisé | 0.9553 | 0.5322 | 0.6406 | 0.5778 | 0.9664 |
| Majoritaire (`No Failure`) | 0.9653 | 0.1609 | 0.1667 | 0.1637 | 0.9483 |

Le **MLP optimisé** est retenu car le **F1 macro Test le plus élevé parmi les modèles expérimentés est 0.5968**, et il avait aussi été sélectionné sur Validation. Le MLP initial affiche en revanche la meilleure accuracy Test ; cela souligne l'importance du choix de métrique.

#### Rappel par classe — MLP optimisé sur Test

| Classe | Détections correctes | Recall |
|---|---:|---:|
| Heat Dissipation Failure | 15 / 17 | 0.8824 |
| No Failure | 1411 / 1448 | 0.9744 |
| Overstrain Failure | 11 / 12 | 0.9167 |
| Power Failure | 13 / 14 | 0.9286 |
| Random Failures | **0 / 2** | **0.0000** |
| Tool Wear Failure | **1 / 7** | **0.1429** |

La matrice de confusion fait ressortir une bonne reconnaissance de certaines pannes physiques, mais un **échec pour `Random Failures`** et une **faible détection de `Tool Wear Failure`**. Sur les 1 448 cas `No Failure`, **37** ont été classés comme une panne (10 + 1 + 7 + 19 dans les différentes colonnes de panne). Avec seulement 2 observations Test de `Random Failures`, le recall de cette catégorie n'est pas estimable de façon stable.

### 9. Interprétation et limites — Partie 2

- Les performances sur classes rares sont plus importantes qu'une accuracy globale élevée ; prédire systématiquement `No Failure` atteint déjà **96,53 % d'accuracy** sur Test mais un **F1 macro de 0.1637**.
- La génération de données avec SMOTENC a permis de former des batches plus riches en pannes, **sans résoudre** la difficulté de certaines classes très rares.
- GridSearchCV améliore légèrement le F1 macro par rapport au MLP initial, mais l'amélioration sur Test est **modeste** (**0.5924 → 0.5968**).
- Dropout, weight decay et Early Stopping, testés conjointement, **n'ont pas amélioré** les scores Test dans cette configuration.
- La comparaison entre scores Train sur données synthétiques et scores Validation sur données naturelles doit être interprétée avec prudence.
- Le dataset est très déséquilibré et les classes très rares sont trop petites pour tirer des conclusions générales ; les résultats restent pédagogiques et ne constituent pas un système de surveillance industrielle validé.

### 10. Synthèse des apprentissages — Partie 2

Cette expérience m'a appris à : nettoyer et explorer des données industrielles, encoder une cible multiclasses et une variable catégorielle, standardiser les données sans fuite, conserver une séparation stratifiée, utiliser SMOTENC uniquement sur Train, développer une architecture MLP à six sorties avec `CrossEntropyLoss`, suivre Loss/Accuracy/F1 macro, interpréter des matrices de confusion et comparer GridSearchCV avec différentes régularisations. Elle a surtout montré pourquoi une **accuracy élevée peut masquer l'échec sur des classes minoritaires**.

## Synthèse générale du TP

Les deux expériences mettent en évidence la différence entre les objectifs de **régression** et de **classification** et la nécessité d'adapter les métriques au problème : **MAE, RMSE et R²** pour une cible continue ; **precision, recall/sensibilité, F1 et matrice de confusion** pour des classes déséquilibrées. J'ai appris à construire des réseaux PyTorch, exploiter les GPU Colab, paramétrer leurs entraînements, utiliser `GridSearchCV` avec une validation adaptée, prévenir les fuites de données et comparer un réseau profond à des références simples.

**Conclusion :** davantage de neurones, des hyperparamètres optimisés ou de la régularisation ne garantissent pas automatiquement de meilleurs résultats. Un bon protocole expérimental, des métriques pertinentes, des bases de comparaison et une analyse des erreurs sont indispensables.

## Organisation du dépôt

```text
atelier1-deeplearning/
├── README.md
└── notebooks/
    ├── 01_regression_nyse.ipynb
    └── 02_classification_maintenance.ipynb
```

*Cette arborescence représente l'organisation cible ; vérifier les chemins réels des notebooks dans le dépôt.*

## Exécution sur Google Colab

1. Ouvrir `01_regression_nyse.ipynb` ou `02_classification_maintenance.ipynb` dans Google Colab.
2. Sélectionner **Exécution → Modifier le type d'exécution → T4 GPU**, si disponible (le CPU reste possible).
3. Exécuter les cellules dans l'ordre. Le téléchargement des fichiers Kaggle est automatisé avec `kagglehub`.
4. Installer les bibliothèques utilisées au besoin : `torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `skorch`, `kagglehub`.
5. Conserver les cellules de prétraitement, d'équilibrage et d'entraînement dans leur ordre : une session Colab réinitialisée ne garde **pas** les variables et modèles Python en mémoire.

**Versions relevées durant la Partie 2 :** `scikit-learn 1.6.1`, `imbalanced-learn 0.14.2`, `skorch 1.4.0`. Les résultats dépendent des versions, du GPU et des sources de hasard ; ils peuvent légèrement varier lors d'une nouvelle exécution. Les tableaux présentent les **résultats effectivement observés** pendant le TP, sans inventer de performances non mesurées.

## Sources des datasets

- [NYSE — Kaggle (dgawlik/nyse)](https://www.kaggle.com/datasets/dgawlik/nyse)
- [Machine Predictive Maintenance Classification — Kaggle (shivamb)](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
