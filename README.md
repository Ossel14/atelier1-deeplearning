# Atelier 1 - Deep Learning avec PyTorch
# Atelier 1 — Deep Learning avec PyTorch

**TP : Régression et classification multiclasses avec des réseaux DNN/MLP**  
**Environnement :** Google Colab (PyTorch, GPU CUDA), pandas, NumPy, scikit-learn, skorch, Matplotlib, Seaborn  
**État d'avancement :** Partie 1 (régression) réalisée ; Partie 2 (classification) à venir.

## Objectifs

Ce TP vise à mettre en œuvre des réseaux de neurones entièrement connectés (*Multi-Layer Perceptron*, MLP) avec PyTorch pour deux tâches :

1. **Régression :** prédiction d'une valeur continue à partir des données boursières NYSE.
2. **Classification multiclasses :** prédiction du type de défaillance d'une machine (à réaliser dans la Partie 2).

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

## Partie 2 — Classification multiclasses (à réaliser)

**Dataset :** [Machine Predictive Maintenance Classification — Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

**Objectif envisagé :** prédire la variable multiclasses `Failure Type` à l'aide d'un MLP PyTorch, après nettoyage, encodage et normalisation des données. Les étapes prévues sont l'EDA, le traitement du déséquilibre des classes **sur Train uniquement**, l'optimisation des hyperparamètres, les courbes Loss/Accuracy, la matrice de confusion, l'accuracy, le recall/sensibilité et le F1-score, puis la comparaison avec des variantes régularisées.

*Cette partie n'est pas encore implémentée ; aucun résultat de classification n'est revendiqué à ce stade.*

## Organisation envisagée du dépôt

```text
atelier1-deeplearning/
├── README.md
└── notebooks/
    ├── 01_regression_nyse.ipynb
    └── 02_classification_maintenance.ipynb  # à venir
```

## Exécution

1. Ouvrir le notebook de régression dans **Google Colab**.
2. Exécuter les cellules dans l'ordre ; le notebook utilise `kagglehub` pour récupérer le dataset NYSE.
3. Installer au besoin les dépendances utilisées dans les cellules (dont `skorch`).
4. Sélectionner un runtime GPU CUDA si disponible ; le modèle peut également être exécuté sur CPU.
5. Recalculer les entraînements pour reproduire les graphiques et les métriques. Les résultats peuvent légèrement varier selon la session, les graines aléatoires et les versions logicielles.

**Remarque :** les fichiers de datasets et les modèles entraînés ne sont pas inclus dans ce README. Les valeurs ci-dessus proviennent des sorties obtenues durant le TP.
