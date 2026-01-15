# Projet Data Science - Prédiction du Niveau de Cancer
**Encadrement :** Dr Khouloud Chelbi  
**Durée :** 3–4 semaines  
**Travail en groupe**

## 📋 Description du Projet
Ce projet a pour objectif de développer un modèle de prédiction du niveau de cancer à partir de données médicales. Il suit un pipeline complet de data science, de l'analyse exploratoire à l'évaluation des modèles.

## 🎯 Objectifs
Ce projet a pour but de :
• Mettre en pratique les concepts vus en cours
• Apprendre à organiser un projet ML réel
• Travailler en environnement professionnel (JupyterLab, virtual env)
• Comprendre les principes de collaboration (Git, dossiers structurés)
• Créer un pipeline ML complet (EDA → Préparation → Modélisation → Évaluation)
## 📁 Structure du Projet
project/
├── data/
│ └── Cance_Pediction_Dataset.csv
├── notebooks/
│ ├── EDA.ipynb
│ ├── data_preparation.ipynb
│ ├── modeling.ipynb
│ └── evaluation.ipynb
├── utils/
│ ├── preprocessing.py
│ ├── models.py
│ └── visualization.py
├── results/
├── README.md
└── requirements.txt


# Installation

# 1. Prérequis
- Python 3.10 ou supérieur
- Git (pour la gestion de version)

# 2. Configuration de l'environnement
# Créer l'environnement
conda create -n cancer_prediction python=3.10
conda activate cancer_prediction
# Installer les dépendances
pip install -r requirements.txt
# (3. Configuration de Jupyter) Installer le kernel
python -m ipykernel install --user --name=cancer_prediction --display-name="Cancer Prediction"
# Lancer JupyterLab
jupyter lab

📊 Pipeline de Travail
# Phase 1 : Analyse Exploratoire (EDA)
Chargement et inspection des données

Analyse statistique descriptive

Visualisations (distributions, corrélations)

Détection des problèmes (valeurs manquantes, outliers)

# Phase 2 : Préparation des Données
Nettoyage et imputation

Encodage des variables catégorielles
Split train/test (80/20)

# Phase 3 : Modélisation
Modèles testés :

Baseline : Régression Logistique

Amélioré : Random Forest

Avancé : XGBoost (optionnel)

Optimisation des hyperparamètres avec GridSearchCV

Validation croisée (5 folds)

# Phase 4 : Évaluation
Métriques : Accuracy, Precision, Recall, F1-Score

Visualisations : Matrice de confusion, Courbe ROC

Interprétation des résultats

Recommandations
Normalisation/standardisation

# 🛠️ Technologies Utilisées
Langage : Python 3.10+

Data Processing : pandas, numpy

Machine Learning : scikit-learn

Visualisation : matplotlib, seaborn

Environnement : JupyterLab

Versioning : Git/GitHub

# 📈 Métriques de Performance
Classification Metrics
Métrique	Formule	Objectif
Accuracy	(TP+TN)/(TP+TN+FP+FN)	Précision globale
Precision	TP/(TP+FP)	Qualité des prédictions positives
Recall	TP/(TP+FN)	Capacité à détecter les vrais positifs
F1-Score	2(PrecisionRecall)/(Precision+Recall)	Moyenne harmonique
AUC-ROC	Aire sous la courbe ROC	Performance globale

# Objectifs
Accuracy cible : > 85%

Recall pour les cas critiques : > 90%

AUC-ROC : > 0.85

# Planning du Projet
Semaine	Activités	Livrables
1	Setup, EDA	Notebook 01, Rapport EDA
2	Préprocessing	Notebook 02, Données nettoyées
3	Modélisation	Notebook 03, Modèles entraînés
4	Évaluation, Rapport	Notebook 04, Présentation finale

## 1. Exécution séquentielle des notebooks
# Exécuter dans l'ordre :
1. 01_EDA.ipynb
2. 02_data_preparation.ipynb
3. 03_modeling.ipynb
4. 4. 04_evaluation.ipynb
## 2. Utilisation des modules utils
# Dans vos notebooks
import sys
sys.path.append('../')

from utils.preprocessing import load_data, clean_data
from utils.visualization import plot_correlation_matrix

# Charger les données
df = load_data('../data/Cancer_Prediction_Dataset.csv')

# Nettoyer les données
df_clean = clean_data(df)

# Visualiser
plot_correlation_matrix(df_clean)
## 3. Entraînement d'un modèle
from sklearn.ensemble import RandomForestClassifier
from utils.models import train_model, evaluate_model

# Entraîner le modèle
model = train_model(X_train, y_train, model_type='random_forest')

# Évaluer
metrics = evaluate_model(model, X_test, y_test)

## Problèmes courants
1. ModuleNotFoundError
# Solution : Ajouter le chemin
import sys
import os
sys.path.append(os.path.abspath('..'))
# 2. Erreur de chargement des données
# Vérifier le chemin
ls data/
# S'assurer que le fichier existe
# 3. Problèmes de mémoire
# Réduire la taille des données
df_sample = df.sample(frac=0.5, random_state=42)
# 4. Graphiques qui n'apparaissent pas
# Dans Jupyter
%matplotlib inline
# Commandes utiles
# Vérifier l'installation
python -c "import pandas; print(pandas.__version__)"

# Lister les packages installés
pip list

# Mettre à jour pip
python -m pip install --upgrade pip