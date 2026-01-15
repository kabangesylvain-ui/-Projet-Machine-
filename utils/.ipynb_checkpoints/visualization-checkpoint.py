"""
utils/visualization.py
Fonctions de visualisation pour l'analyse de données
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_distributions(df, columns, figsize=(15, 10)):
    """
    Affiche les distributions des colonnes numériques.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    columns : list
        Liste des colonnes à visualiser
    figsize : tuple, default=(15, 10)
        Taille de la figure
    """
    # Calculer la disposition des subplots
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        if col in df.columns:
            ax = axes[idx]
            if df[col].dtype in ['int64', 'float64']:
                # Pour les colonnes numériques
                ax.hist(df[col].dropna(), bins=30, edgecolor='black', 
                       color='skyblue', alpha=0.7)
                ax.set_title(f'Distribution de {col}', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('Fréquence')
                ax.grid(True, alpha=0.3)
            else:
                # Pour les colonnes catégorielles
                value_counts = df[col].value_counts().head(10)
                ax.bar(range(len(value_counts)), value_counts.values, 
                      color='lightcoral', alpha=0.7)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_title(f'Top 10 valeurs de {col}', fontsize=12)
                ax.set_ylabel('Nombre')
                ax.grid(True, alpha=0.3)
    
    # Cacher les axes inutilisés
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Distributions des Variables', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, figsize=(12, 10), method='pearson'):
    """
    Affiche la matrice de corrélation avec une heatmap triangulaire.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    figsize : tuple, default=(12, 10)
        Taille de la figure
    method : str, default='pearson'
        Méthode de calcul de corrélation
    """
    # Sélectionner uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        print("⚠️  Pas assez de colonnes numériques pour la matrice de corrélation")
        return
    
    plt.figure(figsize=figsize)
    corr_matrix = numeric_df.corr(method=method)
    
    # Masque pour le triangle supérieur
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0, 
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .8},
                annot_kws={"size": 8})
    
    plt.title('Matrice de Corrélation', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(8, 6), 
                         normalize=False, cmap='Blues'):
    """
    Affiche la matrice de confusion avec labels.
    
    Parameters:
    -----------
    y_true : array-like
        Vraies étiquettes
    y_pred : array-like
        Étiquettes prédites
    labels : list, optional
        Noms des classes
    figsize : tuple, default=(8, 6)
        Taille de la figure
    normalize : bool, default=False
        Si True, normalise par ligne
    cmap : str, default='Blues'
        Colormap pour la heatmap
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Matrice de Confusion (Normalisée)'
    else:
        fmt = 'd'
        title = 'Matrice de Confusion'
    
    plt.figure(figsize=figsize)
    
    if labels is None:
        labels = [f'Classe {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, 
                annot=True, 
                fmt=fmt, 
                cmap=cmap,
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Nombre' if not normalize else 'Proportion'})
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('Vraies étiquettes')
    plt.xlabel('Étiquettes prédites')
    plt.tight_layout()
    plt.show()
    
    return cm


def plot_roc_curve(y_true, y_pred_proba, figsize=(8, 6), title='Courbe ROC'):
    """
    Affiche la courbe ROC avec AUC.
    
    Parameters:
    -----------
    y_true : array-like
        Vraies étiquettes (0 ou 1)
    y_pred_proba : array-like
        Probabilités prédites pour la classe positive
    figsize : tuple, default=(8, 6)
        Taille de la figure
    title : str, default='Courbe ROC'
        Titre du graphique
    """
    # Vérifier la forme de y_pred_proba
    if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
        # Prendre les probabilités pour la classe positive
        y_scores = y_pred_proba[:, 1]
    else:
        y_scores = y_pred_proba
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'Courbe ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Aléatoire (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return fpr, tpr, roc_auc


def plot_feature_importance(feature_importance, feature_names, top_n=20, 
                           figsize=(10, 8)):
    """
    Affiche l'importance des features d'un modèle.
    
    Parameters:
    -----------
    feature_importance : array-like
        Scores d'importance des features
    feature_names : list
        Noms des features
    top_n : int, default=20
        Nombre de features à afficher
    figsize : tuple, default=(10, 8)
        Taille de la figure
    """
    # Créer un DataFrame pour trier
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(importance_df)), importance_df['importance'], 
            color='steelblue', alpha=0.7)
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Features par Importance', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    return importance_df


def plot_pairplot(df, columns=None, hue=None, figsize=(12, 10)):
    """
    Crée un pairplot des variables numériques.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    columns : list, optional
        Colonnes à inclure (toutes les numériques par défaut)
    hue : str, optional
        Colonne pour colorer les points
    figsize : tuple, default=(12, 10)
        Taille de la figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columns) < 2:
        print("⚠️  Pas assez de colonnes numériques pour le pairplot")
        return
    
    # Limiter à un nombre raisonnable de colonnes
    if len(columns) > 8:
        columns = columns[:8]
        print(f"⚠️  Affichage des 8 premières colonnes seulement")
    
    pairplot_df = df[columns + ([hue] if hue else [])]
    
    g = sns.pairplot(pairplot_df, hue=hue, diag_kind='kde', 
                     plot_kws={'alpha': 0.6, 's': 30}, 
                     diag_kws={'fill': True})
    
    g.fig.set_size_inches(figsize)
    plt.suptitle('Pairplot des Variables', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_boxplots(df, columns=None, by=None, figsize=(12, 6)):
    """
    Affiche des boxplots pour les variables numériques.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données
    columns : list, optional
        Colonnes à visualiser
    by : str, optional
        Colonne de regroupement
    figsize : tuple, default=(12, 6)
        Taille de la figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = 4
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        if idx < len(axes):
            ax = axes[idx]
            if by and by in df.columns:
                # Boxplot groupé
                data_to_plot = [df[df[by] == val][col].dropna() 
                               for val in df[by].unique()]
                ax.boxplot(data_to_plot)
                ax.set_xticks(range(1, len(df[by].unique()) + 1))
                ax.set_xticklabels(df[by].unique(), rotation=45)
                ax.set_title(f'{col} par {by}', fontsize=10)
            else:
                # Boxplot simple
                ax.boxplot(df[col].dropna())
                ax.set_title(col, fontsize=10)
            
            ax.set_ylabel('Valeur')
            ax.grid(True, alpha=0.3)
    
    # Cacher les axes inutilisés
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Boxplots des Variables Numériques', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# Fonctions de compatibilité (pour éviter de modifier le code existant)
def plot_distribution(df, column, figsize=(10, 6)):
    """
    Fonction de compatibilité pour afficher une seule distribution.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame
    column : str
        Nom de la colonne
    figsize : tuple, default=(10, 6)
        Taille de la figure
    """
    plot_distributions(df, [column], figsize=figsize)


def plot_correlation(df, figsize=(12, 10)):
    """
    Fonction de compatibilité pour afficher la matrice de corrélation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame
    figsize : tuple, default=(12, 10)
        Taille de la figure
    """
    plot_correlation_matrix(df, figsize=figsize)