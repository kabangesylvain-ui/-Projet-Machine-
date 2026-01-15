"""
Package utils - Utilities for data science projects
"""

from .preprocessing import (
    load_df,
    clean_df,
    encode_labels,
    remove_outliers,
    normalize_df,
    split_data
)

from .models import (
    train_model,           # Fonction principale d'entraînement
    train_baseline_model,  # Modèle baseline
    train_improved_model,  # Modèle amélioré
    evaluate_model,        # Évaluation
    predict_model,         # Prédiction
    cross_validate_model,  # Validation croisée
    compare_clustering_methods  # Clustering
)

from .visualization import (
    plot_distribution,
    plot_correlation,
    plot_confusion_matrix
)

__version__ = '1.0.0'
__author__ = 'SYLVAIN'
__email__ = 'kabange.sylvain@horizon-university.tn'
__description__ = 'Package utils pour le prétraitement de données'
__url__ = 'https://github.com/votre-compte/votre-projet'
__license__ = 'MIT'

__all__ = [
    # Préprocessing
    'load_df',
    'clean_df',
    'encode_labels',
    'remove_outliers',
    'normalize_df',
    'split_data',
    
    # Modèles
    'train_model',
    'train_baseline_model',
    'train_improved_model',
    'evaluate_model',
    'predict_model',
    'cross_validate_model',
    'compare_clustering_methods',
    
    # Visualisation
    'plot_distribution',
    'plot_correlation',
    'plot_confusion_matrix'
]