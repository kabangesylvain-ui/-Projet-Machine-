# utils/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ============================================================================
# FONCTIONS DE BASE DE PRÉTRAITEMENT
# ============================================================================

def load_df(file_path):
    """Charge un fichier CSV dans un DataFrame pandas"""
    return pd.read_csv(file_path)

def clean_df(df):
    """
    Nettoie le DataFrame en supprimant les colonnes vides 
    et en remplissant les valeurs manquantes
    """
    if isinstance(df, tuple):
        # Si c'est un tuple, prendre le premier élément (le DataFrame)
        df = df[0] if isinstance(df[0], pd.DataFrame) else df
    
    df_clean = df.copy()
    
    # 1. Supprimer les colonnes entièrement vides
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # 2. Remplir les valeurs manquantes numériques avec la moyenne
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # 3. Remplir les valeurs manquantes catégorielles avec le mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
            else:
                df_clean[col] = df_clean[col].fillna('Unknown')
    
    return df_clean

def encode_labels(df, columns=None):
    """
    Encode les colonnes catégorielles avec LabelEncoder
    
    Args:
        df: DataFrame pandas
        columns: Liste des colonnes à encoder (si None, encode toutes les colonnes catégorielles)
    
    Returns:
        df_encoded: DataFrame encodé
        le_dict: Dictionnaire des encodeurs
    """
    if isinstance(df, tuple):
        # Si c'est un tuple, prendre le premier élément (le DataFrame)
        df = df[0] if isinstance(df[0], pd.DataFrame) else df
    
    df_encoded = df.copy()
    le_dict = {}
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object']).columns
    
    for col in columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            le_dict[col] = le
    
    return df_encoded, le_dict

def normalize_df(df, columns=None, exclude_cols=None):
    """
    Normalise les colonnes spécifiées du DataFrame
    
    Args:
        df: DataFrame pandas
        columns: Liste des colonnes à normaliser (si None, toutes les colonnes numériques)
        exclude_cols: Liste des colonnes à exclure de la normalisation
    
    Returns:
        df_normalized: DataFrame normalisé
        scaler: Objet StandardScaler utilisé
    """
    if isinstance(df, tuple):
        # Si c'est un tuple, prendre le premier élément (le DataFrame)
        df = df[0] if isinstance(df[0], pd.DataFrame) else df
    
    df_normalized = df.copy()
    
    # Déterminer les colonnes à normaliser
    if columns is None:
        columns = df_normalized.select_dtypes(include=['float64', 'int64']).columns
    
    # Exclure les colonnes si spécifié
    if exclude_cols is not None:
        columns = [col for col in columns if col not in exclude_cols]
    
    # Normaliser seulement s'il y a des colonnes à normaliser
    if len(columns) > 0:
        scaler = StandardScaler()
        df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
        return df_normalized, scaler
    else:
        return df_normalized, None

def normalize_features(df, feature_cols=None):
    """
    Normalise les caractéristiques spécifiées
    
    Args:
        df: DataFrame
        feature_cols: Colonnes à normaliser (si None, toutes les colonnes numériques)
    
    Returns:
        df_normalized: DataFrame normalisé
        scaler: Objet StandardScaler utilisé
    """
    return normalize_df(df, columns=feature_cols)

# ============================================================================
# FONCTIONS AVANCÉES DE PRÉTRAITEMENT
# ============================================================================

def handle_missing_values(df, strategy='median'):
    """
    Gère les valeurs manquantes avec différentes stratégies
    
    Args:
        df: DataFrame pandas
        strategy: 'mean', 'median', 'mode' ou 'drop'
    
    Returns:
        DataFrame sans valeurs manquantes
    """
    if isinstance(df, tuple):
        # Si c'est un tuple, prendre le premier élément (le DataFrame)
        df = df[0] if isinstance(df[0], pd.DataFrame) else df
    
    df_copy = df.copy()
    
    if strategy == 'drop':
        return df_copy.dropna()
    
    for col in df_copy.columns:
        if df_copy[col].isnull().sum() > 0:
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                mode_val = df_copy[col].mode()
                if not mode_val.empty:
                    df_copy[col].fillna(mode_val[0], inplace=True)
                else:
                    # Si pas de mode (toutes valeurs uniques), utiliser une valeur par défaut
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        df_copy[col].fillna(0, inplace=True)
                    else:
                        df_copy[col].fillna('Unknown', inplace=True)
            else:
                # Stratégie par défaut
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
                else:
                    mode_val = df_copy[col].mode()
                    if not mode_val.empty:
                        df_copy[col].fillna(mode_val[0], inplace=True)
                    else:
                        df_copy[col].fillna('Unknown', inplace=True)
    
    return df_copy

def remove_outliers(df, columns=None, threshold=3):
    """
    Supprime les outliers avec la méthode Z-score
    
    Args:
        df: DataFrame pandas
        columns: Colonnes à vérifier pour les outliers (si None, toutes les colonnes numériques)
        threshold: Seuil Z-score pour considérer un point comme outlier
    
    Returns:
        DataFrame sans outliers
    """
    if isinstance(df, tuple):
        # Si c'est un tuple, prendre le premier élément (le DataFrame)
        df = df[0] if isinstance(df[0], pd.DataFrame) else df
    
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=['float64', 'int64']).columns
    
    mask = pd.Series([True] * len(df_copy))
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            # Vérifier si la colonne a un écart-type non nul
            if df_copy[col].std() > 0:
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                mask = mask & (z_scores < threshold)
    
    return df_copy[mask].reset_index(drop=True)

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement et de test
    
    Args:
        df: DataFrame complet
        target_column: Nom de la colonne cible
        test_size: Proportion pour le test set
        random_state: Seed pour la reproductibilité
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if isinstance(df, tuple):
        # Si c'est un tuple, prendre le premier élément (le DataFrame)
        df = df[0] if isinstance(df[0], pd.DataFrame) else df
    
    if target_column not in df.columns:
        raise ValueError(f"Colonne cible '{target_column}' non trouvée dans le DataFrame")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def normalize_train_test(X_train, X_test, feature_cols=None):
    """
    Normalise les données d'entraînement et de test de manière cohérente
    
    Args:
        X_train: Données d'entraînement
        X_test: Données de test
        feature_cols: Colonnes à normaliser
    
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    if isinstance(X_train, tuple):
        # Si c'est un tuple, prendre le premier élément (le DataFrame)
        X_train = X_train[0] if isinstance(X_train[0], pd.DataFrame) else X_train
        X_test = X_test[0] if isinstance(X_test[0], pd.DataFrame) else X_test
    
    if feature_cols is None:
        # Sélectionner uniquement les colonnes numériques
        feature_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    scaler = StandardScaler()
    
    # Créer des copies pour éviter les warnings
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Normaliser uniquement si nous avons des colonnes à normaliser
    if len(feature_cols) > 0:
        X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
        X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
    
    return X_train_scaled, X_test_scaled, scaler

# ============================================================================
# PIPELINE COMPLET DE PRÉTRAITEMENT
# ============================================================================

def preprocess_pipeline(file_path, target_column=None, 
                        missing_strategy='median',
                        remove_outliers_flag=False,
                        encode_categorical=True,
                        normalize_features_flag=True,
                        test_size=0.2,
                        random_state=42):
    """
    Pipeline complet de prétraitement
    
    Args:
        file_path: Chemin vers le fichier CSV
        target_column: Nom de la colonne cible
        missing_strategy: Stratégie pour les valeurs manquantes
        remove_outliers_flag: Si True, supprime les outliers
        encode_categorical: Si True, encode les variables catégorielles
        normalize_features_flag: Si True, normalise les features
        test_size: Proportion pour le test set
        random_state: Seed pour la reproductibilité
    
    Returns:
        Si target_column spécifié: X_train, X_test, y_train, y_test, preprocess_info
        Sinon: df_processed, preprocess_info
    """
    # 1. Chargement des données
    print("1. Chargement des données...")
    df = load_df(file_path)
    print(f"   Shape initial: {df.shape}")
    print(f"   Colonnes: {list(df.columns)}")
    
    # 2. Nettoyage de base
    print("\n2. Nettoyage de base...")
    df = clean_df(df)
    print(f"   Après nettoyage: {df.shape}")
    
    # 3. Gestion des valeurs manquantes
    print("\n3. Gestion des valeurs manquantes...")
    print(f"   Valeurs manquantes avant: {df.isnull().sum().sum()}")
    df = handle_missing_values(df, strategy=missing_strategy)
    print(f"   Valeurs manquantes après: {df.isnull().sum().sum()}")
    
    # 4. Suppression des outliers (optionnel)
    if remove_outliers_flag:
        print("\n4. Suppression des outliers...")
        df_before = df.shape[0]
        df = remove_outliers(df, threshold=3)
        removed_count = df_before - df.shape[0]
        print(f"   Échantillons avant: {df_before}, après: {df.shape[0]}")
        print(f"   {removed_count} outliers supprimés")
    
    # 5. Encodage des variables catégorielles (optionnel)
    if encode_categorical:
        print("\n5. Encodage des variables catégorielles...")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            df, le_dict = encode_labels(df, categorical_cols)
            print(f"   Colonnes encodées: {categorical_cols}")
        else:
            le_dict = {}
            print("   Aucune colonne catégorielle à encoder")
    else:
        le_dict = {}
        categorical_cols = []
    
    # 6. Vérifier si une colonne cible est spécifiée
    if target_column is not None:
        print(f"\n6. Préparation avec colonne cible: '{target_column}'")
        
        # Vérifier que la colonne cible existe
        if target_column not in df.columns:
            raise ValueError(f"Colonne cible '{target_column}' non trouvée dans le DataFrame. "
                           f"Colonnes disponibles: {list(df.columns)}")
        
        # Diviser les données
        X_train, X_test, y_train, y_test = split_data(
            df, target_column, test_size=test_size, random_state=random_state
        )
        
        # Normaliser les features (optionnel)
        if normalize_features_flag:
            print("7. Normalisation des caractéristiques...")
            # Exclure la colonne cible si elle est numérique
            feature_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
            
            # Si la cible est numérique, elle pourrait être dans feature_cols
            # On la retire si c'est le cas
            if target_column in feature_cols:
                feature_cols = feature_cols.drop(target_column)
            
            X_train_scaled, X_test_scaled, scaler = normalize_train_test(
                X_train, X_test, feature_cols
            )
            print(f"   {len(feature_cols)} colonnes normalisées")
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
            scaler = None
            feature_cols = []
        
        # Informations de prétraitement
        preprocess_info = {
            'original_shape': df.shape,
            'label_encoders': le_dict,
            'scaler': scaler,
            'feature_cols': list(feature_cols),
            'train_shape': X_train_scaled.shape,
            'test_shape': X_test_scaled.shape,
            'target_column': target_column,
            'target_classes': list(np.unique(y_train)) if hasattr(y_train, 'unique') else []
        }
        
        return X_train_scaled, X_test_scaled, y_train, y_test, preprocess_info
    
    else:
        # Pas de colonne cible spécifiée
        print("\n6. Préparation sans colonne cible spécifiée")
        
        # Normalisation (optionnel)
        if normalize_features_flag:
            print("7. Normalisation des données...")
            # Normaliser toutes les colonnes numériques
            df_normalized, scaler = normalize_df(df)
            print(f"   Normalisation terminée")
        else:
            df_normalized = df.copy()
            scaler = None
        
        # Informations de prétraitement
        preprocess_info = {
            'original_shape': df.shape,
            'processed_shape': df_normalized.shape,
            'label_encoders': le_dict,
            'scaler': scaler,
            'categorical_cols': categorical_cols
        }
        
        return df_normalized, preprocess_info

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_data_summary(df):
    """Retourne un résumé des données"""
    if isinstance(df, tuple):
        # Si c'est un tuple, prendre le premier élément (le DataFrame)
        df = df[0] if isinstance(df[0], pd.DataFrame) else df
    
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'numeric_columns': list(df.select_dtypes(include=['float64', 'int64']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 ** 2)  # en MB
    }
    return summary

def save_processed_data(df, output_path, index=False):
    """Sauvegarde le DataFrame traité"""
    if isinstance(df, tuple):
        # Si c'est un tuple, prendre le premier élément (le DataFrame)
        df = df[0] if isinstance(df[0], pd.DataFrame) else df
    
    df.to_csv(output_path, index=index)
    print(f"Données sauvegardées dans: {output_path}")
    return output_path