# utils/models.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def train_baseline_model(X_train, y_train):
    """Entraîne un modèle baseline (régression logistique)"""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_improved_model(X_train, y_train):
    """Entraîne un modèle amélioré (Random Forest)"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_model(X_train, y_train, model_type="baseline"):
    """
    Entraîne un modèle selon le type spécifié
    
    Parameters:
    -----------
    X_train : array-like
        Features d'entraînement
    y_train : array-like
        Labels d'entraînement
    model_type : str
        'baseline' pour régression logistique
        'improved' pour Random Forest
    
    Returns:
    --------
    model : estimator
        Modèle entraîné
    """
    if model_type == "baseline":
        return train_baseline_model(X_train, y_train)
    elif model_type == "improved":
        return train_improved_model(X_train, y_train)
    else:
        raise ValueError("model_type must be 'baseline' or 'improved'")

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle avec différentes métriques"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics, y_pred

def predict_model(model, X):
    """
    Fait des prédictions avec un modèle entraîné
    
    Parameters:
    -----------
    model : estimator
        Modèle entraîné
    X : array-like
        Données à prédire
    
    Returns:
    --------
    predictions : array
        Prédictions
    """
    return model.predict(X)

def cross_validate_model(model, X, y, cv=5):
    """Effectue une validation croisée"""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

def compare_clustering_methods(X, n_clusters_range=range(2, 11)):
    """Compare différentes méthodes de clustering"""
    inertias = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, inertias, 'bo-')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Méthode du Coude pour K-Means')
    plt.show()
    
    return inertias

   # les fonctions de régression 
def train_baseline_regression(X_train, y_train):
    """Modèle baseline pour régression: Régression Linéaire"""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_improved_regression(X_train, y_train):
    """Modèle amélioré pour régression: Random Forest Regressor"""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_regression_model(model, X_test, y_test):
    """Évalue un modèle de régression"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    print(f"📊 Métriques de régression:")
    print(f"   MSE: {metrics['mse']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   R²: {metrics['r2']:.4f}")
    
    return metrics, y_pred