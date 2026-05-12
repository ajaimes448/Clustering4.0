import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

def load_data(source, file=None, kaggle_url=None):
    """Load data from CSV file (Kaggle URL not directly supported)"""
    if source == "Upload CSV" and file is not None:
        return pd.read_csv(file)
    elif source == "Kaggle URL":
        # No se implementa automáticamente porque requiere credenciales
        # Se devuelve None y el mensaje se maneja en app.py
        return None
    return None

def preprocessing(df, features, scale=True, encode=True, handle_missing='drop'):
    """
    Preprocess data for clustering with options for scaling and encoding.
    Returns: (X_values, kept_index) where kept_index are the row indices that remain.
    """
    X = df[features].copy()
    
    # Handle missing values
    if handle_missing == 'drop':
        X = X.dropna()
    else:  # fill
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].mean())
            else:
                mode_val = X[col].mode()
                X[col] = X[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")
    
    # Capturar el índice después de manejar missing
    kept_index = X.index
    
    # Separate numerical and categorical columns
    numerical = X.select_dtypes(include=[np.number]).columns
    categorical = X.select_dtypes(include=['object']).columns
    
    # Scale numerical features
    if scale and len(numerical) > 0:
        scaler = StandardScaler()
        X[numerical] = scaler.fit_transform(X[numerical])
    
    # Encode categorical features
    if encode and len(categorical) > 0:
        encoder = LabelEncoder()
        for col in categorical:
            X[col] = encoder.fit_transform(X[col].astype(str))
    
    return X.values, kept_index

def reduce_dimensions_pca(X, n_components=2):
    """Apply PCA dimensionality reduction"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    return X_pca, explained_variance

def clustering_kmeans(X, n_clusters, random_state=42):
    """K-Means clustering"""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels

def clustering_kmedoids(X, n_clusters, random_state=42):
    """K-Medoids clustering (uses K-Means if sklearn_extra not available)"""
    try:
        from sklearn_extra.cluster import KMedoids
        model = KMedoids(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(X)
        return model, labels
    except ImportError:
        # Fallback to K-Means with warning (warning shown in app.py)
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        return model, labels

def clustering_dbscan(X, eps=0.5, min_samples=5):
    """DBSCAN clustering"""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels

def clustering_hierarchical(X, n_clusters, linkage='ward'):
    """Hierarchical clustering"""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return model, labels

def get_clustering_model(model_name):
    """Return the appropriate clustering function"""
    models = {
        "K-Means": clustering_kmeans,
        "K-Medoids": clustering_kmedoids,
        "DBSCAN": clustering_dbscan,
        "Hierarchical": clustering_hierarchical
    }
    return models.get(model_name)

def evaluate_clustering(X, labels):
    """Evaluate clustering performance, returns (silhouette, davies_bouldin) or (None, None)"""
    unique_labels = set(labels)
    if len(unique_labels) > 1 and len(labels) > 1:
        try:
            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            return sil_score, db_score
        except Exception:
            return None, None
    return None, None
