import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

def load_data(source, file=None, kaggle_url=None):
    """Load data from CSV file or Kaggle URL"""
    if source == "Upload CSV":
        if file is not None:
            return pd.read_csv(file)
    elif source == "Kaggle URL":
        if kaggle_url:
            st.warning("Kaggle download requires additional setup. Please download CSV manually.")
            return None
    return None

def preprocessing(df, features, scale=True, encode=True, handle_missing='drop'):
    """
    Preprocess data for clustering with options for scaling and encoding
    """
    X = df[features].copy()
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        if handle_missing == 'drop':
            X = X.dropna()
        else:
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else "Unknown")
    
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
    
    return X.values

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
    """K-Medoids clustering (using K-Means as fallback if sklearn-extra not available)"""
    try:
        from sklearn_extra.cluster import KMedoids
        model = KMedoids(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(X)
        return model, labels
    except ImportError:
        # Fallback to K-Means
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
    """Evaluate clustering performance"""
    if len(set(labels)) > 1 and len(labels) > 1:
        try:
            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            return sil_score, db_score
        except:
            return None, None
    return None, None
