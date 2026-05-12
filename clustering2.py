import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
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
            # Extract dataset path from Kaggle URL
            import opendatasets as od
            import os
            
            # Download dataset
            od.download(kaggle_url)
            # Get the downloaded folder name
            folder_name = kaggle_url.split('/')[-1]
            # Find CSV file in the downloaded folder
            csv_files = [f for f in os.listdir(folder_name) if f.endswith('.csv')]
            if csv_files:
                return pd.read_csv(os.path.join(folder_name, csv_files[0]))
    return None

def preprocessing(df, features, scale=True, encode=True, handle_missing='drop'):
    """
    Preprocess data for clustering with options for scaling and encoding
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    features : list
        List of feature names to use
    scale : bool
        Whether to scale numerical features
    encode : bool
        Whether to encode categorical features
    handle_missing : str
        How to handle missing values ('drop', 'fill')
    """
    X = df[features].copy()
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        if handle_missing == 'drop':
            X = X.dropna()
            st.info(f"Dropped rows with missing values. Remaining: {len(X)} rows")
        else:
            X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
    
    # Separate numerical and categorical columns
    numerical = X.select_dtypes(include=[np.number]).columns
    categorical = X.select_dtypes(include=['object']).columns
    
    # Scale numerical features
    if scale and len(numerical) > 0:
        scaler = StandardScaler()
        X[numerical] = scaler.fit_transform(X[numerical])
        st.info(f"Scaled {len(numerical)} numerical features")
    
    # Encode categorical features
    if encode and len(categorical) > 0:
        encoder = LabelEncoder()
        for col in categorical:
            X[col] = encoder.fit_transform(X[col].astype(str))
        st.info(f"Encoded {len(categorical)} categorical features")
    
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
    """K-Medoids clustering"""
    model = KMedoids(n_clusters=n_clusters, random_state=random_state)
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
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        return sil_score, db_score
    return None, None
