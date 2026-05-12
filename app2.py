import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from clustering2 import *

# Page configuration
st.set_page_config(page_title="Advanced Clustering App", layout="wide")

st.title("Clustering App")
st.markdown("---")

# Data loading section
col1, col2 = st.columns(2)
with col1:
    data_source = st.radio("Select data source:", ["Upload CSV", "Kaggle URL"])

df = None

if data_source == "Upload CSV":
    file = st.file_uploader("Upload your dataset", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
else:
    kaggle_url = st.text_input("Enter Kaggle dataset URL:", 
                               placeholder="https://www.kaggle.com/datasets/username/dataset-name")
    if kaggle_url and st.button("Load Dataset"):
        with st.spinner("Loading dataset from Kaggle..."):
            df = load_data("Kaggle URL", kaggle_url=kaggle_url)
            if df is not None:
                st.success(f"Dataset loaded successfully! Shape: {df.shape}")

if df is None:
    st.stop()

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.write(f"**Dataset shape:** {df.shape[0]} rows, {df.shape[1]} columns")

# Feature selection method
st.subheader("Feature Selection")
feature_method = st.radio("Select feature selection method:", 
                          ["Manual Selection", "PCA Reduction"])

features = []
pca_components = 2

if feature_method == "Manual Selection":
    features = st.multiselect(
        "Select features for clustering (minimum 2):",
        df.columns.tolist()
    )
    if len(features) < 2:
        st.warning("Please select at least 2 features")
        st.stop()
else:  # PCA Reduction
    all_features = st.multiselect(
        "Select features for PCA (optional - if none selected, all numerical will be used):",
        df.columns.tolist()
    )
    if not all_features:
        # Use all numerical features
        all_features = df.select_dtypes(include=[np.number]).columns.tolist()
        st.info(f"Using all {len(all_features)} numerical features for PCA")
    
    pca_components = st.slider("Number of PCA components:", 2, min(10, len(all_features)), 2)
    
    if len(all_features) < 2:
        st.warning("Need at least 2 features for PCA")
        st.stop()

# Preprocessing options
st.subheader("Preprocessing Options")
col1, col2, col3 = st.columns(3)
with col1:
    scale_data = st.checkbox("Scale numerical features", value=True)
with col2:
    encode_data = st.checkbox("Encode categorical features", value=True)
with col3:
    handle_missing = st.selectbox("Handle missing values", ["drop", "fill"])

# Clustering model selection
st.subheader("Clustering Model")
model_name = st.selectbox(
    "Select clustering algorithm:",
    ["K-Means", "K-Medoids", "DBSCAN", "Hierarchical"]
)

# Model parameters
params = {}
if model_name == "K-Means" or model_name == "K-Medoids" or model_name == "Hierarchical":
    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
    params['n_clusters'] = n_clusters
    
    if model_name == "Hierarchical":
        linkage = st.selectbox("Linkage criterion:", ["ward", "complete", "single", "average"])
        params['linkage'] = linkage
elif model_name == "DBSCAN":
    col1, col2 = st.columns(2)
    with col1:
        eps = st.slider("Epsilon (neighborhood radius):", 0.1, 5.0, 0.5, 0.1)
        params['eps'] = eps
    with col2:
        min_samples = st.slider("Minimum samples per cluster:", 2, 20, 5)
        params['min_samples'] = min_samples

# Run clustering button
if st.button("Run Clustering", type="primary"):
    with st.spinner("Processing data and running clustering..."):
        # Preprocess data
        if feature_method == "Manual Selection":
            X = preprocessing(df, features, scale=scale_data, encode=encode_data, 
                            handle_missing=handle_missing)
            feature_names = features
        else:  # PCA Reduction
            # Preprocess all selected features
            X_full = preprocessing(df, all_features, scale=scale_data, encode=encode_data,
                                  handle_missing=handle_missing)
            # Apply PCA
            X, explained_var = reduce_dimensions_pca(X_full, n_components=pca_components)
            feature_names = [f"PC{i+1} ({var:.2%})" for i, var in enumerate(explained_var)]
            
            st.info(f"PCA explained variance: {explained_var.sum():.2%} total")
            for i, var in enumerate(explained_var):
                st.write(f"  - PC{i+1}: {var:.2%}")
        
        # Get and run clustering model
        clustering_func = get_clustering_model(model_name)
        
        if model_name == "DBSCAN":
            model, labels = clustering_func(X, **params)
        elif model_name == "Hierarchical":
            model, labels = clustering_func(X, **params)
        else:
            model, labels = clustering_func(X, **params)
        
        # Add labels to dataframe (for manual selection) or create result dataframe
        if feature_method == "Manual Selection":
            result_df = df[features].copy()
            result_df["Cluster"] = labels
        else:
            result_df = pd.DataFrame(X, columns=feature_names)
            result_df["Cluster"] = labels
        
        # Display results
        st.success(f"Clustering completed! Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        
        # Evaluation metrics
        st.subheader("Clustering Evaluation")
        sil_score, db_score = evaluate_clustering(X, labels)
        
        if sil_score is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{sil_score:.3f}")
                st.caption("Range: -1 to 1 (higher is better)")
            with col2:
                st.metric("Davies-Bouldin Index", f"{db_score:.3f}")
                st.caption("Lower is better")
            with col3:
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                st.metric("Number of Clusters", n_clusters_found)
        else:
            st.warning("Could not compute evaluation metrics (need at least 2 clusters)")
        
        # Visualization
        st.subheader("Cluster Visualization")
        
        if X.shape[1] == 2:  # 2D visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                                alpha=0.6, s=50)
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_title(f"{model_name} Clustering Results")
            plt.colorbar(scatter, ax=ax, label='Cluster')
            st.pyplot(fig)
        elif X.shape[1] == 3:  # 3D visualization
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, 
                               cmap='viridis', alpha=0.6, s=50)
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_zlabel(feature_names[2])
            ax.set_title(f"{model_name} Clustering Results (3D)")
            plt.colorbar(scatter, ax=ax, label='Cluster')
            st.pyplot(fig)
        else:
            st.info(f"Data has {X.shape[1]} dimensions. Showing first 2 features/components")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                                alpha=0.6, s=50)
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.set_title(f"{model_name} Clustering Results (First 2 dimensions)")
            plt.colorbar(scatter, ax=ax, label='Cluster')
            st.pyplot(fig)
        
        # Show clustered data
        st.subheader("Clustered Data Preview")
        st.dataframe(result_df.head(20))
        
        # Download option
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download clustered data as CSV",
            data=csv,
            file_name=f"clustered_data_{model_name.lower()}.csv",
            mime="text/csv"
        )
        
        # Cluster statistics
        if st.checkbox("Show cluster statistics"):
            st.subheader("Cluster Statistics")
            if feature_method == "Manual Selection":
                cluster_stats = result_df.groupby('Cluster')[features].mean()
            else:
                cluster_stats = result_df.groupby('Cluster')[feature_names].mean()
            st.dataframe(cluster_stats)

st.markdown("---")
st.caption("Tip: For DBSCAN, clusters with label -1 are considered noise/outliers")

