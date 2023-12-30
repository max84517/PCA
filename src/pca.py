from sklearn.decomposition import PCA

def perform_pca(X_scaled):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca
