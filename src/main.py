import pandas as pd
from dataprocessing import process_data
from pca import perform_pca
from plotting import plot_3d_pca

# Dataprocessing
X_scaled, y = process_data()

# PCA
X_pca = perform_pca(X_scaled)

# Plotting
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['target'] = y

plot_3d_pca(df_pca)
