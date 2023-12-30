import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_pca(df_pca):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    targets = df_pca['target'].unique()
    colors = ['r', 'g', 'b']

    flower_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

    for target, color in zip(targets, colors):
        indices_to_keep = df_pca['target'] == target
        ax.scatter(df_pca.loc[indices_to_keep, 'PC1'],
                   df_pca.loc[indices_to_keep, 'PC2'],
                   df_pca.loc[indices_to_keep, 'PC3'],
                   c=color,
                   label=flower_names[target])

    ax.set_xlabel('Principal Component 1 (PC1)')
    ax.set_ylabel('Principal Component 2 (PC2)')
    ax.set_zlabel('Principal Component 3 (PC3)')
    ax.set_title('PCA of Iris Dataset in 3D')
    ax.legend()
    plt.show()
