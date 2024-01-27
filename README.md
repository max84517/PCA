# Introduction to Principal Component Analysis(PCA)
PCA is a famous supervised learning method in ML field, it can use to cluster and it's simple but useful in many ways. I will take a brief introduction to PCA from linear algebrea aspect then demonstrate PCA on iris dataset.

# PCA in Linear Algebra
In real world, the data usually looks like
|id|height|weight|$\cdots$|blood_pressure|
|-|-|-|-|-|
1 | $x_{1,1}$ | $x_{1,2}$ | $\cdots$ | $x_{1,n}$ |
2 | $x_{2,1}$ | $x_{2,2}$ | $\cdots$ | $x_{2,n}$ |
$\vdots$ | $\vdots$ | $\vdots$ | $\ddots$ | $\vdots$ | 
n | $x_{m,1}$ | $x_{m,2}$ | $\cdots$ | $x_{m,n}$

Normally, we will normalized the data in preprocessing process to make comparison easy and prevant a super large number suddenly appear in our model. So we can obtain a matrix

$$
B = 
\begin{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,n} \\
y_{2,1} & y_{2,2} & \cdots & y_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
y_{m,1} & y_{m,2} & \cdots & y_{m,n} \\
\end{bmatrix}
$$

Where 

$$
y_{i,j} = \frac{x_{i,j} - \overline{x_{j}}}{\sigma_{j}}
$$

The data usually have a problems -- **There are correlations between features, so some information may be overlapped**, for example, the higher blood pressure level often comes with the higher cholesterol level! keep this in mind, we would return to this later.


Since the information is too complicated and large sometimes, we come up with an idea, we want to find a weighted index that can extract all information from each observations, that is

$$
z_{k} = a_1y_{k,1} + a_2y_{k,2} + \cdots + a_ny_{k,n}, \quad\forall k \in [1, m]
$$

But how do we decide the relative weight $a$? **The answer is we don't need to decide it by our knowledge! The optimal relative weight can be generate by math!**

Since the informations we extract from observations must be simple to compare to each others, in other words, the difference must be large enough for us to tell the difference from observations. IN statistics aspects, we want to maximize our variance

$$
\max Var(z) = \max\frac{\sum(z_{k} - \overline{z})^2}{m-1} = \max\frac{\sum(a^Ty_{k} - 0)^2}{m -1} = \max \frac{a^TB^TBa}{m -1} = a^T\rho a
$$

Since we want the relative weights' length add to 1 (we can always do it by devided by its length)

$$
\sum a_i^2 = a^Ta = 1
$$

So, the problem becomes 

$$
\max a^TB^TBa 
$$

$$
s.t. \quad a^Ta = 1
$$

According to Rayleigh quotient, we can obtain the optimal relative weight is the units eighevectors of $\rho$

$$
a = e_{i}, \quad i \in [1, n]
$$

The variance will be 

$$
Var(z) = \lambda_{i} ,\quad\forall i \in [1, n]
$$

So, the greater the eigenvalues, the greater the varience. we can sort the eigenvalue in descending order

$$
\lambda_1 > \lambda_2 > \cdots > \lambda_n
$$

So the unit eigenvectors that corresponding to $\lambda_1$ is called the first PC! That is

$$
a = e_1
$$

Since $B^TB$ is a symmetric matrix, the different eigenvalues corrsponding eigenspace must be orthogonal 

$$
Cov(e_2^Ty, e_1^Ty) = e_2^TB^TBe_1 = e_2^T\lambda_1e_1 = \lambda_1e_2^Te_1 = 0
$$

So every principal components must be uncorrelated! In other words, each principal components must reflect different information!

# Applying PCA on Iris Dataset
To better understanding the PCA, we demonstrate a simple application by using a famous dataset - iris dataset

## Data preprocessing

`process_data()` will first import the data and attach the label to each observations (Note that PCA is still a unsupervised learning, this step is only to make the visualized result more clear) and normalized the data 

```python
def process_data():
    iris = load_iris()
    df = pd.DataFrame(np.concatenate((iris.data, np.array([iris.target]).T), axis=1),
                      columns=iris.feature_names + ['target'])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
```
Then, data should looks like

| |sepal length (cm)| sepal width (cm) | petal length (cm) | petal width (cm) | target |
|-|-|-|-|-|-|
|0|5.1	|3.5	|1.4	|0.2	|0.0|
|1|4.9	|3.0	|1.4	|0.2	|0.0|
|2|4.7	|3.2	|1.3	|0.2	|0.0|
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
|149|5.9	|3.0	|5.1	|1.8	|2.0|

## PCA 

Then, we simply perform PCA and since there are three different types of iris, we try to use three principal components to distinguish them. After that the `perform_pca()` return the results

```python
def perform_pca(X_scaled):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca
```
## Plotting 

At the end, we use `plot_3d_pca()` to plot the result in 3-d graph

```python
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
```

Attach the names and color of each type of iris, we get a beautiful graph that demonstrate how PCA distinguish different type of iris simply by delivering differnt weights to each features!

![alt text](https://github.com/max84517/PCA/blob/main/graph/iris_pca.png)
