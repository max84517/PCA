# Introduction to Principal Component Analysis(PCA)
PCA is a famous supervised learning method in ML field, it can use to classification and it's simple but useful in many ways. I will take a brief introduction to PCA from linear algebrea aspect.

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

Since we wnat the relative weights' length add to 1 (we can always do it by devided by its length)

$$
\sum a_i^2 = a^Ta = 1
$$

So, the problem becomes 

$$
\max a^TB^TBa 
$$

$$
s.t. a^Ta = 1
$$

According to Rayleigh quotient, we can obtain the optimal relative weight is the units eighevectors of $\rho$

$$
a = e_{i} \quad i \in [1, n]
$$

The variance will be 

$$
Var(z) = \lambda_{i} \forall i \in [1, n]
$$

So, the greater the eigenvalues, the greater the varience. we can sort the eigenvalue in descending order

$$
\lambda_1 > \lambda_2 > \cdots > \lambda_n
$$

So the unit eigenvectors that corresponding to $\lambda_1$ is called the first PC! That is

$$
a = e_1
$$
