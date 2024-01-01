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
\begain{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,n}\\\
y_{2,1} & y_{2,2} & \cdots & y_{2,n}\\\
\vdots & \vdots & \ddots & \vdots
y_{m,1} & y_{m,2} & \cdots & y_{m,n}\\\
\end{bmatrix}
$$
The data usually have a problems -- **There are correlations between features, so some information may be overlapped**, for example, the higher blood pressure level often comes with the higher cholesterol level! keep this in mind, we would return to this later.





Since the information is too complicated and large sometimes, we come up with an idea, we want to find a weighted index that can extract all information from each observations, that is
$$

$$
