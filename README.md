# Introduction to Principal Component Analysis(PCA)
PCA is a famous supervised learning method in ML field, it can use to classification and it's simple but useful in many ways. I will take a brief introduction to PCA from linear algebrea aspect.

# PCA in Linear Algebra
In real world, the data usually looks like
|id|height|weight|$\cdots$|blood_pressure|
|-|-|-|-|-|
1 | $x_{1,1}$ | $x_{1,2}$ | $\cdots$ | $x_{1,n}$ |
2 | $x_{2,1}$ | $x_{2,2}$ | $\cdots$ | $x_{2,n}$ |
$\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | 
n | $x_{m,1}$ | 76 | $\cdots$ | $x_{m,n}$

The data usually have a problems -- **There are correlations between features, so some information may be overlapped**, for example, the higher blood pressure level often comes with the higher cholesterol level!
 
So, we come up with an idea, we want to find a weighted index that are 
