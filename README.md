# Introduction to Principal Component Analysis(PCA)
PCA is a famous supervised learning method in ML field, it can use to classification and it's simple but useful in many ways. I will take a brief introduction to PCA from linear algebrea aspect.

# PCA in Linear Algebra
In real world, the data usually looks like
|id|height|weight|$\cdots$|blood_pressure|
|-|-|-|-|-|
1 | 174 | 63 | $\cdots$ | 69 |
2 | 170 | 72 | $\cdots$ | 90 |
$\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | 
n | 183 | 76 | $\cdots$ | 66

The data usually have a problems -- **There are correlations between features, so some information may be overlapped**, for example, the higher blood pressure level often comes with the higher cholesterol level!
 
