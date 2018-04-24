# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:35:03 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 

data = np.loadtxt("GMDataSet_HW7.txt") #1000 x 5
length = data.shape[0]
order = data.shape[1]

fig = plt.figure(figsize = (10,20))   
# plot data by PCA for data
x1 = data[:,0]
x2 = data[:,1]
x3 = data[:,2]
x4 = data[:,3]
x5 = data[:,4]

E1 = sum(x1)/x1.size
E2 = sum(x2)/x2.size
E3 = sum(x3)/x3.size
E4 = sum(x4)/x4.size
E5 = sum(x5)/x5.size

Mx1 = np.array([(x1[m]-E1) for m in range(length)])
Mx2 = np.array([(x2[m]-E2) for m in range(length)])
Mx3 = np.array([(x3[m]-E3) for m in range(length)])
Mx4 = np.array([(x4[m]-E4) for m in range(length)])
Mx5 = np.array([(x5[m]-E5) for m in range(length)])

M1 = np.matrix([Mx1,Mx2,Mx3,Mx4,Mx5])
Cov1 = M1@M1.T/1000
eigen_vals_1, eigen_vecs_1 = np.linalg.eig(Cov1)

eigen_pairs_1 = [(np.abs(eigen_vals_1[i]), np.array(eigen_vecs_1[:,i].T)[0]) for i in range(len(eigen_vals_1))]
eigen_pairs_1.sort(key = lambda x : x[0],reverse=True)
print(eigen_pairs_1[0][1])
#print(eigen_pairs_1)
w1 = np.hstack((eigen_pairs_1[0][1][:, np.newaxis], eigen_pairs_1[1][1][:, np.newaxis]))
M1_pca = M1.T@w1
p1 = fig.add_subplot(*[2,1,1])
p1.scatter(np.array(M1_pca[:,0].T)[0],np.array(M1_pca[:,1].T)[0])