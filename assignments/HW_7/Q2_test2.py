# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:04:11 2017

@author: JSZJZ
"""


import numpy as np
import matplotlib.pyplot as plt
import math 
import copy
import time
import random
from scipy.stats import multivariate_normal


data = np.loadtxt("GMDataSet_HW7.txt") #1000 x 5

k = 3
length = data.shape[0]
order = data.shape[1]

U = np.zeros([k,5])
U_old = np.zeros([k,5])

P = np.zeros(k)
P_1 = np.zeros(k)
P_old = np.zeros(k)

Sig = np.zeros([5,5])
S = np.zeros([5,5*k])
Sig = np.cov(data.T)
si = np.zeros(k)
S_old = np.zeros([5,5*k])

Cu = np.zeros([length,k])
Cd = np.zeros(length)
C = np.zeros([length,k])

Sig = np.cov(data.T)
for i in range(k):
    P[i] = 1/k
    U[i,:] = data[round(np.random.randint(0,1000)),:]
    S[:,(i*5):((i+1)*5)] = 1*np.eye(U[i,:].size)


diff = 1 # the threshold
o = 0
while diff > 0.0001:
    
    if o > 1000:
        break
    
    U_old = copy.copy(U)
    S_old = copy.copy(S)
    P_old = copy.copy(P)
    
    # E-step   
    for i in range(length):
        for j in range(k):
            mvn = multivariate_normal(U[j,:],S[:,j*5:(j+1)*5])
            Cu[i,j] = P[j]*mvn.pdf(data[i,:])
            #Cu[i,j] = P[j]*(((2*math.pi)**(-5/2))*(np.linalg.det(S[:,j*5:(j+1)*5])**(-0.5))*math.exp(-0.5*np.matrix((data[i,:]-U[j,:]))@np.linalg.inv(S[:,j*5:(j+1)*5])@np.matrix((data[i,:]-U[j,:])).T))
        Cd[i] = sum(Cu[i,:])
    for i in range(length):
        for j in range(k):
            C[i,j] = Cu[i,j]/Cd[i]
      
    # M-step
    for i in range(k):
        P[i] = sum(C[:,i])/length
    
    for i in range(k):
        U[i,:] = sum(np.matrix(C[:,i])*data)/sum(C[:,i])
    o = o + 1
    
    for i in range(k): 
        S1 = 0
        S2 = 0
        for j in range(length):
            S1 = S1 + C[j,i]*(np.matrix((data[j,:]-U[i,:]))@np.matrix((data[j,:]-U[i,:])).T)
            S2 = S2 + 5*C[j,i]
        S[:,i*5:(i+1)*5] = np.asscalar(S1/S2)*np.identity(5)
    
     
    #converge
    diff = sum(sum(abs(U-U_old))) + sum(sum(abs(S-S_old))) + sum(abs(P-P_old))
    #print(h) 
data1 = np.zeros([1000,5])
data2 = np.zeros([1000,5])
data3 = np.zeros([1000,5])
add = np.zeros([1,5])
i1 = 0
i2 = 0
i3 = 0
for i in range(length):
    b = 0
    for j in range(k):
        if C[i,j] > C[i,b]:
            b = j
    if b == 0:
        print(0)
        data1[i1,:] = data[i,:]
        i1 = i1 + 1
    else:
        if b == 1:
            print(1)
            data2[i1,:] = data[i,:]
            print(data2.shape)
            i2 = i2 + 1
        else:
            if b == 2:
                print(2)
                data3[i3,:] = data[i,:]
                i3 = i3 + 1
print(i1)
print(i2)
print(i3)
a1 = data1[0:i1,:]
a2 = data2[0:i2,:]
a3 = data2[0:i3,:]    
fig = plt.figure(figsize = (10,20))    
# plot data by PCA for data
x1 = a1[:,0]
x2 = a1[:,1]
x3 = a1[:,2]
x4 = a1[:,3]
x5 = a1[:,4]

E1 = sum(x1)/x1.size
E2 = sum(x2)/x2.size
E3 = sum(x3)/x3.size
E4 = sum(x4)/x4.size
E5 = sum(x5)/x5.size

Mx1 = np.array([(x1[m]-E1) for m in range(i1)])
Mx2 = np.array([(x2[m]-E2) for m in range(i1)])
Mx3 = np.array([(x3[m]-E3) for m in range(i1)])
Mx4 = np.array([(x4[m]-E4) for m in range(i1)])
Mx5 = np.array([(x5[m]-E5) for m in range(i1)])

M1 = np.matrix([Mx1,Mx2,Mx3,Mx4,Mx5])
Cov1 = M1@M1.T/i1
eigen_vals_1, eigen_vecs_1 = np.linalg.eig(Cov1)

eigen_pairs_1 = [(np.abs(eigen_vals_1[i]), np.array(eigen_vecs_1[:,i].T)[0]) for i in range(len(eigen_vals_1))]
eigen_pairs_1.sort(key = lambda x : x[0],reverse=True)
#print(eigen_pairs_1)
w1 = np.hstack((eigen_pairs_1[0][1][:, np.newaxis], eigen_pairs_1[1][1][:, np.newaxis]))
M1_pca = M1.T@w1

x1 = a2[:,0]
x2 = a2[:,1]
x3 = a2[:,2]
x4 = a2[:,3]
x5 = a2[:,4]

E1 = sum(x1)/x1.size
E2 = sum(x2)/x2.size
E3 = sum(x3)/x3.size
E4 = sum(x4)/x4.size
E5 = sum(x5)/x5.size

Mx1 = np.array([(x1[m]-E1) for m in range(i2)])
Mx2 = np.array([(x2[m]-E2) for m in range(i2)])
Mx3 = np.array([(x3[m]-E3) for m in range(i2)])
Mx4 = np.array([(x4[m]-E4) for m in range(i2)])
Mx5 = np.array([(x5[m]-E5) for m in range(i2)])

M1 = np.matrix([Mx1,Mx2,Mx3,Mx4,Mx5])
Cov1 = M1@M1.T/i2
eigen_vals_1, eigen_vecs_1 = np.linalg.eig(Cov1)

eigen_pairs_1 = [(np.abs(eigen_vals_1[i]), np.array(eigen_vecs_1[:,i].T)[0]) for i in range(len(eigen_vals_1))]
eigen_pairs_1.sort(key = lambda x : x[0],reverse=True)
#print(eigen_pairs_1)
w1 = np.hstack((eigen_pairs_1[0][1][:, np.newaxis], eigen_pairs_1[1][1][:, np.newaxis]))
M2_pca = M1.T@w1

x1 = a3[:,0]
x2 = a3[:,1]
x3 = a3[:,2]
x4 = a3[:,3]
x5 = a3[:,4]

E1 = sum(x1)/x1.size
E2 = sum(x2)/x2.size
E3 = sum(x3)/x3.size
E4 = sum(x4)/x4.size
E5 = sum(x5)/x5.size

Mx1 = np.array([(x1[m]-E1) for m in range(i3)])
Mx2 = np.array([(x2[m]-E2) for m in range(i3)])
Mx3 = np.array([(x3[m]-E3) for m in range(i3)])
Mx4 = np.array([(x4[m]-E4) for m in range(i3)])
Mx5 = np.array([(x5[m]-E5) for m in range(i3)])

M1 = np.matrix([Mx1,Mx2,Mx3,Mx4,Mx5])
Cov1 = M1@M1.T/i3
eigen_vals_1, eigen_vecs_1 = np.linalg.eig(Cov1)

eigen_pairs_1 = [(np.abs(eigen_vals_1[i]), np.array(eigen_vecs_1[:,i].T)[0]) for i in range(len(eigen_vals_1))]
eigen_pairs_1.sort(key = lambda x : x[0],reverse=True)
#print(eigen_pairs_1)
w1 = np.hstack((eigen_pairs_1[0][1][:, np.newaxis], eigen_pairs_1[1][1][:, np.newaxis]))
M3_pca = M1.T@w1
p1 = fig.add_subplot(*[2,1,1])
p1.scatter(np.array(M1_pca[:,0].T)[0],np.array(M1_pca[:,1].T)[0])
p1.scatter(np.array(M2_pca[:,0].T)[0],np.array(M2_pca[:,1].T)[0])
p1.scatter(np.array(M3_pca[:,0].T)[0],np.array(M3_pca[:,1].T)[0])