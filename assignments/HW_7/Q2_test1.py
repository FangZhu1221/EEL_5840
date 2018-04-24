# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:39:17 2017

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
    
    for i in range(k): 
        S1 = 0
        S2 = 0
        for j in range(length):
            S1 = S1 + C[j,i]*(np.matrix((data[j,:]-U[i,:]))@np.matrix((data[j,:]-U[i,:])).T)
            S2 = S2 + 5*C[j,i]
        S[:,i*5:(i+1)*5] = np.asscalar(S1/S2)*np.identity(5)
        

    o = o + 1
     
    #converge
    diff = sum(sum(abs(U-U_old))) + sum(sum(abs(S-S_old))) + sum(abs(P-P_old))
    #print(h)  
    
print("the mean:")
print(U[0,:])
print(U[1,:])
print(U[2,:])
print("the weight:")
print(P[0])
print(P[1])
print(P[2])
print("the sigma:")
print(S[:,0:5])
print(S[:,5:10])
print(S[:,10:15])
print("iterating number:") 
print(o) 



