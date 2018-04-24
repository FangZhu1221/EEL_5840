# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 22:35:45 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import copy
import time
from scipy.stats import multivariate_normal

def fisher_ratio(mean1,mean2,S1,S2,P1,P2): #dimension x number
    mean0 = (P1*mean1+P2*mean2)/(P1+P2)
    Sb = (P1/(P1 + P2))*np.matrix((mean1-mean0)).T@np.matrix((mean1-mean0)) + (P1/(P1 + P2))*np.matrix((mean2-mean0)).T@np.matrix((mean2-mean0))
    Sw = S1 + S2
    r = np.trace(Sb@np.linalg.inv(Sw))
    return r

data = np.loadtxt("GMDataSet_HW7.txt") #1000 x 5
 
for m in range(3):
    k = m + 2
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
    if k == 4:
        print(k)
        print(fisher_ratio(U[0,:],U[1,:],S[:,0:5],S[:,5:10],P[0],P[1]))
        print(fisher_ratio(U[0,:],U[2,:],S[:,0:5],S[:,10:15],P[0],P[2]))
        print(fisher_ratio(U[0,:],U[3,:],S[:,0:5],S[:,15:20],P[0],P[3]))
        print(fisher_ratio(U[1,:],U[2,:],S[:,5:10],S[:,10:15],P[1],P[2]))
        print(fisher_ratio(U[1,:],U[3,:],S[:,5:10],S[:,15:20],P[1],P[3]))
        print(fisher_ratio(U[2,:],U[3,:],S[:,10:15],S[:,15:20],P[2],P[3]))
    else:
        if k == 3:
            print(k)
            print(fisher_ratio(U[0,:],U[1,:],S[:,0:5],S[:,5:10],P[0],P[1]))
            print(fisher_ratio(U[0,:],U[2,:],S[:,0:5],S[:,10:15],P[0],P[2]))
            print(fisher_ratio(U[1,:],U[2,:],S[:,5:10],S[:,10:15],P[1],P[2]))
        else:
            if k == 2:
                print(k)
                print(fisher_ratio(U[0,:],U[1,:],S[:,0:5],S[:,5:10],P[0],P[1]))

