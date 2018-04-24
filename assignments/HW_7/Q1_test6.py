# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:13:07 2017

@author: JSZJZ
"""


import numpy as np
import matplotlib.pyplot as plt
import math 
import time

#initial parameters and load data
data1 = np.matrix(np.loadtxt("WhiteWine_HW7.txt")).T
data2 = np.matrix(np.loadtxt("RedWine_HW7.txt")).T
rate = np.zeros(13)
d1 = data1.shape[1]
d2 = data2.shape[1]
d = d1 + d2
data = np.hstack((data1,data2))
vali = np.hstack((np.zeros(d1),np.ones(d2)))
al = np.vstack((data[6,:],data[7,:],data[3,:],data[1,:],data[10,:],data[0,:],data[5,:],data[9,:],data[4,:],data[8,:],data[2,:],data[11,:],data[12,:]))
for j in range(13):
    if j == 0:
        alldata = al
    else:
        alldata = np.delete(al,13-j,0)
    ald1 = alldata[:,0:d1]
    ald2 = alldata[:,d1:d]
    p_prior1 = d1/d
    p_prior2 = d2/d
      
    mean1 = ald1.mean(1)
    mean2 = ald2.mean(1)

    sig1 = 0
    sig2 = 0

    for i in range(d1):
        sig1 = sig1 + (1/d1)*(ald1[:,i]-mean1)@(ald1[:,i]-mean1).T

    for i in range(d2):
        sig2 = sig2 + (1/d2)*(ald2[:,i]-mean2)@(ald2[:,i]-mean2).T

    count1 = 0
    count2 = 0

    for i in range(d):
        g1 = -0.5*((np.asmatrix(alldata[:,i]) - np.asmatrix(mean1)).T@np.linalg.inv(sig1)@(np.asmatrix(alldata[:,i]) - np.asmatrix(mean1))) - 3.5*math.log1p(2*math.pi) - 0.5*math.log1p(np.linalg.det(sig1)) + math.log1p(p_prior1)
        g0 = -0.5*((np.asmatrix(alldata[:,i]) - np.asmatrix(mean2)).T@np.linalg.inv(sig2)@(np.asmatrix(alldata[:,i]) - np.asmatrix(mean2))) - 3.5*math.log1p(2*math.pi) - 0.5*math.log1p(np.linalg.det(sig2)) + math.log1p(p_prior2)
        if g1 >= g0:
            if vali[i] == 0:
                count1 = count1 + 1
        else:
            if g0 > g1:
                if vali[i] == 1:
                    count2 = count2 + 1

    error1 = d1 - count1
    error2 = d2 - count2

    Matrix1 = np.ones([2,2])
    Matrix1[0,0] = count1
    Matrix1[1,0] = error1
    Matrix1[0,1] = error2
    Matrix1[1,1] = count2

    print(Matrix1)
    print(1-(error1+error2)/(count1+count2))
    rate[12-j] = 1-(error1+error2)/(count1+count2)

t = np.arange(13)
plt.plot(t,rate)