# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:11:42 2017

@author: JSZJZ
"""


import numpy as np
import matplotlib.pyplot as plt
import math 
import time
import random

def fisher_ratio(x1,x2): #dimension x number
    mean1 = x1.mean(1)
    mean2 = x2.mean(1)
    num1 = x1.shape[1]
    num2 = x2.shape[1]
    mean0 = (num1*mean1+num2*mean2)/(num1+num2)
    Sb = np.trace((num1/(num1+num2))*(mean1-mean0)@(mean1-mean0).T + (num2/(num1+num2))*(mean2-mean0)@(mean2-mean0).T)
    Sw = np.trace(np.cov(x1)) + np.trace(np.cov(x2))
    r = np.asscalar(Sb/Sw)
    return r

#generate data
data1 = np.matrix(np.loadtxt("WhiteWine_HW7.txt")).T
data2 = np.matrix(np.loadtxt("RedWine_HW7.txt")).T
d1 = data1.shape[1]
d2 = data2.shape[1]
count = np.zeros(13)
index = np.zeros(13)


#find the best FFS
#find the best in 1st dimension
count1 = 0
index1 = 0
for i in range(13):
    x1 = data1[i,:]
    x2 = data2[i,:]
    mean1 = x1.mean(1)
    mean2 = x2.mean(1)
    num1 = x1.shape[1]
    num2 = x2.shape[1]
    mean0 = (num1*mean1+num2*mean2)/(num1+num2)
    Sb = np.trace((num1/(num1+num2))*(mean1-mean0)@(mean1-mean0).T + (num2/(num1+num2))*(mean2-mean0)@(mean2-mean0).T)
    Sw = np.cov(x1) + np.cov(x2)
    r = np.asscalar(Sb/Sw)
    if count1 < r: 
        count1 = r
        index1 = i
count[0] = count1
index[0] = index1
print(index1)
print(count1)

#find a best in 2nd dimensions
count2 = 0
index2 = 0
for i in range(13):
    if i != index1:
        x1 = np.vstack((data1[index1,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count2 < r:
            count2 = r
            index2 = i
count[1] = count2
index[1] = index2           
print(index2)   
print(count2) 
