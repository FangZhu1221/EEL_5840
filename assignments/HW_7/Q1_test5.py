# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:04:36 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import time
import random

count = np.zeros(13)
index = np.zeros(13)

def fisher_ratio(x1,x2): #dimension x number
    mean1 = x1.mean(1)
    mean2 = x2.mean(1)
    num1 = x1.shape[1]
    num2 = x2.shape[1]
    mean0 = (num1*mean1+num2*mean2)/(num1+num2)
    Sb = (num1/(num1+num2))*(mean1-mean0)@(mean1-mean0).T + (num2/(num1+num2))*(mean2-mean0)@(mean2-mean0).T
    Sw = np.cov(x1) + np.cov(x2)
    r = np.trace(Sb@np.linalg.inv(Sw))
    return r

#generate data
data1 = np.matrix(np.loadtxt("WhiteWine_HW7.txt")).T
data2 = np.matrix(np.loadtxt("RedWine_HW7.txt")).T
d1 = data1.shape[1]
d2 = data2.shape[1]
count = np.zeros(13)
index = np.zeros(13)

#find the best BFS
#find the best in 13th dimension
print("BFS: ")
count13 = 0
index13 = 0
x1 = data1
x2 = data2
r = fisher_ratio(x1,x2)
count13 = r
count[0] = count13
index[0] = index13



xx1 = data1
xx2 = data2
k = 13
#find the best in 12th dimension
for j in range(13):
    count1 = 0
    index1 = 0
    if j == 11:
        break
    for i in range(13-j):
        x1 = np.delete(xx1,i,0)
        x2 = np.delete(xx2,i,0)
        r = fisher_ratio(x1,x2)
        if count1 < r: 
            count1 = r
            index1 = i
    xx1 = np.delete(xx1,index1,0)
    xx2 = np.delete(xx2,index1,0)
    count[j+1] = count1
    #index[j] = index1
t = np.arange(13)
plt.plot(t,count)