# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:27:31 2017

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
#find the best in 12th dimension
count1 = 0
index1 = 0
for i in range(13):
    x1 = np.vstack((data1[0:(i),:],data1[(i+1):13,:]))
    x2 = np.vstack((data2[0:(i),:],data2[(i+1):13,:]))
    r = fisher_ratio(x1,x2)
    if count1 < r: 
        count1 = r
        index1 = i
count[0] = count1
index[0] = index1
print(index1)
print(count1)

#find the best in 11th dimension
count2 = 0
index2 = 0
for i in range(13):
    if i != index1:
        if i < index1:
            x1 = np.vstack((data1[0:(i),:],data1[i+1:index1,:],data1[index1+1:13,:]))
            x2 = np.vstack((data2[0:(i),:],data2[i+1:index1,:],data2[index1+1:13,:]))
        else:
            if i > index1:
                x1 = np.vstack((data1[0:index1,:],data1[index1:i,:],data1[index1+1:13,:]))
                x2 = np.vstack((data2[0:index1,:],data2[index1:i,:],data2[index1+1:13,:]))
        r = fisher_ratio(x1,x2)
        if count2 < r: 
            count2 = r
            index2 = i
        count[1] = count2
        index[1] = index2
print(index2)
print(count2)

#find the best in 10th dimension
count3 = 0
index3 = 0
for i in range(13):
    if i != index1 and i != index2:
        if i < index2 and index2 < index1:
            x1 = np.vstack((data1[0:(i),:],data1[i+1:index2,:],data1[index2+1:index1,:],data1[index1+1:13,:]))
            x2 = np.vstack((data2[0:(i),:],data2[i+1:index2,:],data2[index2+1:index1,:],data2[index1+1:13,:]))
        else:
            if i < index1 and index1 < index2:
                x1 = np.vstack((data1[0:(i),:],data1[i+1:index1,:],data1[index1+1:index2,:],data1[index2+1:13,:]))
                x2 = np.vstack((data2[0:(i),:],data2[i+1:index2,:],data2[index1+1:index2,:],data2[index2+1:13,:]))
            else:
                if index2 < i and i < index1:
                    x1 = np.vstack((data1[0:index2,:],data1[index2+1:i,:],data1[i+1:index1,:],data1[index1+1:13,:]))
                    x2 = np.vstack((data2[0:index2,:],data2[index2+1:i,:],data2[i+1:index1,:],data2[index1+1:13,:]))
                else:
                    if index2 < index1 and index1 <i:
                        x1 = np.vstack((data1[0:index2,:],data1[index2+1:index1,:],data1[index1+1:i,:],data1[i+1:13,:]))
                        x2 = np.vstack((data2[0:index2,:],data2[index2+1:index1,:],data2[index1+1:i,:],data2[i+1:13,:]))
                    else:
                        if index1 < index2 and index2 < i:
                            x1 = np.vstack((data1[0:index1,:],data1[index1+1:index2,:],data1[index2+1:i,:],data1[i+1:13,:]))
                            x2 = np.vstack((data2[0:index1,:],data2[index1+1:index2,:],data2[index2+1:i,:],data2[i+1:13,:]))
                        else:
                            if index1 < i and i < index2:
                                x1 = np.vstack((data1[0:index1,:],data1[index1+1:i,:],data1[i+1:index2,:],data1[index2+1:13,:]))
                                x2 = np.vstack((data2[0:index1,:],data2[index1+1:i,:],data2[i+1:index2,:],data2[index2+1:13,:]))
        r = fisher_ratio(x1,x2)
        if count3 < r: 
            count3 = r
            index3 = i
        count[2] = count3
        index[2] = index3
print(index3)
print(count3)

