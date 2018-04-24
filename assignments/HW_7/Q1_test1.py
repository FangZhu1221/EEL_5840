# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:07:29 2017

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


#find the best FFS
#find the best in 1st dimension
print("FFS: ")
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
    Sb = (num1/(num1+num2))*(mean1-mean0)@(mean1-mean0).T + (num2/(num1+num2))*(mean2-mean0)@(mean2-mean0).T
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
    
#find a best in 3rd dimensions
count3 = 0
index3 = 0
for i in range(13):
    if (i != index1 and i != index2):
        x1 = np.vstack((data1[index1,:],data1[index2,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[index2,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count3 < r:
            count3 = r
            index3 = i
count[2] = count3
index[2] = index3           
print(index3) 
print(count3)

    
#find a best in 4th dimensions
count4 = 0
index4 = 0
for i in range(13):
    if (i != index1 and i != index2 and i != index3):
        x1 = np.vstack((data1[index1,:],data1[index2,:],data1[index3,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[index2,:],data2[index3,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count4 < r:
            count4 = r
            index4 = i
count[3] = count4
index[3] = index4           
print(index4) 
print(count4)

    
#find a best in 5th dimensions
count5 = 0
index5 = 0
for i in range(13):
    if (i != index1 and i != index2 and i != index3 and i != index4):
        x1 = np.vstack((data1[index1,:],data1[index2,:],data1[index3,:],data1[index4,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[index2,:],data2[index3,:],data2[index4,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count5 < r:
            count5 = r
            index5 = i
count[4] = count5
index[4] = index5           
print(index5) 
print(count5)

#find a best in 6th dimensions
count6 = 0
index6 = 0
for i in range(13):
    if (i != index1 and i != index2 and i != index3 and i != index4 and i != index5):
        x1 = np.vstack((data1[index1,:],data1[index2,:],data1[index3,:],data1[index4,:],data1[index5,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[index2,:],data2[index3,:],data2[index4,:],data2[index5,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count6 < r:
            count6 = r
            index6 = i
count[5] = count6
index[5] = index6           
print(index6) 
print(count6)

#find a best in 7th dimensions
count7 = 0
index7 = 0
for i in range(13):
    if (i != index1 and i != index2 and i != index3 and i != index4 and i != index5 and i != index6):
        x1 = np.vstack((data1[index1,:],data1[index2,:],data1[index3,:],data1[index4,:],data1[index5,:],data1[index6,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[index2,:],data2[index3,:],data2[index4,:],data2[index5,:],data2[index6,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count7 < r:
            count7 = r
            index7 = i
count[6] = count7
index[6] = index7           
print(index7) 
print(count7)

#find a best in 8th dimensions
count8 = 0
index8 = 0
for i in range(13):
    if (i != index1 and i != index2 and i != index3 and i != index4 and i != index5 and i != index6 and i != index7):
        x1 = np.vstack((data1[index1,:],data1[index2,:],data1[index3,:],data1[index4,:],data1[index5,:],data1[index6,:],data1[index7,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[index2,:],data2[index3,:],data2[index4,:],data2[index5,:],data2[index6,:],data2[index7,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count8 < r:
            count8 = r
            index8 = i
count[7] = count8
index[7] = index8           
print(index8) 
print(count8)

#find a best in 9th dimensions
count9 = 0
index9 = 0
for i in range(13):
    if (i != index1 and i != index2 and i != index3 and i != index4 and i != index5 and i != index6 and i != index7 and i != index8 ):
        x1 = np.vstack((data1[index1,:],data1[index2,:],data1[index3,:],data1[index4,:],data1[index5,:],data1[index6,:],data1[index7,:],data1[index8,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[index2,:],data2[index3,:],data2[index4,:],data2[index5,:],data2[index6,:],data2[index7,:],data2[index8,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count9 < r:
            count9 = r
            index9 = i
count[8] = count9
index[8] = index9           
print(index9) 
print(count9)

#find a best in 10th dimensions
count10 = 0
index10 = 0
for i in range(13):
    if (i != index1 and i != index2 and i != index3 and i != index4 and i != index5 and i != index6 and i != index7 and i != index8 and i != index9 ):
        x1 = np.vstack((data1[index1,:],data1[index2,:],data1[index3,:],data1[index4,:],data1[index5,:],data1[index6,:],data1[index7,:],data1[index8,:],data1[index9,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[index2,:],data2[index3,:],data2[index4,:],data2[index5,:],data2[index6,:],data2[index7,:],data2[index8,:],data2[index9,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count10 < r:
            count10 = r
            index10 = i
count[9] = count10
index[9] = index10           
print(index10) 
print(count10)

#find a best in 11th dimensions
count11 = 0
index11 = 0
for i in range(13):
    if (i != index1 and i != index2 and i != index3 and i != index4 and i != index5 and i != index6 and i != index7 and i != index8 and i != index9  and i != index10 ):
        x1 = np.vstack((data1[index1,:],data1[index2,:],data1[index3,:],data1[index4,:],data1[index5,:],data1[index6,:],data1[index7,:],data1[index8,:],data1[index9,:],data1[index10,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[index2,:],data2[index3,:],data2[index4,:],data2[index5,:],data2[index6,:],data2[index7,:],data2[index8,:],data2[index9,:],data2[index10,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count11 < r:
            count11 = r
            index11 = i
count[10] = count11
index[10] = index11           
print(index11) 
print(count11)

#find a best in 12th dimensions
count12 = 0
index12 = 0
for i in range(13):
    if (i != index1 and i != index2 and i != index3 and i != index4 and i != index5 and i != index6 and i != index7 and i != index8  and i != index9  and i != index10  and i != index11):
        x1 = np.vstack((data1[index1,:],data1[index2,:],data1[index3,:],data1[index4,:],data1[index5,:],data1[index6,:],data1[index7,:],data1[index8,:],data1[index9,:],data1[index10,:],data1[index11,:],data1[i,:]))
        x2 = np.vstack((data2[index1,:],data2[index2,:],data2[index3,:],data2[index4,:],data2[index5,:],data2[index6,:],data2[index7,:],data2[index8,:],data2[index9,:],data2[index10,:],data2[index11,:],data2[i,:]))
        r = fisher_ratio(x1,x2)
        if count12 < r:
            count12 = r
            index12 = i
count[11] = count12
index[11] = index12           
print(index12) 
print(count12)

#find the best in 13th dimension
print("BFS: ")
count13 = 0
index13 = 0
x1 = np.vstack((data1[0:(i),:],data1[(i+1):13,:]))
x2 = np.vstack((data2[0:(i),:],data2[(i+1):13,:]))
r = fisher_ratio(x1,x2)
count13 = r
count[12] = count13
index[12] = index13
print(index13)
print(count13)

t = np.arange(13)
plt.plot(t,count)