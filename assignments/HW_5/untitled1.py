# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:15:38 2017

@author: JSZJZ
"""
#0.Species	2.FrontalLip 3.RearWidth	4.Length 5.Width 6.Depth	Male 7.Female

import numpy as np
import matplotlib.pyplot as plt
import math 
import time
start_time = time.time()

#initial parameters and load data
data = np.loadtxt("dataset.txt")
alldata = data[:,1:8]
training = data[0:140,1:8]
test = data[140:200,1:8]
validate_training = data[0:140,0]
validate_test = data[140:200,0]
validate_data = data[:,0]

count = 0
for i in range(140):
    if validate_training[i] == 1:
        count = count + 1
p_prior1 = count/140
p_prior0 = 1 - (count/140)
      
tr_1 = np.zeros([count,7])
tr_0 = np.zeros([140-count,7])

ind1 = 0
ind0 = 0
for i in range(140):
    if validate_training[i] == 1:
        tr_1[ind1] = training[i,:]
        ind1 = ind1 + 1
    if validate_training[i] == 0:
        tr_0[ind0] = training[i,:]
        ind0 = ind0 + 1
        
mean_1 = np.zeros(7)
mean_0 = np.zeros(7)

for i in range(7):
    mean_1[i] = sum(tr_1[:,i])/count
    M1 = mean_1[i]*np.ones([count,1])
    
    mean_0[i] = sum(tr_1[:,i])/(140-count)
    M0 = mean_0[i]*np.ones([(140-count),1])

sig_1 = 0
sig_0 = 0
for i in range(count):
    sig_1 = sig_1 + (tr_1[i,:]-np.asmatrix(mean_1)).T@(tr_1[i,:]-np.asmatrix(mean_1))/count
for i in range(140-count):
    sig_0 = sig_0 + (tr_0[i,:]-np.asmatrix(mean_0)).T@(tr_0[i,:]-np.asmatrix(mean_0))/(140-count)
#sig_1 = np.cov(tr_1.T)
#sig_0 = np.cov(tr_0.T)

count1 = 0
count0 = 0

for i in range(60):
    g1 = -0.5*((np.asmatrix(test[i,:]) - np.asmatrix(mean_1))@np.linalg.inv(sig_1)@(np.asmatrix(test[i,:]) - np.asmatrix(mean_1)).T) - 3.5*math.log1p(2*math.pi) - 0.5*math.log1p(np.linalg.det(sig_1)) + math.log1p(p_prior1)
    g0 = -0.5*((np.asmatrix(test[i,:]) - np.asmatrix(mean_0))@np.linalg.inv(sig_0)@(np.asmatrix(test[i,:]) - np.asmatrix(mean_0)).T) - 3.5*math.log1p(2*math.pi) - 0.5*math.log1p(np.linalg.det(sig_0)) + math.log1p(p_prior0)
    if g1 >= g0:
        if validate_test[i] == 1:
            count1 = count1 + 1
    else:
        if g0 > g1:
            if validate_test[i] == 0:
                count0 = count0 + 1

c = 0
for i in range(60):
    if validate_test[i] == 1:
        c = c + 1

error1 = c - count1
error0 = 60 - c - count0

Matrix1 = np.ones([2,2])
Matrix1[0,0] = count0
Matrix1[1,0] = error0
Matrix1[0,1] = count1
Matrix1[1,1] = error1


count1 = 0
count0 = 0

for i in range(140):
    g1 = -0.5*((np.asmatrix(training[i,:]) - np.asmatrix(mean_1))@np.linalg.inv(sig_1)@(np.asmatrix(training[i,:]) - np.asmatrix(mean_1)).T) - 3.5*math.log1p(2*math.pi) - 0.5*math.log1p(np.linalg.det(sig_1)) + math.log1p(p_prior1)
    g0 = -0.5*((np.asmatrix(training[i,:]) - np.asmatrix(mean_0))@np.linalg.inv(sig_0)@(np.asmatrix(training[i,:]) - np.asmatrix(mean_0)).T) - 3.5*math.log1p(2*math.pi) - 0.5*math.log1p(np.linalg.det(sig_0)) + math.log1p(p_prior0)
    if g1 >= g0:
        if validate_training[i] == 1:
            count1 = count1 + 1
    else:
        if g0 > g1:
            if validate_training[i] == 0:
                count0 = count0 + 1

c = 0
for i in range(140):
    if validate_training[i] == 1:
        c = c + 1

error1 = c - count1
error0 = 140 - c - count0

Matrix2 = np.ones([2,2])
Matrix2[0,0] = count0
Matrix2[1,0] = error0
Matrix2[0,1] = count1
Matrix2[1,1] = error1

count1 = 0
count0 = 0

for i in range(200):
    g1 = -0.5*((np.asmatrix(alldata[i,:]) - np.asmatrix(mean_1))@np.linalg.inv(sig_1)@(np.asmatrix(alldata[i,:]) - np.asmatrix(mean_1)).T) - 3.5*math.log1p(2*math.pi) - 0.5*math.log1p(np.linalg.det(sig_1)) + math.log1p(p_prior1)
    g0 = -0.5*((np.asmatrix(alldata[i,:]) - np.asmatrix(mean_0))@np.linalg.inv(sig_0)@(np.asmatrix(alldata[i,:]) - np.asmatrix(mean_0)).T) - 3.5*math.log1p(2*math.pi) - 0.5*math.log1p(np.linalg.det(sig_0)) + math.log1p(p_prior0)
    if g1 >= g0:
        if validate_data[i] == 1:
            count1 = count1 + 1
    else:
        if g0 > g1:
            if validate_data[i] == 0:
                count0 = count0 + 1

c = 0
for i in range(200):
    if validate_data[i] == 1:
        c = c + 1

error1 = c - count1
error0 = 200 - c - count0

Matrix3 = np.ones([2,2])
Matrix3[0,0] = count0
Matrix3[1,0] = error0
Matrix3[0,1] = count1
Matrix3[1,1] = error1

print(Matrix1)#test
print(Matrix2)#training
print(Matrix3)#all
print("--- %s seconds ---" % (time.time() - start_time))#processing time