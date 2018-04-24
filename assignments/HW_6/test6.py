# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:26:08 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import time
import random

def der_sig(x):
    t = (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))
    return t

w1 = np.random.uniform(size=(2, 3))
w2 = np.random.uniform(size=(1, 3))
x1 = np.zeros([1,3])
x2 = np.zeros([1,3])

step1 = 0.03
step2 = 0.03
test_data = np.zeros(8)
test_data[0] = 0.32
test_data[1] = 0.92
test_data[2] = 0.78
test_data[3] = -1.82
test_data[4] = -0.76
test_data[5] = 1.54
test_data[6] = -0.77
test_data[7] = -1.29
#sigmoid = 1/(1+np.exp())

data = np.loadtxt("HW6_Data.txt")
validate = data[:,2]
training = data[:,0:2]
length = validate.size
fig = plt.figure(figsize=(10,10))


for j in range(1000):
    for i in range(length):
        x1[0,0] = training[i,0]
        x1[0,1] = training[i,1]
        x1[0,2] = 1
        t1 = w1@x1.T #2x1
        x2[0,0] = 1/(1+np.exp(-t1[0,0]))
        x2[0,1] = 1/(1+np.exp(-t1[1,0]))
        x2[0,2] = 1
        t2 = w2@x2.T
        y = 1/(1+np.exp(-t2))
        e = validate[i]-y
        w2 = w2 + step2*e*np.asscalar(der_sig(t2))*x2
        w1 = w1 + ((step1*np.asscalar(der_sig(t2))*e)*x1.T@np.multiply(der_sig(t1).T,w2[0,0:2])).T


t = np.arange(-2,2,0.1)
z1 = (-w1[0,0]/w1[0,1])*t + (-w1[0,2]/w1[0,1])
z2 = (-w1[1,0]/w1[1,1])*t + (-w1[1,2]/w1[1,1])

p1 = fig.add_subplot(*[1,1,1])
p1.scatter(training[0:100,0],training[0:100,1],c="b")
p1.scatter(training[100:200,0],training[100:200,1],c="b")
p1.scatter(training[200:300,0],training[200:300,1],c="r")
p1.scatter(training[300:400,0],training[300:400,1],c="r")
p1.plot(t,z1)
p1.plot(t,z2)

#test data
count = 0 
for i in range(400):
    #input layer
    n11 = np.random.uniform(-2, 2)
    n12 = np.random.uniform(-2, 2)
    #final weight
    k11 = w1[0,0]
    k12 = w1[1,0]
    k21 = w1[0,1]
    k22 = w1[1,1]
    v11 = w1[0,2]
    v12 = w1[1,2]
    k31 = w2[0,0]
    k32 = w2[0,1]
    v3 = w2[0,2]
    #hidden layer 1
    p11 = k11*n11 + k21*n12 + v11
    p12 = k12*n11 + k22*n12 + v12
    n21 = 1/(1+np.exp(-p11))
    n22 = 1/(1+np.exp(-p12))    
    p3 = k31*n21 + k32*n22 + v3
    n = 1/(1+np.exp(-p3)) 
    if (n11*n12 >= 0):
        true = 0
        if n<=0.5:
            count = count + 1
    else: 
        true = 1
        if n>=0.5:
            count = count + 1
print(count/400)
    
print('finish')