# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:16:19 2017

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

w1 = np.random.uniform(size=(1, 4))
x1 = np.zeros([1,4])

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
        x1[0,2] = training[i,0]*training[i,1]
        x1[0,3] = 1
        t1 = w1@x1.T #2x1
        y = 1/(1+np.exp(-t1))
        e = validate[i]-y
        w1 = w1 + step2*e*np.asscalar(der_sig(t1))*x1


x1 = np.arange(-2,2,0.1)
x2 = np.arange(-2,2,0.1)
X,Y = np.meshgrid(x1,x2)
Z1 = w1[0,0]*X + w1[0,1]*Y + w1[0,2]*X*Y + w1[0,3]

p1 = fig.add_subplot(*[1,1,1])
p1.scatter(training[0:100,0],training[0:100,1],c="b")
p1.scatter(training[100:200,0],training[100:200,1],c="b")
p1.scatter(training[200:300,0],training[200:300,1],c="r")
p1.scatter(training[300:400,0],training[300:400,1],c="r")
#p1.contour(X,Y,Z1,0)
p1.contour(X,Y,Z1,0)

#test data
count = 0 
for i in range(400):
    #input layer
    n11 = np.random.uniform(-2, 2)
    n12 = np.random.uniform(-2, 2)
    n13 = n11*n12
    #final weight
    k11 = w1[0,0]
    k21 = w1[0,1]
    k31 = w1[0,2]
    v11 = w1[0,3]
    #hidden layer 1
    p = k11*n11 + k21*n12 + k31*n13 + v11
    n = 1/(1+np.exp(-p))    
    if (n13 >= 0):
        true = 0
        if n<=0.5:
            count = count + 1
    else: 
        true = 1
        if n>=0.5:
            count = count + 1
print(count/400)
    
print('finish')