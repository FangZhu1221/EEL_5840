# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:28:54 2017

@author: JSZJZ
"""


import numpy as np
import matplotlib.pyplot as plt
import math 
import time

def der_sig(x):
    t = (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))
    return t

w1 = np.matrix([[0,100,0],[100,0,0]])
w2 = np.random.uniform(size=(2, 3))
w3 = np.random.uniform(size=(1, 3))
x1 = np.zeros([1,3])
x2 = np.zeros([1,3])
x3 = np.zeros([1,3])
x = np.zeros(400)
v = np.zeros(400)
step1 = 0.2
step2 = 0.2
step3 = 0.2
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
fig = plt.figure(figsize=(20,10))

for j in range(1000):
    for i in range(length):
        x1[0,0] = training[i,0]
        x1[0,1] = training[i,1]
        x1[0,2] = 1
        t1 = w1@x1.T #2x1
        x2[0,0] = 1/(1+np.exp(-t1[0,0]))
        x2[0,1] = 1/(1+np.exp(-t1[1,0]))
        x2[0,2] = 1
        x[i] = x2[0,0]
        v[i] = x2[0,1]
        t2 = w2@x2.T
        x3[0,0] = 1/(1+np.exp(-t2[0,0]))
        x3[0,1] = 1/(1+np.exp(-t2[1,0]))
        x3[0,2] = 1
        t3 = w3@x3.T
        y = 1/(1+np.exp(-t3))
        e = validate[i]-y        
        w3 = w3 + step3*e*np.asscalar(der_sig(t3))*x3
        w2 = w2 + ((step2*np.asscalar(der_sig(t3))*e)*x2.T@np.multiply(der_sig(t2).T,w3[0,0:2])).T        
        w1 = w1
        if i == 999:
            print(y)


t = np.arange(-2,2,0.1)
z11 = (-w1[0,0]/w1[0,1])*t + (-w1[0,2]/w1[0,1])
z12 = (-w1[1,0]/w1[1,1])*t + (-w1[1,2]/w1[1,1])
z21 = (-w2[0,0]/w2[0,1])*t + (-w2[0,2]/w2[0,1])
z22 = (-w2[1,0]/w2[1,1])*t + (-w2[1,2]/w2[1,1])

p1 = fig.add_subplot(*[2,1,1])
p1.scatter(training[0:100,0],training[0:100,1],c="b")
p1.scatter(training[100:200,0],training[100:200,1],c="b")
p1.scatter(training[200:300,0],training[200:300,1],c="r")
p1.scatter(training[300:400,0],training[300:400,1],c="r")
p1.plot(t,z11)
p1.plot(np.zeros(length),t)
ymin, ymax = -2, 2
p1.set_ylim([ymin,ymax])

p2 = fig.add_subplot(*[2,1,2])
p2.scatter(x[0:100],v[0:100],c="b")
p2.scatter(x[100:200],v[100:200],c="b")
p2.scatter(x[200:300],v[200:300],c="r")
p2.scatter(x[300:400],v[300:400],c="r")
p2.plot(t,z21)
p2.plot(t,z22)
ymin, ymax = -2, 2
p2.set_ylim([ymin,ymax])

#test data    
for i in range(4):
    #input layer
    n11 = test_data[i*2]
    n12 = test_data[i*2+1]
    #final weight
    k11 = w1[0,0]
    k12 = w1[1,0]
    k21 = w1[0,1]
    k22 = w1[1,1]
    v11 = w1[0,2]
    v12 = w1[1,2]
    k31 = w2[0,0]
    k32 = w2[1,0]
    k41 = w2[0,1]
    k42 = w2[1,1]
    v21 = w2[0,2]
    v22 = w2[1,2]    
    k51 = w3[0,0]
    k52 = w3[0,1]
    v3 = w3[0,2]
    #hidden layer 1
    p11 = k11*n11 + k21*n12 + v11
    p12 = k12*n11 + k22*n12 + v12
    n21 = 1/(1+np.exp(-p11))
    n22 = 1/(1+np.exp(-p12))    
    p21 = k31*n21 + k42*n22 + v21
    p22 = k32*n21 + k42*n22 + v22
    n31 = 1/(1+np.exp(-p21))
    n32 = 1/(1+np.exp(-p22))
    h = k51*n31 + k52*n32 + v3
    n = 1/(1+np.exp(-h))
    print(n)
    
print('finish')

