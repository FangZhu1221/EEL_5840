# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:05:07 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import time

w11 = 0
w21 = 0
w12 = 0
w22 = 0
w31 = 0
w32 = 0
b1 = 0
b2 = 0
b3 = 0
x21 = 0
x22 = 0

step1 = -0.1
step2 = -0.1
test_data = np.zeros(2)
test_data[0] = -0.32
test_data[1] = -0.92
#sigmoid = 1/(1+np.exp())

data = np.loadtxt("HW6_Data.txt")
validate = data[:,2]
training = data[:,0:2]
length = validate.size
track1 = np.zeros(length)
track2 = np.zeros(length)
track3 = np.zeros(length)
error = np.zeros(length)
fig = plt.figure(figsize=(10,10))

for i in range(length):
    x11 = training[i,0]
    x12 = training[i,1]
    t1 = w11*x11 + w21*x12 + b1
    t2 = w12*x11 + w22*x12 + b2
    x21 = 1/(1+np.exp(-t1))
    x22 = 1/(1+np.exp(-t2))
    t3 = w31*x21 + w32*x22 + b3
    y = 1/(1+np.exp(-t3))
    e = validate[i]-y

    w31 = w31 + step2*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*e*x21
    w32 = w32 + step2*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*e*x22
    b3 = b3 + step2*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*e
     
    w11 = w11 + step1*((1/(1+np.exp(-(t1))))*(1-1/(1+np.exp(-(t1)))))*e*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*w31*x11
    w12 = w12 + step1*((1/(1+np.exp(-(t2))))*(1-1/(1+np.exp(-(t2)))))*e*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*w32*x11
    w21 = w21 + step1*((1/(1+np.exp(-(t1))))*(1-1/(1+np.exp(-(t1)))))*e*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*w31*x12
    w22 = w22 + step1*((1/(1+np.exp(-(t2))))*(1-1/(1+np.exp(-(t2)))))*e*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*w32*x12
    b1 = b1 + step2*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*e
    b2 = b2 + step2*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*e

    
t1 = w11*test_data[0] + w21*test_data[1] + b1
t2 = w12*test_data[0] + w22*test_data[1] + b2
x21 = 1/(1+np.exp(-t1))
x22 = 1/(1+np.exp(-t2))
t3 = w31*x21 + w32*x22 + b2
y = 1/(1+np.exp(-t3))
print(y)

t = np.arange(-2,0.1,2)
z1 = (-w11/w21)*t + (-b1/w21)
z2 = (-w12/w22)*t + (-b2/w22)

p1 = fig.add_subplot(*[1,1,1])
p1.scatter(training[0:100,0],training[0:100,1],c="b")
p1.scatter(training[100:200,0],training[100:200,1],c="b")
p1.scatter(training[200:300,0],training[200:300,1],c="r")
p1.scatter(training[300:400,0],training[300:400,1],c="r")
p1.plot(t,z1)
p1.plot(t,z2)

print('finish')