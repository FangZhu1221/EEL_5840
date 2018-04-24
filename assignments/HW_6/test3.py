# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:15:39 2017

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
x21 = 0
x22 = 0

step1 = 0
step2 = 0
step3 = 0.1
test_data = np.zeros(2)
test_data[0] = -0.32
test_data[1] = -0.92
#sigmoid = 1/(1+np.exp())

data = np.loadtxt("HW6_Data.txt")
validate = data[:,2]
training = data[:,0:2]
length = validate.size

for i in range(length):
    x11 = training[i,0]
    x12 = training[i,1]
    t1 = w11*x11 + w21*x12 + b1
    t2 = w12*x11 + w22*x12 + b1
    x21 = 1/(1+np.exp(-t1))
    x22 = 1/(1+np.exp(-t2))
    t3 = w31*x21 + w32*x22 + b2
    y = 1/(1+np.exp(-t3))
    e = validate[i]-y
    w11 = w11 + step1*((1/(1+np.exp(-(t1))))*(1-1/(1+np.exp(-(t1)))))*e*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*w31*x11
    w12 = w12 + step1*((1/(1+np.exp(-(t2))))*(1-1/(1+np.exp(-(t2)))))*e*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*w32*x11
    w21 = w21 + step2*((1/(1+np.exp(-(t1))))*(1-1/(1+np.exp(-(t1)))))*e*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*w31*x12
    w22 = w22 + step2*((1/(1+np.exp(-(t2))))*(1-1/(1+np.exp(-(t2)))))*e*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*w32*x12
    w31 = w31 + step3*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*e*x21
    w32 = w32 + step3*((1/(1+np.exp(-(t3))))*(1-1/(1+np.exp(-(t3)))))*e*x22
    
t1 = w11*test_data[0] + w21*test_data[1] + b1
t2 = w12*test_data[0] + w22*test_data[1] + b1
x21 = 1/(1+np.exp(-t1))
x22 = 1/(1+np.exp(-t2))
t3 = w31*x21 + w32*x22 + b2
y = 1/(1+np.exp(-t3))
print(y)