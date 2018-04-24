# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:41:56 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import time

#load data
data = np.loadtxt("HW6_Data.txt")

#define function
def der_sig(x = np.asmatrix(np.zeros([400,1]))):
    b = np.asmatrix(np.zeros([400,1]))
    for i in range(len(x)):
        t = (1/(1+np.exp(-x[i,0])))*(1-(1/(1+np.exp(-x[i,0]))))
        b[i,0] = t
    return b

def sum1(x = np.asmatrix(np.zeros([400,1]))):
    t = 0
    for i in range(len(x)):
        t = t + x[i,0]
    return t

#initial parameters
#step size
step1 = 0.01
step2 = 0.1
step3 = 0.1
#validate data
validate = data[:,2]
#weight matrix
w11 = np.asmatrix(np.zeros([400,1]))
w12 = np.asmatrix(np.zeros([400,1]))
b11 = np.asmatrix(np.zeros([400,1]))
w21 = np.asmatrix(np.zeros([400,1]))
w22 = np.asmatrix(np.zeros([400,1]))
b12 = np.asmatrix(np.zeros([400,1]))
w31 = np.asmatrix(np.zeros([400,1]))
w32 = np.asmatrix(np.zeros([400,1]))
b21 = np.asmatrix(np.zeros([400,1]))
w41 = np.asmatrix(np.zeros([400,1]))
w42 = np.asmatrix(np.zeros([400,1]))
b22 = np.asmatrix(np.zeros([400,1]))
w51 = np.asmatrix(np.zeros([400,1]))
w52 = np.asmatrix(np.zeros([400,1]))
b3 = np.asmatrix(np.zeros([400,1]))
#input data
x11 = np.asmatrix(np.zeros([400,1]))
x12 = np.asmatrix(np.zeros([400,1]))
x13 = np.asmatrix(np.zeros([400,1]))
x21 = np.asmatrix(np.zeros([400,1]))
x22 = np.asmatrix(np.zeros([400,1]))
x23 = np.asmatrix(np.zeros([400,1]))
x31 = np.asmatrix(np.zeros([400,1]))
x32 = np.asmatrix(np.zeros([400,1]))
x33 = np.asmatrix(np.zeros([400,1]))
y = np.asmatrix(np.zeros([400,1])) 
#fill the input data
x11 = np.asmatrix(data[:,0]).T
x12 = np.asmatrix(data[:,1]).T
x13 = np.asmatrix(np.ones([400])).T
#test data
test_data = np.zeros(2)
test_data[0] = -0.32
test_data[1] = 0.92

for i in range(1000):
    #calculate the answers for the output layer
    #hidden layer 1
    t11 = np.multiply(x11,w11) + np.multiply(w21,x12) + np.multiply(b11,x13)
    t12 = np.multiply(w12,x11) + np.multiply(w22,x12) + np.multiply(b12,x13)

    x21 = 1/(1+np.exp(-t11))
    x22 = 1/(1+np.exp(-t12)) 
    x23 = np.asmatrix(np.ones([400]))
    #hidden layer 2
    t21 = np.multiply(w31,x21) + np.multiply(w41,x22) + np.multiply(b21,x23)
    t22 = np.multiply(w32,x21) + np.multiply(w42,x22) + np.multiply(b22,x23)
    x31 = 1/(1+np.exp(-t21))
    x32 = 1/(1+np.exp(-t22))  
    x33 = np.asmatrix(np.ones([400]))
    #output layer
    t3 = np.multiply(w51,x31) + np.multiply(w52,x32) + np.multiply(b3,x33)
    y = 1/(1+np.exp(-t3))
    #error
    e = np.asmatrix(validate) - y
        
    #calculate the backpropagation 
    #output layer
    w51 = w51 + step3*np.multiply(np.multiply(der_sig(t3),e),x31)
    w52 = w52 + step3*np.multiply(np.multiply(der_sig(t3),e),x32)
    b3 = b3 + step3*np.multiply(np.multiply(der_sig(t3),e),x33)
    #hidden layer 2
    w31 = w31 + step2*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t21),e),der_sig(t3)),w51),x21)
    w32 = w32 + step2*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t22),e),der_sig(t3)),w52),x21)
    w41 = w41 + step2*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t21),e),der_sig(t3)),w51),x22)
    w42 = w42 + step2*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t22),e),der_sig(t3)),w52),x22)
    b21 = b21 + step2*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t21),e),der_sig(t3)),b3),x23)
    b22 = b22 + step2*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t22),e),der_sig(t3)),b3),x23)
    #hidden layer 1
    w11 = w11 + step1*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t11),e),der_sig(t3)),w51),x11)
    w12 = w12 + step1*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t12),e),der_sig(t3)),w52),x11)
    w21 = w21 + step1*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t11),e),der_sig(t3)),w51),x12)
    w22 = w22 + step1*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t12),e),der_sig(t3)),w52),x12)
    b11 = b11 + step1*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t11),e),der_sig(t3)),b3),x13)
    b12 = b12 + step1*np.multiply(np.multiply(np.multiply(np.multiply(der_sig(t12),e),der_sig(t3)),b3),x13)
    
    #fill the input data
    x11 = np.asmatrix(data[:,0])
    x12 = np.asmatrix(data[:,1])
    x13 = np.asmatrix(np.ones([400]))

#final weight
k11 = sum1(w11)/400
k12 = sum1(w12)/400
k21 = sum1(w21)/400
k22 = sum1(w22)/400
v11 = sum1(b11)/400
v12 = sum1(b12)/400
k31 = sum1(w31)/400
k32 = sum1(w32)/400
k41 = sum1(w41)/400
k42 = sum1(w42)/400
v21 = sum1(b21)/400
v22 = sum1(b22)/400
k51 = sum1(w51)/400
k52 = sum1(w52)/400
v3 = sum1(b3)/400

#test data    
#input layer
n11 = test_data[0]
n12 = test_data[1]
#hidden layer 1
p11 = k11*n11 + k21*n12 + v11
p12 = k12*n11 + k22*n12 + v12
n21 = 1/(1+np.exp(-p11))
n22 = 1/(1+np.exp(-p12))    
#hidden layer 2
p21 = k31*n21 + k41*n22 + v21
p22 = k32*n21 + k42*n22 + v22
n31 = 1/(1+np.exp(-p21))
n32 = 1/(1+np.exp(-p22))  
#output layer
p3 = k51*n31 + k52*n32 + v3
n = 1/(1+np.exp(-p3)) 
print(n)
