# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:29:58 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import time

w11 = 1
w21 = 0
w12 = 0
w22 = 1
w31 = -0.5
w32 = 0.5
b1 = 0
b2 = 0
b3 = 0

data = np.loadtxt("HW6_Data.txt")
validate = data[:,2]
training = data[:,0:2]
length = validate.size
classified = 2*np.ones([length,1])

for i in range(length):
    x = training[i,0]
    y = training[i,1]
    g = (w11*x + w21*y + b1) + (w12*x + w22*y + b2) + b3
    if g == 0:
        classified[i] = 0
    else:
        if g > 0:
            classified[i] = 1

error = 0
for i in range(length):
    if validate[i] != classified[i]:
        error = error + 1

print(error)