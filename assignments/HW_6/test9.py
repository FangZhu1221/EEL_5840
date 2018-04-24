# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:58:28 2017

@author: JSZJZ
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import time



seed(2)
g = np.array([np.random.uniform(-2,2) for i in range(200)]).reshape((100,2))
l = np.zeros([100,1])
for i in range(100):
    if (g[i,0]>0 and g[i,1]>0) or (g[i,0]<0 and g[i,1]<0):
        l[i] = 0
    else:
        l[i] = 1
p = model(np.c_[np.ones([100,1]),g],W_i,W_h)
for i in range(100):
    if p[i]>0.5:
        p[i]=1
    else:
        p[i]=0
plt.scatter(g[:,0],g[:,1])
confusion_matrix(l,p)
