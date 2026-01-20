#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 15:08:38 2026

@author: ed
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

bmd1 = pd.read_csv("bmd.csv")
# Define fracture or no fracture as 0 or 1
bmd1.loc[bmd1['fracture']=='no fracture', 'frac'] = 0
bmd1.loc[bmd1['fracture']=='fracture', 'frac'] = 1

# Determine the X and Y variables
x1 = np.array(bmd1['bmd']).reshape(-1,1)
y1 = np.array(bmd1['frac']).reshape(-1,1)

plt.scatter( x1, y1 )
plt.xlabel('bmd')
plt.ylabel('fracture')

reg2 = LogisticRegression()
reg2.fit( x1, y1.ravel() )

x3 = np.linspace( np.min(x1),np.max(x1), 101 )
a1 = reg2.intercept_
b1 = reg2.coef_
y3 = np.exp( a1 + b1*x3 )/(1+np.exp(a1 + b1*x3 ) )

plt.scatter( x1, y1 )
plt.plot( x3, y3.reshape(-1,1), c = "red" )
plt.xlabel('bmd')
plt.ylabel('fracture')
