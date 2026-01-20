#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 17:15:41 2026

@author: ed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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
y1_pred = reg2.predict(x1)
a1 = reg2.intercept_
b1 = reg2.coef_

confusion_matrix(y1,y1_pred)

y3_p = np.exp( a1 + b1*x1 )/(1+np.exp(a1 + b1*x1 ) )
U1 = np.random.uniform(0,1,x1.shape)
y3_gen = np.zeros( y3_p.shape )
y3_gen[ U1 < y3_p ] = 1
