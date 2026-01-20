#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 08:49:31 2026

@author: ed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
ct1 = pd.read_csv("CherryTree.csv")
x1 = np.array(ct1['Diam']).reshape(-1,1)
y1 = np.array(ct1['Height']).reshape(-1,1)
y2 = np.array(ct1['Volume']).reshape(-1,1)
x_data = np.hstack([x1,y1])
# Split the data
X_train, X_test, y_train, y_test = train_test_split(x_data, y2, test_size=0.3, random_state=5)
# Fit the regression model 1
reg1 = LinearRegression()
reg1.fit( X_train[:,0].reshape(-1, 1), y_train )
y2_test1 = reg1.predict( X_test[:,0].reshape(-1, 1) )
mean_squared_error(y_test, y2_test1) 
# Fit the regression model 2
reg2 = LinearRegression()
reg2.fit( X_train[:,1].reshape(-1, 1), y_train )
y2_test2 = reg2.predict( X_test[:,1].reshape(-1, 1) )
mean_squared_error(y_test, y2_test2) 
# Fit the regression model 3
reg3 = LinearRegression()
reg3.fit( X_train, y_train )
y2_test3 = reg3.predict( X_test )
mean_squared_error(y_test, y2_test3) 
reg3.intercept_
reg3.coef_
plt.scatter( y_test, y2_test3 )
plt.xlabel('y')
plt.ylabel(r'$\hat{y}$')
plt.plot( [10,80],[10,80], c='red')
r2_score( y_test, y2_test3 )
