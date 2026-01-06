#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 12:09:14 2026

@author: ed
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

ct1 = pd.read_csv("CherryTree.csv")

x1 = np.array(ct1['Diam']).reshape(-1,1)
y1 = np.array(ct1['Height']).reshape(-1,1)
y2 = np.array(ct1['Volume']).reshape(-1,1)

# Split the data for prediction testing
X_train, X_test, y_train, y_test = train_test_split(x1, y2, test_size=0.3, random_state=0)

# Fit the regression model
reg1 = LinearRegression()
reg1.fit( X_train, y_train )

# Predict the split data
y2_pred = reg1.predict( X_train )
y2_test = reg1.predict( X_test )

# Get the predictive error standard deviation
s1_pred = np.sqrt( mean_squared_error(y_test, y2_test) )

# Fit the full regression model
reg1 = LinearRegression()
reg1.fit( x1, y2 )

# Get generative predictions
y2_gen = reg1.predict( x1 ) + np.random.normal(0,s1_pred, x1.shape )

# Plot it
plt.scatter(x1,y2, c='blue')
plt.scatter(x1, y2_gen, c='red')
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')