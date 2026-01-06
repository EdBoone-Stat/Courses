#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 13:05:51 2025

@author: ed
"""

# Load libraries we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read in the dataset
ct1 = pd.read_csv("CherryTree.csv")

# Shape the datasets into something we can use
x1 = np.array(ct1['Diam']).reshape(-1,1)
y1 = np.array(ct1['Height']).reshape(-1,1)
y2 = np.array(ct1['Volume']).reshape(-1,1)

#Make a picture of the data
plt.scatter(x1,y2)
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')

# Run the regression line
reg1 = LinearRegression()
reg1.fit( x1, y2 )

# Get the coefficients
b1 = reg1.coef_
a1 = reg1.intercept_

