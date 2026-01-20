#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:53:19 2026

@author: ed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

mc1 = pd.read_csv( "MC1.csv" ) 

plt.plot( mc1['X'], mc1['Y'])
plt.xlabel('X')
plt.ylabel('Y')

# Basic Spline parameters
n_k1 = 10    # number of knots
degree1 = 2  # Degree of spline
#  Create knots
k1 = np.linspace( np.min(mc1['X']),np.max(mc1['X']), n_k1)

# Create Basis Function
def basis1( x1, knot1, degree1 = 3):
    res1 = np.zeros( x1.shape )
    res1[ x1 > knot1 ] = ( x1[x1 > knot1] - knot1 )**degree1
    return res1

def SplineBuild1( x1, k1, degree1 = 2):
    x1_tmp = np.array( x1 )
    x1_k1_tmp = np.array( x1 )
    for i in range( 2,(degree1+1) ):
        x2_tmp = x1_tmp**i
        x1_tmp = np.vstack( [x1_tmp, x2_tmp] )
    for i in range( k1.shape[0] ):
        x2_tmp = basis1( x1_k1_tmp, k1[i], degree1 = degree1)
        x1_tmp = np.vstack( [x1_tmp, x2_tmp] )
    return x1_tmp.T

X_1 = SplineBuild1( mc1['X'], k1, degree1 = 2 )

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_1, mc1['Y'], test_size=0.3, random_state=5)

reg1 = LinearRegression()
reg1.fit( X_train, y_train)
pred1 = reg1.predict( X_test )
r2_score( y_test, pred1 )

plt.scatter( y_test, pred1, c = "blue")
plt.plot( [0,2], [0,2], c ="red" )
plt.xlabel('X')
plt.ylabel('Y')

