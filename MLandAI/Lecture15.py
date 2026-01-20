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

reg1 = LinearRegression()
reg1.fit( X_1, mc1['Y'])
pred1 = reg1.predict( X_1 )

plt.plot( mc1['X'], mc1['Y'], c = "blue")
plt.plot( mc1['X'], pred1, c ="red" )
plt.scatter( k1, np.zeros(n_k1), c="orange" )
plt.xlabel('X')
plt.ylabel('Y')

