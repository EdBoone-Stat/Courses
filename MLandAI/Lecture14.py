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

sp1 = pd.read_csv( "Spline1Data.csv" ) 

plt.scatter( sp1['x'], sp1['y'])
plt.xlabel('X')
plt.ylabel('Y')

reg1 = LinearRegression()
reg1.fit( sp1[['x']], sp1['y'])
pred1 = reg1.predict( sp1[['x']] )
plt.scatter( sp1['x'], sp1['y'])
plt.plot( sp1['x'], pred1, c = "red")
plt.xlabel('X')
plt.ylabel('Y')

reg1.coef_
reg1.intercept_

sp2 = sp1[ sp1['x'] <= 30 ]
reg2 = LinearRegression()
reg2.fit( sp2[['x']], sp2['y'])
pred2 = reg2.predict( sp2[['x']] )

reg2.coef_
reg2.intercept_

sp3 = sp1.query("x  >= 30 and x <= 60"  )
reg3 = LinearRegression()
reg3.fit( sp3[['x']], sp3['y'])
pred3 = reg3.predict( sp3[['x']] )

reg3.coef_
reg3.intercept_

sp4 = sp1[ sp1['x'] >= 60 ]
reg4 = LinearRegression()
reg4.fit( sp4[['x']], sp4['y'])
pred4 = reg4.predict( sp4[['x']] )

reg4.coef_
reg4.intercept_

plt.scatter( sp1['x'], sp1['y'])
plt.plot( sp2['x'], pred2, c = "red")
plt.plot( sp3['x'], pred3, c = "blue")
plt.plot( sp4['x'], pred4, c = "orange")
plt.xlabel('X')
plt.ylabel('Y')


def basis1( x1, knot1, degree1 = 3):
    res1 = np.zeros( x1.shape )
    res1[ x1 > knot1 ] = ( x1[x1 > knot1] - knot1 )**degree1
    return res1

x2a = np.linspace(-3,3,11)
x2 = basis1( x2a, knot1 = 0.0, degree1 = 1 )

plt.plot( x2a, x2, c="blue")
plt.xlabel('x')
plt.ylabel(r'$x_+$')
plt.title('Reticulated Linear Unit')

# Get the basis functions
xb1 = basis1( sp1['x'], knot1 = 30, degree1 = 1)
xb2 = basis1( sp1['x'], knot1 = 60, degree1 = 1)

# Combine it all together into a dataframe
xdata = pd.DataFrame( {'x': sp1['x'], 'x1': xb1, 'x2': xb2})

regSpline1 = LinearRegression()
regSpline1.fit( xdata, sp1['y'])
regSpline1pred = regSpline1.predict( xdata )

regSpline1.intercept_
regSpline1.coef_

plt.scatter( sp1['x'], sp1['y'])
plt.plot( sp1['x'], regSpline1pred, c = "red")
plt.xlabel('X')
plt.ylabel('Y')

x1q = sp1['x'].pow(2)
xb1q = xb1**2
xb2q = xb2**2
xdataq = pd.DataFrame( {'x': sp1['x'], 'x2': x1q, 'x3': xb1q, 'x4': xb2q } )

regSpline2 = LinearRegression()
regSpline2.fit( xdataq, sp1['y'])
regSpline2pred = regSpline2.predict( xdataq )

plt.scatter( sp1['x'], sp1['y'])
plt.plot( sp1['x'], regSpline2pred, c = "red")
plt.xlabel('X')
plt.ylabel('Y')


x1q = sp1['x'].pow(2)
x1c = sp1['x'].pow(3)
xb1c = xb1**3
xb2c = xb2**3
xdatac = pd.DataFrame( {'x': sp1['x'], 'x2': x1q, 'x3': x1c, 'x4': xb1c, 'x5': xb2c } )

regSpline3 = LinearRegression()
regSpline3.fit( xdatac, sp1['y'])
regSpline3pred = regSpline3.predict( xdatac )

plt.scatter( sp1['x'], sp1['y'])
plt.plot( sp1['x'], regSpline3pred, c = "red")
plt.xlabel('X')
plt.ylabel('Y')


