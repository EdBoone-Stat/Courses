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

b1 = pd.read_csv( "Battery.csv" ) 

plt.plot( b1['Time'], b1['Sales'])
plt.xlabel('Time')
plt.ylabel('Sales')

# Use sines and cosines
t1 = np.array( b1['Time'])
s1 = np.sin( 2*np.pi*t1/12 )
c1 = np.cos( 2*np.pi*t1/12 )
xdata = np.vstack([t1,s1,c1]).T


reg1 = LinearRegression()
reg1.fit(xdata, b1['Sales'] )
pred1 = reg1.predict( xdata )

plt.plot( b1['Time'], b1['Sales'],c="blue")
plt.plot( b1['Time'], pred1, c="red")
plt.xlabel('Time')
plt.ylabel('Sales')

# Use sines and cosines
s2 = np.sin( 2*np.pi*t1/12*2 )
c2 = np.cos( 2*np.pi*t1/12*2 )
xdata2 = np.vstack([t1,s1,c1,s2,c2]).T

reg2 = LinearRegression()
reg2.fit(xdata2, b1['Sales'] )
pred2 = reg2.predict( xdata2 )

plt.plot( b1['Time'], b1['Sales'],c="blue")
plt.plot( b1['Time'], pred2, c="red")
plt.xlabel('Time')
plt.ylabel('Sales')

# Use sines and cosines
s3 = np.sin( 2*np.pi*t1/12*3 )
c3 = np.cos( 2*np.pi*t1/12*3 )
xdata3 = np.vstack([t1,s1,c1,s2,c2,s3,c3]).T

reg3 = LinearRegression()
reg3.fit(xdata3, b1['Sales'] )
pred3 = reg3.predict( xdata3 )

plt.plot( b1['Time'], b1['Sales'],c="blue")
plt.plot( b1['Time'], pred3, c="red")
plt.xlabel('Time')
plt.ylabel('Sales')


# Use sines and cosines
s4 = np.sin( 2*np.pi*t1/12*4 )
c4 = np.cos( 2*np.pi*t1/12*4 )
xdata4 = np.vstack([t1,s1,c1,s2,c2,s3,c3,s4,c4]).T

reg4 = LinearRegression()
reg4.fit(xdata4, b1['Sales'] )
pred4 = reg4.predict( xdata4 )

plt.plot( b1['Time'], b1['Sales'],c="blue")
plt.plot( b1['Time'], pred4, c="red")
plt.xlabel('Time')
plt.ylabel('Sales')


# Use sines and cosines
s5 = np.sin( 2*np.pi*t1/12*5 )
c5 = np.cos( 2*np.pi*t1/12*5 )
s6 = np.sin( 2*np.pi*t1/12*6 )
c6 = np.cos( 2*np.pi*t1/12*6 )
xdata6 = np.vstack([t1,s1,c1,s2,c2,s3,c3,s4,c4,s5,c5,s6,c6]).T

reg6 = LinearRegression()
reg6.fit(xdata6, b1['Sales'] )
pred6 = reg6.predict( xdata6 )

plt.plot( b1['Time'], b1['Sales'],c="blue")
plt.plot( b1['Time'], pred6, c="red")
plt.xlabel('Time')
plt.ylabel('Sales')

xdata0 = np.zeros( [t1.shape[0], 25 ] )
count1 = 1
k1 = 1
xdata0[:,0] = t1
for i in range(12):
    xdata0[:,count1] = np.sin( 2*np.pi*t1/k1)
    count1 = count1 + 1
    xdata0[:,count1] = np.cos( 2*np.pi*t1/k1)
    count1 = count1 + 1
    k1 = k1 +1 

reg12 = LinearRegression()
reg12.fit(xdata0, b1['Sales'] )
pred12 = reg12.predict( xdata0 )

plt.plot( b1['Time'], b1['Sales'],c="blue")
plt.plot( b1['Time'], pred12, c="red")
plt.xlabel('Time')
plt.ylabel('Sales')

tpred0 = np.linspace( np.max( t1 )+1, np.max(t1) + 12, 12)

xpred0 = np.zeros( [tpred0.shape[0], 25 ] )
count1 = 1
k1 = 1
xpred0[:,0] = tpred0
for i in range(12):
    xpred0[:,count1] = np.sin( 2*np.pi*tpred0/k1)
    count1 = count1 + 1
    xpred0[:,count1] = np.cos( 2*np.pi*tpred0/k1)
    count1 = count1 + 1
    k1 = k1 +1 

pred12f = reg12.predict(xpred0)

plt.plot( b1['Time'], b1['Sales'],c="blue")
plt.plot( b1['Time'], pred12, c="red")
plt.plot( tpred0, pred12f, c="red")
plt.xlabel('Time')
plt.ylabel('Sales')


t_temp0 = np.linspace( 0, 4*np.pi, 200 )
y_s1 = np.sin( t_temp0 )
y_c1 = np.cos( t_temp0 )
y_0 = t_temp0*0

plt.plot( t_temp0, y_s1, c="blue", label = 'sine' )
plt.plot( t_temp0, y_c1, c="red" , label = 'cosine' )
plt.plot( t_temp0, y_0, c="black")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Y')
