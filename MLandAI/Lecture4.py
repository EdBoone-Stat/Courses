# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

plt.scatter(x1,y1)
plt.xlabel('Diameter (in)')
plt.ylabel('Height (ft)')


plt.scatter(x1,y2)
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')

plt.scatter(x1,y2)
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')
plt.plot( [np.min(x1),np.max(x1)],[np.median(y2),np.median(y2)], c='red')


reg1 = LinearRegression()
reg1.fit( x1, y2 )

b1 = reg1.coef_
a1 = reg1.intercept_

y2_pred = reg1.predict( x1 )

plt.scatter(x1,y2)
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')
plt.plot( x1, y2_pred, c = "red" )

np.median( y2 )

plt.scatter(x1,y2)
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')
plt.plot( x1, y2_pred, c = "red" )
for i in range(x1.shape[0]):
    plt.plot( [x1[i], x1[i]], [y2[i],y2_pred[i]], c = "orange")


resid1 = y2 - y2_pred
plt.scatter(x1, resid1)
plt.xlabel('Diameter (in)')
plt.ylabel(r'$r=y-\hat{y}$')
plt.plot( [np.min(x1),np.max(x1)],[0,0], c='black')


plt.scatter(x1,y2)
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')
plt.plot( [np.min(x1),np.max(x1)],[np.mean(y2),np.mean(y2)], c='red')


y2_mean = np.ones( y2.shape)*np.mean(y2)
plt.scatter(x1,y2)
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')
plt.plot( [np.min(x1),np.max(x1)],[np.mean(y2),np.mean(y2)], c='red')
for i in range(x1.shape[0]):
    plt.plot( [ x1[i], x1[i] ], [ y2[i], y2_mean[i] ], c = "orange")


# Split the data for prediction testing
X_train, X_test, y_train, y_test = train_test_split(x1, y2, test_size=0.3, random_state=0)
   

plt.scatter(X_train,y_train, c = "blue" )
plt.scatter(X_test, y_test, c = "orange" )
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')

reg1 = LinearRegression()
reg1.fit( X_train, y_train )

y2_pred = reg1.predict( X_train )
y2_test = reg1.predict( X_test )

plt.scatter(X_train,y_train, c = "blue" )
plt.scatter(X_test, y_test, c = "orange" )
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')
plt.plot( X_train, y2_pred, c='red')


plt.scatter(X_test, y_test, c = "orange" )
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')
plt.plot( X_train, y2_pred, c='red')
for i in range(X_test.shape[0]):
    plt.plot( [ X_test[i], X_test[i] ], [ y_test[i], y2_test[i] ], c = "orange")


s1_pred = np.sqrt( mean_squared_error(y_test, y2_test) )

reg1 = LinearRegression()
reg1.fit( x1, y2 )

y2_gen = reg1.predict( x1 ) + np.random.normal(0,s1_pred, x1.shape )

plt.scatter(x1,y2, c='blue')
plt.scatter(x1, y2_gen, c='red')
plt.xlabel('Diameter (in)')
plt.ylabel(r'Volume ($ft^3$)')





