#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 15:07:42 2026

@author: ed
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split 

bh1_r = pd.read_table( "bostonhousing.csv", delimiter='\t') 
bh2_r = bh1_r.loc[ pd.notna(bh1_r['MEDV']) ]
y1 = bh2_r['MEDV']
x1 = bh2_r['LSTAT']
x2 = bh2_r['LSTAT'].pow(2)
x3 = bh2_r['LSTAT'].pow(3)
x_data1 = pd.DataFrame({'LSTAT': x1, 'LSTAT2': x2, 'LSTAT3': x3})

X_train, X_test, y_train, y_test = train_test_split(x_data1, y1, test_size=0.5, random_state=0)

reg1 = LinearRegression()
reg1.fit( X_train[['LSTAT']], y_train )
y_pred1 = reg1.predict(X_test[['LSTAT']]) 
r2_score(y_pred1, y_test)

reg2 = LinearRegression()
reg2.fit( X_train[['LSTAT','LSTAT2']], y_train )
y_pred2 = reg2.predict(X_test[['LSTAT','LSTAT2']]) 
r2_score(y_pred2, y_test)

reg3 = LinearRegression()
reg3.fit( X_train[['LSTAT','LSTAT2','LSTAT3']], y_train )
y_pred3 = reg3.predict(X_test[['LSTAT','LSTAT2','LSTAT3']]) 
r2_score(y_pred3, y_test)

mean_squared_error( y_pred1, y_test )
mean_squared_error( y_pred2, y_test )
mean_squared_error( y_pred3, y_test )

