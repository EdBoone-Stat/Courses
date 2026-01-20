#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 15:08:38 2026

@author: ed
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
bmd1 = pd.read_csv("bmd.csv")
bmd1.loc[bmd1['fracture']=='no fracture', 'frac'] = 0
bmd1.loc[bmd1['fracture']=='fracture', 'frac'] = 1
bmd1.loc[bmd1['sex']=='F', 'MF' ] = 1
bmd1.loc[bmd1['sex']=='M', 'MF' ] = 0
x_data1 = bmd1[['bmd','MF','weight_kg','height_cm','age']]
y1 = np.array(bmd1['frac']).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(x_data1, y1, test_size=0.5, random_state=0)
reg1 = LogisticRegression()
reg1.fit( np.array(X_train['bmd']).reshape(-1,1), y_train.ravel() )
y1_pred = reg1.predict( np.array(X_test['bmd']).reshape(-1,1) )
confusion_matrix(y_test,y1_pred)
reg2 = LogisticRegression()
reg2.fit( X_train[['bmd','MF']], y_train.ravel() )
y1_pred2 = reg2.predict( X_test[['bmd','MF']] )   
confusion_matrix(y_test,y1_pred2)
reg3 = LogisticRegression()
reg3.fit( X_train[['bmd','MF','weight_kg']], y_train.ravel() )
y1_pred3 = reg3.predict( X_test[['bmd','MF','weight_kg']] )   
confusion_matrix(y_test,y1_pred3)
reg4 = LogisticRegression()
reg4.fit( X_train[['bmd','MF','weight_kg','height_cm']], y_train.ravel() )
y1_pred4 = reg4.predict( X_test[['bmd','MF','weight_kg','height_cm']] )   
confusion_matrix(y_test,y1_pred4)
reg5 = LogisticRegression()
reg5.fit( X_train[['bmd','MF','weight_kg','height_cm','age']], y_train.ravel() )
y1_pred5 = reg5.predict( X_test[['bmd','MF','weight_kg','height_cm','age']] )   
confusion_matrix(y_test,y1_pred5)








