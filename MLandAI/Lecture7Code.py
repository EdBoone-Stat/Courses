#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 10:33:41 2026

@author: ed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ct1 = pd.read_csv("CherryTree.csv")

x1 = np.array(ct1['Diam']).reshape(-1,1)
y1 = np.array(ct1['Height']).reshape(-1,1)
y2 = np.array(ct1['Volume']).reshape(-1,1)

reg1 = LinearRegression()
reg1.fit(x1,y2)
reg1.intercept_
reg1.coef_
y2_pred1 = reg1.predict(x1)
r2_score(y2,y2_pred1)

reg2 = LinearRegression()
reg2.fit(y1,y2)
reg2.intercept_
reg2.coef_
y2_pred2 = reg2.predict(y1)
r2_score(y2,y2_pred2)

x_data = np.hstack([x1,y1])
reg3 = LinearRegression()
reg3.fit(x_data,y2)
reg3.intercept_
reg3.coef_
y2_pred3 = reg3.predict(x_data)
r2_score(y2,y2_pred3)
