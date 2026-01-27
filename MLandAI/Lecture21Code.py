#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:51:05 2026

@author: ed
"""

import numpy as np
import scipy as sc
import pandas as pd

def normdiscrim( xA, xB ):
    xAmean = np.mean( xA )
    xBmean = np.mean( xB )
    xAsd = np.std( xA )
    xBsd = np.std( xB )
    def f1(x, xAmean, xAsd, xBmean, xBsd ):
        fA1 = sc.stats.norm.pdf( x, xAmean, xAsd )
        fB1 = sc.stats.norm.pdf( x, xBmean, xBsd )
        res1 = fA1 - fB1
        return res1
    x0 = (xAmean + xBmean)/2
    out1 = sc.optimize.newton(f1, x0,
                              args = (xAmean, xAsd, xBmean, xBsd) )
    return out1 

data0 = pd.read_csv("Driving.csv")

dataY = data0.loc[ data0["Passed"] == "yes", 'Hours']
dataN = data0.loc[ data0["Passed"] == "no", 'Hours']

normdiscrim( dataY, dataN )

