import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def expFit(data, propTimeThreshold:int):
    
    upperValley = data[(data['P_site'] > 0.9) & (data['P_bond'] > 0.9)]
    upperValleyMeanTime = np.mean(upperValley['time'])
    propTimeThreshold = upperValleyMeanTime + (np.max(data['time']) - upperValleyMeanTime) * 0.7

    print(np.mean(upperValley['time']), propTimeThreshold)
    # Filter those of interest based on propagation time
    phaseTransitionData = data[ data['time'] > propTimeThreshold ]
    
    # Extrac the desired data for fitting
    ps =  phaseTransitionData['P_site']
    pb =  phaseTransitionData['P_bond']
    
    
    function = lambda ps,A,B,C : A * np.exp( -B *ps ) + C
    
    # Execute the fit 
    popt, pcov = curve_fit(function, ps, pb,maxfev=5000)
    
    a,b,c = popt
    print(a*np.exp(-b) + c)


    return function,ps,pb,popt
    
    






