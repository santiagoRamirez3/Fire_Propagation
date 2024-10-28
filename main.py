from classes import simulation
from menu import menu
import numpy as np

import matplotlib.pyplot as plt


if __name__ == '__main__':
    usrChoice = menu()
    matrix = np.ones((100,100))
    matrix[50,50] = 2
    
    if usrChoice ==1:
        forest = simulation.squareForest(burningThreshold=0.55, initialForest=matrix, saveHistoricalPropagation=True)
        forest.animate('intento_1')
    
    elif usrChoice == 2:
        n = 50    # Amount of values to consider for p
        m = 15      # Amount of trials per p 
        saveRoute = './graphs/finalTimes.png'
        
        forest = simulation.squareForest(burningThreshold=0.55, initialForest=matrix)
        forest.propagationTime(saveRoute,n,m, matrix)
            
    elif usrChoice == 3:
        n = 35    # Amount of values to consider for p
        m = 5      # Amount of trials per p        
        saveRoute = './graphs/percolationThreshold.png'
        
        forest = simulation.squareForest(burningThreshold=0.55, initialForest=matrix)
        p_c = forest.percolationThreshold(saveRoute,n,m,matrix,True)
        
    elif usrChoice == 4:
        n = 20    # Amount of values to consider for p in the range (0,1)to fin p_c
        m1 = 5      # Amount of trials per p        
        m2 = 10     # Amount of trials per p to find M
        saveRoute = './graphs/percolationThreshold.png'
        epsilon = 0.1
        delta = 0.001
        
        forest = simulation.squareForest(burningThreshold=0.55, initialForest=matrix)
        criticalExponent = forest.criticalExponent(saveRoute,epsilon,delta,n,m1,m2,matrix)
        print(criticalExponent)
            
        
    