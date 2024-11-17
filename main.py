from classes.regular import simulation
from classes.voronoi import voronoi_fire

from menu import menu
import numpy as np
from scipy.spatial import Voronoi

import matplotlib.pyplot as plt


if __name__ == '__main__':
    usrChoice = menu()
    matrix = np.ones((100,100))
    matrix[50,50] = 2
    
    if usrChoice ==1:
        forest = simulation.triangularForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
        forest.animate('intento_5')
    
    elif usrChoice == 2:
        n = 150    # Amount of values to consider for p
        m = 150      # Amount of trials per p 
        saveRoute = './graphs/finalTimes.png'
        
        forest = simulation.triangularForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
        forest.propagationTime(saveRoute,n,m, matrix)
            
    elif usrChoice == 3:
        n = 35    # Amount of values to consider for p
        m = 5      # Amount of trials per p        
        saveRoute = './graphs/percolationThreshold.png'
        
        forest = simulation.triangularForest(burningThreshold=0.55,occuProba=0.92 , initialForest=matrix)
        p_c = forest.percolationThreshold(saveRoute,n,m,matrix,True)
        print("The percolation threshold is: ",p_c)
        
    elif usrChoice == 4:
        n = 20      # Amount of values to consider for p in the range (0,1)to fin p_c
        m1 = 5      # Amount of trials per p        
        m2 = 30     # Amount of trials per p to find M
        saveRoute = './graphs/percolationThreshold.png'
        epsilon = 0.002*6
        delta = 0.002
        
        forest = simulation.squareForest(burningThreshold=0.55,occuProba=0.92 , initialForest=matrix)
        criticalExponent = forest.criticalExponent(saveRoute,epsilon,delta,n,m1,m2,matrix)
        print(criticalExponent)
            
    elif usrChoice == 5:
        # Create Voronoi diagram
        nPoints = 10000
        points = np.random.rand(nPoints, 2)
        vor = Voronoi(points)
        
        voronoi = voronoi_fire.voronoiFire(0.5,0.5,vor,1,)
        voronoi.animate('primera_prueba_voronoi')
        #voronoi.propagateFire()

    