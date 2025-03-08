from classes.regular import simulation
from classes.voronoi import voronoi_fire

from classes.fit import fitting

from scripts.menu import menu
import numpy as np
from scipy.spatial import Voronoi

from scripts.routes import routes_dict, data_route


if __name__ == '__main__':
    usrChoice = menu()
    matrix = np.ones((100,100))
    matrix[50,50] = 2
  
  
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
    if usrChoice == 1:
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2   Triangular\n3   Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')
            
            
        if tessellation == 1:
            
            name = 'squaredAnimation_test'
            route = routes_dict['squared'] +  name
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            forest.animate(route)    
            
            
        elif tessellation == 2:
            
            name = 'triangularAnimation'
            route = routes_dict['triangular'] + name
            forest = simulation.triangularForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            forest.animate(route)
        
        
        elif tessellation == 3:
            
            name = 'hexagonalAnimation'
            route = routes_dict['hexagon'] + name    
            forest = simulation.heaxgonalForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            forest.animate(route)
            
        elif tessellation == 4:
            # Create Voronoi diagram
            nPoints = 10000
            points = np.random.rand(nPoints, 2)
            vor = Voronoi(points)
            
            name = 'voronoiAnimation'
            route = routes_dict['voronoi'] + name

            voronoi = voronoi_fire.voronoiFire(1.,0.5,vor,1,)
            voronoi.animate(route)
            
        else:
            print('That is not an option, try again.')
            
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    elif usrChoice == 2:
        
        n = 100    # Amount of values to consider for p
        m = 20      # Amount of trials per p 
        
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2   Triangular\n3   Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')
            
        
        if tessellation == 1:
            
            name = 'SquaredFinalTimes'
            route = routes_dict['squared'] +  name
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            forest.propagationTime(route,n,m, matrix)
            
            
        elif tessellation == 2:
            
            name = 'TriangularFinalTimes'
            route = routes_dict['triangular'] + name 
            forest = simulation.triangularForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
            forest.propagationTime(route,n,m, matrix)
            
        elif tessellation == 3:
            
            name = 'hexagonFinalTimes'
            route = routes_dict['hexagon'] + name
            forest = simulation.heaxgonalForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
            forest.propagationTime(route,n,m, matrix)
        
        elif tessellation == 4:
            # Create Voronoi diagram
            nPoints = 10000
            points = np.random.rand(nPoints, 2)
            vor = Voronoi(points)
            
            name = 'voronoiFinalTimes'
            route = routes_dict['voronoi'] + name

            voronoi = voronoi_fire.voronoiFire(0.4,0.5,vor,1,)
            voronoi.propagationtime(saveName=route,n=80,m=100)
            
        else:
            print('That is not an option, try again.')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif usrChoice == 3:
        n = 35    # Amount of values to consider for p
        m = 5      # Amount of trials per p        
        
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2  Triangular\n3    Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')    
        
        if tessellation == 1:
            
            name = 'SquaredPercolationThreshold'
            route = routes_dict['squared'] +  name
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
            p_c = forest.percolationThreshold(route,n,m,matrix,True)
            print("The percolation threshold is: ",p_c)
            
            
        elif tessellation == 2:
            
            name = 'TriangularPercolationThreshold'
            route = routes_dict['triangular'] + name 
            forest = simulation.triangularForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
            p_c = forest.percolationThreshold(route,n,m,matrix,True)
            print("The percolation threshold is: ",p_c)
            
        elif tessellation == 3:
            
            name = 'hexagonalPercolationThreshold'
            route = routes_dict['hexagon'] + name
            forest = simulation.heaxgonalForest(burningThreshold=0.55, occuProba=1 ,initialForest=matrix)
            p_c = forest.percolationThreshold(route,n,m,matrix,True)
            print("The percolation threshold is: ",p_c)
        
        elif tessellation == 4:
            print('Not implemented just yet')
        
        else:
            print('That is not an option, try again.')
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif usrChoice == 4:
        n = 20      # Amount of values to consider for p in the range (0,1)to fin p_c
        m1 = 5      # Amount of trials per p        
        m2 = 30     # Amount of trials per p to find M
        saveRoute = './graphs/percolationThreshold.png'
        epsilon = 0.002*6
        delta = 0.002
        
        forest = simulation.squareForest(burningThreshold=0.55,occuProba=1. , initialForest=matrix)
        criticalExponent = forest.criticalExponent(saveRoute,epsilon,delta,n,m1,m2,matrix)
        print(criticalExponent)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    elif usrChoice == 5:
        n = 35    # Amount of values to consider for p
        m = 5      # Amount of trials per p       
        
        try:
            tessellation = int(input("Choose one: \n1   Squared\n2  Triangular\n3    Hexagonal\n4   Voronoi\n"))
        except:
            print('Not a valid option.')    
        
        if tessellation == 1:
            
            folder_path = data_route['squared']
            file_name = "datos_alta_resolucion.csv"
            name = 'squaredCompareProbabilities_alta_resolucion'
            imagePath = routes_dict['squared'] + name
            propTimeThreshold = 150
            
            forest = simulation.squareForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.compareBondSite(1000,imagePath,folder_path, file_name,matrix, 'squared', propTimeThreshold) 
            
            
        elif tessellation == 2:
            
            folder_path = data_route['triangular']
            file_name = "datos_alta_resolucion.csv"
            name = 'triangularCompareProbabilities_alta_resolucion'
            imagePath = routes_dict['triangular'] + name
            propTimeThreshold = 215
            
            forest = simulation.triangularForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.compareBondSite(1000,imagePath,folder_path, file_name,matrix, 'triangular',propTimeThreshold) 
            
        elif tessellation == 3:
            
            folder_path = data_route['hexagon']
            file_name = "datos_alta_resolucion.csv"
            name = 'hexagonalCompareProbabilities_alta_resolucion'
            imagePath = routes_dict['hexagon'] + name
            propTimeThreshold = 150
            
            forest = simulation.heaxgonalForest(burningThreshold=0.95,occuProba=0.95 ,initialForest=matrix)
            forest.compareBondSite(1000,imagePath,folder_path, file_name,matrix, 'hexagonal',propTimeThreshold) 
        
        elif tessellation == 4:
            nPoints = 10000
            points = np.random.rand(nPoints, 2)
            vor = Voronoi(points)
            folder_path = data_route['voronoi']
            file_name = "datos_alta_resolucion.csv"
            propTimeThreshold = 150
            

            name = 'voronoiCompareProbabilities_alta_resolucion'
            imagePath = routes_dict['voronoi'] + name
            forest = voronoi_fire.voronoiFire(burningThreshold=0.95, occuProba=0.95, voronoi=vor, initialFire=1)
            forest.compareBondSite(1000,imagePath,folder_path, file_name, propTimeThreshold) 
        
        else:
            print('That is not an option, try again.')
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    elif usrChoice == 6:
        for tessellation in range(1,5):
            for i,probability in enumerate(np.arange(0.1,1.1,0.1)):
                print(f'Tessellation: {tessellation}, probability: {probability}')

                if tessellation == 1:
                    matrix = np.ones((100,100))
                    matrix[50,50] = 2

                    name = 'squaredAnimation' + '_' + str(i)
                    route = routes_dict['squared'] +  name
                    forest = simulation.squareForest(burningThreshold=probability, occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
                    forest.animate(route)    


                elif tessellation == 2:
                    matrix = np.ones((100,100))
                    matrix[50,50] = 2

                    name = 'triangularAnimation' + '_' + str(i)
                    route = routes_dict['triangular'] + name
                    forest = simulation.triangularForest(burningThreshold=probability, occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
                    forest.animate(route)


                elif tessellation == 3:
                    matrix = np.ones((100,100))
                    matrix[50,50] = 2

                    name = 'hexagonalAnimation' + '_' + str(i)
                    route = routes_dict['hexagon'] + name    
                    forest = simulation.heaxgonalForest(burningThreshold=probability, occuProba=0.95 ,initialForest=matrix, saveHistoricalPropagation=True)
                    forest.animate(route)

                elif tessellation == 4:
                    # Create Voronoi diagram
                    nPoints = 10000
                    points = np.random.rand(nPoints, 2)
                    vor = Voronoi(points)

                    name = 'voronoiAnimation' + '_' + str(i)
                    route = routes_dict['voronoi'] + name

                    voronoi = voronoi_fire.voronoiFire(burningThreshold=probability, occuProba=0.95, voronoi=vor, initialFire=1)
                    voronoi.animate(route)
    
    elif usrChoice == 7:
        saveRoute = routes_dict['voronoi']
        fitting.expFit(dataRoute='data/voronoi/datos.csv',propTimeThreshold=130,saveRoute=saveRoute)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    