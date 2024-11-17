from classes.voronoi.voronoi_teselation import generateAnimation
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from shapely.geometry import Polygon


class voronoiFire():
    def __init__(self,
                 burningThreshold:float, occuProba:float, voronoi:object, initialFire:int,saveHistoricalPropagation:bool = False) -> None:
        
        # Extract the object attributes fron the arguments
        self.T = burningThreshold
        self.occuProba = occuProba
        self.voronoi = voronoi
        self.initial = initialFire
        
        # Extract useful information
        self.neighbours = voronoi.ridge_points
        self.numPoints = self.voronoi.points.shape[0]
        
        # Set the initial fire status
        self.status = np.ones(self.numPoints)
        self.createBorder()
        self.status[initialFire] = 2
        
    
        # Create the neighbours table
        
        self.neighboursTable = dok_matrix((self.numPoints,self.numPoints))
        #dok = self.neighboursTable.todok()
        
        for i,j in self.neighbours:
            self.neighboursTable[i,j] = 1
            self.neighboursTable[j,i] = 1
            
        self.neighboursTable = self.neighboursTable.tocsr()
        
        # Space to save historical fire status
        self.historicalFirePropagation = [np.copy(self.status)]
        self.saveHistoricalPropagation = saveHistoricalPropagation
    
    def propagateFire(self):
        
        if np.sum(self.status == 2) == 0:
            print('The forest does not have burning trees')
            
        else:
            thereIsFire = True
            propagationTime = 0
            
            while thereIsFire:
                propagationTime += 1
                mask = (self.status == 2).astype(int)
                
                # Matrix that contains the amount of burning neighbours each tree has
                
                N = self.neighboursTable.dot(mask)

                # Get the modified Threshold for each tree
                newThreshold = 1-(1-self.T)**N
                
                # Generate aleatory number for each point
                probability = np.random.rand(self.numPoints)
                
                # find which trees could burn
                couldBurn = (probability < newThreshold)

                # Find those trees that will brun in the next step
                newBurningTrees = (self.status == 1) & couldBurn & (N>0)
                
                # State burned trees
                self.status[self.status == 2] = 3
                
                # Set new burning trees
                self.status[newBurningTrees] = 2
                
                if (self.saveHistoricalPropagation):
                    self.historicalFirePropagation.append(np.copy(self.status))
                
                
                thereIsFire = False if np.sum(newBurningTrees) == 0 else True
            
            return propagationTime
        
    def animate(self,filename, interval = 100):
        self.saveHistoricalPropagation = True
        print('Starting simulation, wait a sec...')
        # Simulate fire
        _ = self.propagateFire()

        print('Simulation has finished. Initializing animation...')
        generateAnimation(self.voronoi,
                          filename,
                          self.historicalFirePropagation,
                          interval)
    
    def createBorder(self):
        max_length = 10./np.sqrt(self.numPoints)
        for i in range(self.numPoints):
            region_index = self.voronoi.point_region[i]  # Get index for the i point's region
            region = self.voronoi.regions[region_index]  # Get region by index
            
            # if region is infinite, set status 0
            if -1 in region:
                self.status[i] = 0
                continue
            
            # if region is finite, calculate length\perimeter
            polygon = Polygon(self.voronoi.vertices[region])  # create region's polygon
            perimeter = polygon.length  # perimeter of polygon
            
            # if perimeter is higher than max_length, asign status 0
            if perimeter > max_length:
                self.status[i] = 0
