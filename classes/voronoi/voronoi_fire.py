from classes.voronoi.voronoi_teselation import generateAnimation
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix


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
        self.status[initialFire] = 2
        
    
        # Create the neighbours table
        #self.neighboursTable = np.zeros((self.numPoints,self.numPoints), dtype=np.int8)
        #print(self.neighboursTable.shape)
        
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
                #N = np.dot(self.neighboursTable,mask)
                N = self.neighboursTable.dot(mask)

                # Get the modified Threshold for each tree
                newThreshold = 1-(1-self.T)**N
                
                # Generate aleatory number for each point
                probability = np.random.rand(self.numPoints)
                
                # find which trees could burn
                couldBurn = (probability < newThreshold)
                #print(propagationTime, np.sum(self.status==2),np.sum(couldBurn), np.sum(N>0))
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
