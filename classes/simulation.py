import numpy as np
from classes import teselado
import matplotlib.pyplot as plt
from scipy.stats import linregress

from classes.auxiliarfunc import percolation_check, Apply_occupation_proba

zeroArray = np.zeros(1)

class forestFire():
    
    def __init__(self, 
                 burningThreshold:float, occuProba:float,initialForest:np.ndarray,
                 neighbours:list,
                 neighboursBoolTensor: np.ndarray,
                 wind:np.ndarray = zeroArray,
                 topography:np.ndarray = zeroArray,
                 saveHistoricalPropagation:bool = False,
                 ) -> None:
        
        
        self.burningThreshold = burningThreshold
        self.occuProba = occuProba
        self.forest = np.copy(initialForest)
        self.neighbours = neighbours
        self.neighboursBoolTensor = neighboursBoolTensor
        self.wind = wind
        self.topography = topography

        self.forestSize = initialForest.shape
        self.historicalFirePropagation = [np.copy(initialForest)]
        self.saveHistoricalPropagation = saveHistoricalPropagation
        
    
    def propagateFire(self,p:float):
        # Apply ocuppation probability
        self.forest = Apply_occupation_proba(self.forest,self.occuProba)
        
        if np.sum(self.forest == 2) == 0:
            print('The forest does not have burning trees')
        else:
            
            thereIsFire = True
            propagationTime = 0
            while thereIsFire:
                # Record time propagation in terms of iterations
                propagationTime += 1
                
                neighboursTensor = self.createNeighbourTensor()
                couldPropagate = neighboursTensor == 2
                amoungOfBurningNeighbours = np.sum(couldPropagate, axis=0)
                cellsToEvaluate = amoungOfBurningNeighbours > 0
                burningTrees = (self.forest == 2)

                probabilityMatrixForest = np.random.rand(*self.forestSize)
                probabilityMatrixForest[cellsToEvaluate] = 1. - (1. - probabilityMatrixForest[cellsToEvaluate]) ** (1/amoungOfBurningNeighbours[cellsToEvaluate])

                #-----------------------------------------------------------------------------------------------------
                # Here could appear a function to modificate probabilityMatrixForest depending of wind and topography
                #-----------------------------------------------------------------------------------------------------

                couldBurn = (probabilityMatrixForest <= p)

                newBurningTrees = (self.forest == 1) & couldBurn & np.logical_or.reduce(couldPropagate,axis=0)

                # Update forest matrix for the next step
                self.forest[burningTrees] = 3
                self.forest[newBurningTrees] = 2

                # Only save historical data to plot if required
                if (self.saveHistoricalPropagation):
                    self.historicalFirePropagation.append(np.copy(self.forest))
                
        
                thereIsFire = False if np.sum(newBurningTrees) == 0 else True
            
            #print('Fire extinguished')
            return propagationTime
        
        
    def createNeighbourTensor(self):
        neighborhoodSize = len(self.neighbours)
        tensor = np.zeros((neighborhoodSize, *self.forestSize))

        for i, neigh in enumerate(self.neighbours):
            x,y = neigh
            tensor[i] = np.roll(self.forest, (-x,y), axis=(1,0))
            if x:
                if x == 1.:
                    tensor[i, : , -1 ] = 0
                elif x == -1.:
                    tensor[i, : , 0 ] = 0
                else:
                    continue # Maybe Another condition and method for second neighbours and more
            if y:
                if y == 1.:
                    tensor[i, 0 , : ] = 0
                elif y == -1.:
                    tensor[i, -1 , : ] = 0
                else:
                    continue # Maybe Another condition and method for second neighbours and more
            else:
                continue
        tensor = tensor * self.neighboursBoolTensor
        return tensor

    #-------------------------------------------------------------------------------------------------
    # Methods to calculate statistic params of many simulations
    #-------------------------------------------------------------------------------------------------

    def propagationTime(self,saveRoute:str, n:int,m:int, matrix:np.ndarray):
        '''
         args: 
         - n: amount of values for p to consider in the interval 0 to 1
         - m: amount ot tials for each p
         - matrix: Initial fire matrix
        '''
        
        # Define the array to store the final propagation time
        # Each row is a fixed p, the columns are trials
        finalTimes = np.zeros((n,m))
        meanFinaltimes = np.zeros(n)
        meanFinaltimesStd = np.zeros(n)
        P = np.linspace(0,1,n)
        
    
        for i,p in enumerate(P):
            #self.burningThreshold = p
            for j in range(m):
                self.forest = np.copy(matrix)
                finalTimes[i,j] = self.propagateFire(p)
            
            meanFinaltimes[i] = np.mean(finalTimes[i,:])
            meanFinaltimesStd[i] = np.std(finalTimes[i,:])
            
            
        plt.errorbar(P, meanFinaltimes, yerr=meanFinaltimesStd, capsize=5, ecolor='red', marker='o', linestyle='None')
        plt.xlabel('$P$')
        plt.ylabel('$t(p)$')
        plt.title(r'Burning time as a function of p\nErrorbar = 1$\sigma$')
        plt.savefig(saveRoute)
        
    def percolationThreshold(self,saveRoute:str,n:int,m:int, matrix:np.ndarray, plot:bool=False):
        '''
         args: 
         - n: amount of values for p to consider in the interval 0 to 1
         - m: amount ot tials for each p
         - matrix: Initial fire matrix
        '''
        percolationResults = np.zeros((n,m))
        P = np.linspace(0,1,n)
        
        for i,p in enumerate(P):
            for j in range(m):
                self.forest = np.copy(matrix)
                _ = self.propagateFire(p)
                percolationResults[i,j] = percolation_check(self.forest)
                
        # Delta of p
        delta = np.round(1/n,2)
        
        # Calculate the frequency of percolation for each p
        percolation_frequencies = percolationResults.mean(axis=1)
        
        # Get the percolation threshold
        p_c = np.round(P[percolation_frequencies > 0.5][0],2)

        if plot:
            # Plot
            plt.plot(P, percolation_frequencies, marker='o')
            plt.xlabel("$P$")
            plt.ylabel("Percolation Frequency")
            plt.title("Percolation Probability vs. p")
            plt.grid()
            plt.text(0.63, 1.15, f'Percolation threshold: {p_c} +- {delta}', fontsize=10, color="blue")
            plt.savefig(saveRoute)
        
        return p_c
    #-------------------------------------------------------------------------------------------------
    def criticalExponent(self, saveRoute:str,epsilon:float,delta:float, n:int, m1:int,m2:int, initial:np.ndarray):
        self.forest = np.copy(initial)
        #p_c = self.percolationThreshold(saveRoute, n,m1,self.forest)
        p_c=0.9
        
        # Possible p values to consider around p_c
        P = np.arange( p_c - epsilon, p_c, delta)
        #P = np.arange(p_c - epsilon, p_c, delta)
        t = np.abs(P-p_c)
        #print(t)
        #print(np.log(t))
        #print(t)
        
        # Registered Percolating cluster size
        meanM = np.zeros(len(t))
        
        # Simulate and calculate the mean percolation cluster size
        for i,p in enumerate(P):
            
            M = np.zeros(m2)
            #self.occuProba = p
            for j in range(m2):
                self.forest = np.copy(initial)
                self.propagateFire(self.burningThreshold)

                # Given the finished board, calculate the size of percolating cluster

                # Check if it percolated
                if percolation_check(self.forest):

                    # Calculate the size of the cluster (cells with zeros)
                    M[j] = np.sum(self.forest == 3)
                    
            # Take the mean
            if np.sum(M) == 0: meanM[i] = 0
            else: meanM[i] = np.mean(M[ M>0 ] )
        
        # Now we can find the critical exponent
        
        # Convert to logarithm scale
        mask = (meanM != 0) & (t != 0)
        log_t = np.log(t[mask])
        log_meanM = np.log(meanM[mask])
        
        # Make the linear regression
        slope, intercept, r_value, p_value, std_err = linregress(log_t, log_meanM)
        
        # Get the critical exponent B
        B = -slope
        #
        #x = np.arange(0,1,0.01)
        plt.plot(log_t,log_meanM)
        #plt.ylim(0,20)
        plt.savefig('./prueba')
        #print(log_t)
        
        #print(meanM)

        #print(p_c)

        #print(log_meanM)
        return B
        
    
#=============================================================================================================================================
class squareForest(forestFire):
    """
    This is a subclass of the general class forestFire, but specifily designed for simulate
    the propagation in a central square distribution of trees
    """
    def __init__(self,
                 burningThreshold:float,
                 occuProba:float,
                 initialForest:np.ndarray,
                 wind:np.ndarray = zeroArray,
                 topography:np.ndarray = zeroArray,
                 saveHistoricalPropagation:bool = False):
        
        neighboursBoolTensor = np.ones((4,*initialForest.shape), dtype=bool)
        neighbours = [(-1,0),(1,0),(0,1),(0,-1)]
        super().__init__(burningThreshold, occuProba, initialForest, neighbours, neighboursBoolTensor, wind, topography, saveHistoricalPropagation)
    
    def animate(self, fileName, interval=100):
        
        if (self.saveHistoricalPropagation):
            
            print('Starting simulation, wait a sec...')
            # Simulate fire
            _ = self.propagateFire(self.burningThreshold)

            print('Simulation has finished. Initializing animation...')
            teselado.squareAnimationPlot(fileName,
                                         self.historicalFirePropagation,
                                         interval)
            print('Done.')
        else:
            print('Historical data not found.')
    
    def plot(self, fileName):
        return
    
    
#=============================================================================================================================================
class heaxgonalForest(forestFire):
    """
    This is a subclass of the general class forestFire, but specifily designed for simulate
    the propagation in a central hexagonal distribution of trees
    """
    def __init__(self,
                 burningThreshold:float,
                 occuProba:float,
                 initialForest:np.ndarray,
                 wind:np.ndarray = zeroArray,
                 topography:np.ndarray = zeroArray,
                 saveHistoricalPropagation:bool = False):
        
        
        rows,columns = initialForest.shape
        neighboursBoolTensor = hexagonalNeighboursBooleanTensor(columns,rows)
        neighbours = [(0,1),(0,-1),(-1,0),(1,0),(-1,1),(-1,1),(-1,-1),(-1,-1)]
        super().__init__(burningThreshold, occuProba,initialForest, neighbours, neighboursBoolTensor, wind, topography, saveHistoricalPropagation)
    
    def animate(self, fileName, interval=100):
        if (self.saveHistoricalPropagation):
            
            print('Starting simulation, wait a sec...')
            # Simulate fire
            _ = self.propagateFire(self.burningThreshold)
            
            print('Simulation has finished. Initializing animation...')
            teselado.hexagonalAnimationPlot(filename=fileName,
                                            historical= self.historicalFirePropagation,
                                            interval=interval,
                                            size=self.forestSize)
            print('Done.')
        else:
            print('Historical data not found.')
    
#=============================================================================================================================================
class triangularForest(forestFire):
    """
    This is a subclass of the general class forestFire, but specifily designed for simulate
    the propagation in a central triangular distribution of trees
    """
    def __init__(self, burningThreshold:float,occuProba:float, initialForest:np.ndarray, wind:np.ndarray = zeroArray, topography:np.ndarray = zeroArray, saveHistoricalPropagation:bool = False):
        rows,columns = initialForest.shape
        neighboursBoolTensor = triangularNeighboursBooleanTensor(columns,rows)
        neighbours = [(0,1),(0,-1),(1,0),(-1,0)]
        super().__init__(burningThreshold, occuProba,initialForest, neighbours, neighboursBoolTensor, wind, topography, saveHistoricalPropagation)
    
    def animate(self, fileName, interval=100):
        if (self.saveHistoricalPropagation):
            
            print('Starting simulation, wait a sec...')
            # Simulate fire
            _ = self.propagateFire(self.burningThreshold)
            
            print('Simulation has finished. Initializing animation...')
            teselado.triangularAnimationPlot(filename=fileName,
                                            historical= self.historicalFirePropagation,
                                            interval=interval,
                                            size=self.forestSize)
            print('Done.')
        else:
            print('Historical data not found.')

#=======================================================================================
# Maybe more forest types, boronoy, etc
#=======================================================================================

#---------------------------------------------------------------------------------------
# Auxiliar functions to change probability depending rounding conditions
#---------------------------------------------------------------------------------------
def windContribution(x):
    return x

def topographyContribution(x):
    return x

def hexagonalNeighboursBooleanTensor(columns,rows):
    """
    This function compute the boolean neighbours tensor for an hexagonal forest
    of size (y,x)
    """
    booleanTensor = np.ones((8,rows,columns), dtype=bool)

    evenColumns = np.zeros((rows,columns), dtype=bool)
    evenColumns[:, ::2] = True

    oddColumns = np.zeros((rows,columns), dtype=bool)
    oddColumns[:, 1::2] = True

    booleanTensor[4] = booleanTensor[5] = evenColumns
    booleanTensor[6] = booleanTensor[7] = oddColumns
    return booleanTensor

def triangularNeighboursBooleanTensor(columns,rows):
    """
    This function compute the boolean neighbours tensor for an hexagonal forest
    of size (y,x)
    """
    booleanTensor = np.ones((4,rows,columns), dtype=bool)

    evenColumns = np.zeros((rows,columns), dtype=bool)
    evenColumns[:, ::2] = True
    for i in range(1, rows, 2):  # Fila impar, comenzando desde 1
        evenColumns[i] = np.roll(evenColumns[i], shift=1)

    oddColumns = ~evenColumns

    booleanTensor[2] = evenColumns
    booleanTensor[3] = oddColumns
    return booleanTensor