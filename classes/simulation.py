import numpy as np
from classes import teselado
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.stats import linregress


# ===============================================================
def percolation_check(array):
    # Identificar regiones conectadas de árboles quemados, representados por el número 3
    labeled_array, num_features = label(array == 3)
    
    # Obtener las etiquetas presentes en los bordes superior e inferior (percolación vertical)
    first_row_labels = set(labeled_array[0, :])
    last_row_labels = set(labeled_array[-1, :])
    
    # Verificar si alguna etiqueta de la primera fila está en la última fila (percolación vertical)
    vertical_common_labels = first_row_labels.intersection(last_row_labels)
    
    # Obtener las etiquetas presentes en los bordes izquierdo y derecho (percolación horizontal)
    first_col_labels = set(labeled_array[:, 0])
    last_col_labels = set(labeled_array[:, -1])
    
    # Verificar si alguna etiqueta de la primera columna está en la última columna (percolación horizontal)
    horizontal_common_labels = first_col_labels.intersection(last_col_labels)
    
    # Si hay etiquetas en común en cualquiera de las direcciones, hay percolación
    return bool(vertical_common_labels - {0}) or bool(horizontal_common_labels - {0})  # Excluir 0 porque no es una región etiquetada

# ===============================================================



zeroArray = np.zeros(1)

class forestFire():
    
    def __init__(self, 
                 burningThreshold:float, initialForest:np.ndarray,
                 neighbours:list,
                 neighboursBoolTensor: np.ndarray,
                 wind:np.ndarray = zeroArray,
                 topography:np.ndarray = zeroArray,
                 saveHistoricalPropagation:bool = False,
                 ) -> None:
        
        
        self.burningThreshold = burningThreshold
        self.forest = np.copy(initialForest)
        self.neighbours = neighbours
        self.neighboursBoolTensor = neighboursBoolTensor
        self.wind = wind
        self.topography = topography

        self.forestSize = initialForest.shape
        self.historicalFirePropagation = [np.copy(initialForest)]
        self.saveHistoricalPropagation = saveHistoricalPropagation
        
    
    def propagateFire(self,p:float):
        if np.sum(self.forest == 2) == 0:
            print('The forest does not have burning trees')
        else:
            
            thereIsFire = True
            propagationTime = 0
            while thereIsFire:
                # Record time propagation in terms of iterations
                propagationTime += 1
                
                burningTrees = (self.forest == 2)

                probabilityMatrixForest = np.random.rand(*self.forestSize)

                #=====================================================================================================
                # Here could appear a function to modificate probabilityMatrixForest depending of wind and topography
                #=====================================================================================================

                #couldBurn = (probabilityMatrixForest <= self.burningThreshold)
                couldBurn = (probabilityMatrixForest <= p)

                neighboursTensor = self.createNeighbourTensor()
                couldPropagate = np.logical_or.reduce(neighboursTensor == 2, axis=0)
                newBurningTrees = (self.forest == 1) & couldBurn & couldPropagate

                # implementar potencia de vecinos

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
                if x == -1.:
                    tensor[i, : , 0 ] = 0
                else:
                    continue # Maybe Another condition and method for second neighbours and more
            if y:
                if y == 1.:
                    tensor[i, 0 , : ] = 0
                if y == -1.:
                    tensor[i, -1 , : ] = 0
                else:
                    continue # Maybe Another condition and method for second neighbours and more
            else:
                continue
        tensor = tensor * self.neighboursBoolTensor
        return tensor
    
    
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
        plt.title('Burning time as a function of p\nErrorbar = 1$\sigma$')
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
    
    def criticalExponent(self, saveRoute:str,epsilon:float,delta:float, n:int, m1:int,m2:int, initial:np.ndarray):
        self.forest = np.copy(initial)
        #p_c = self.percolationThreshold(saveRoute, n,m1,self.forest)
        p_c=0.50
        
        # POssible p values to consider around p_c
        P = np.arange( p_c, p_c + epsilon, delta)
        #P = np.arange(p_c - epsilon, p_c, delta)
        t = np.abs(P-p_c)
        print(t)
        print(np.log(t))
        #print(t)
        
        # Registered Percolating cluster size
        meanM = np.zeros(len(t))
        
        # Simulate and calculate the mean percolation cluster size
        for i,p in enumerate(P):
            
            M = np.zeros(m2)
            for j in range(m2):
                self.forest = np.copy(initial)
                self.propagateFire(p)

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

        print(log_meanM)
        return B
        
    
#=============================================================================================================================================
class squareForest(forestFire):
    """
    This is a subclass of the general class forestFire, but specifily designed for simulate
    the propagation in a central square distribution of trees
    """
    def __init__(self,
                 burningThreshold:float,
                 initialForest:np.ndarray,
                 wind:np.ndarray = zeroArray,
                 topography:np.ndarray = zeroArray,
                 saveHistoricalPropagation:bool = False):
        
        neighboursBoolTensor = np.ones((4,*initialForest.shape), dtype=bool)
        neighbours = [(-1,0),(1,0),(0,1),(0,-1)]
        super().__init__(burningThreshold, initialForest, neighbours, neighboursBoolTensor, wind, topography, saveHistoricalPropagation)
    
    def animate(self, fileName, interval=100):
        
        if (self.saveHistoricalPropagation):
            
            print('Generating and saving animation, wait a sec...')
            # Simulate fire
            _ = self.propagateFire(self.burningThreshold)
            teselado.squareAnimationPlot(fileName, self.historicalFirePropagation, interval)
            print('Done.')
        else:
            print('Historical data not found.')
        
        return
    
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
                 initialForest:np.ndarray,
                 wind:np.ndarray = zeroArray,
                 topography:np.ndarray = zeroArray,
                 saveHistoricalPropagation:bool = False):
        
        neighboursBoolTensor = np.ones((6,*initialForest.shape), dtype=bool)
        neighbours = [(-1,0),(1,0),(0,1),(0,-1),(1,1),(-1,-1)]
        super().__init__(burningThreshold, initialForest, neighbours, neighboursBoolTensor, wind, topography, saveHistoricalPropagation)
    
    def animate(self, fileName, interval=100):
        teselado.hexagonalAnimationPlot(filename=fileName,
                                        historical= self.historicalFirePropagation,
                                        interval=interval,
                                        size=self.forestSize)
        return
    
    def plot(self, fileName):
        return
    
    
#=============================================================================================================================================
class triangularForest(forestFire):
    """
    This is a subclass of the general class forestFire, but specifily designed for simulate
    the propagation in a central triangular distribution of trees
    """
    def __init__(self, burningThreshold:float, initialForest:np.ndarray, wind:np.ndarray = zeroArray, topography:np.ndarray = zeroArray):
        rows,columns = initialForest.shape
        neighboursBoolTensor = hexagonalNeighboursBooleanTensor(columns,rows)
        neighbours = [(-1,0),(1,0),(0,1),(0,-1)]
        super().__init__(burningThreshold, initialForest, neighbours, neighboursBoolTensor, wind, topography)
    
    def animate(self, fileName, interval=100):
        return
    
    def plot(self, fileName):
        return

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
    booleanTensor = np.ones((4,rows,columns), dtype=bool)

    roll_0 = np.zeros(columns, dtype=bool)
    roll_1 = np.zeros(columns, dtype=bool)
    rightNeighbours = np.zeros((rows,columns), dtype=bool)

    for i in range(1,columns,2):
        roll_1[i] = True
        roll_0[i-1] = True

    for j in range(rows):
        rightNeighbours[j] = roll_0 if j%2 == 0 else roll_1

    booleanTensor[0] = ~rightNeighbours
    booleanTensor[1] = rightNeighbours
    return booleanTensor