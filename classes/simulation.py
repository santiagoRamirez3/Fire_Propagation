import numpy as np
import teselado

zeroArray = np.zeros(1)

class forestFire():
    def __init__(self, burningThreshold:float, initialForest:np.ndarray, neighbours:list, neighboursBoolTensor, wind:np.ndarray = zeroArray, topography:np.ndarray = zeroArray):
        
        self.burningThreshold = burningThreshold
        self.forest = np.copy(initialForest)
        self.neighbours = neighbours
        self.neighboursBoolTensor = neighboursBoolTensor
        self.wind = wind
        self.topography = topography

        self.forestSize = initialForest.shape
        self.historicalFirePropagation = [np.copy(initialForest)]
    
    def propagateFire(self):
        if np.sum(self.forest == 2) == 0:
            print('The forest does not have burning trees')
        else:
            thereIsFire = True
            while thereIsFire:
                burningTrees = self.forest == 2

                probabilityMatrixForest = np.random.rand(*self.forestSize)

                #=====================================================================================================
                # Here could appear a function to modificate probabilityMatrixForest depending of wind and topography
                #=====================================================================================================

                couldBurn = probabilityMatrixForest <= self.burningThreshold

                neighboursTensor = self.createNeighbourTensor()
                couldPropagate = np.logical_or.reduce(neighboursTensor == 2, axis=0)
                newBurningTrees = (self.forest == 1) & couldBurn & couldPropagate

                # implementar potencia de vecinos

                self.forest[burningTrees] = 3
                self.forest[newBurningTrees] = 2

                self.historicalFirePropagation.append(np.copy(self.forest))
                thereIsFire = False if np.sum(newBurningTrees) == 0 else True
            
            print('Fire finished')
    
    def createNeighbourTensor(self):
        neighborhoodSize = len(self.neighbours)
        tensor = np.zeros((neighborhoodSize, *self.forestSize))

        for i, neigh in enumerate(self.neighbours):
            x,y = neigh
            tensor[i] = np.roll(self.forest, (-x,y), axis=(1,0))
            #=============================================================================
            # Maybe some way could be [j+i: , :] where i don't know what i,j are ajjajaja
            #=============================================================================
            if x:
                if x == 1.:
                    tensor[i, : , -1 ] = 0
                if x == -1.:
                    tensor[i, : , 0 ] = 0
                else:
                    continue # Another condition and method for second neighbours and more
            if y:
                if y == 1.:
                    tensor[i, 0 , : ] = 0
                if y == -1.:
                    tensor[i, -1 , : ] = 0
                else:
                    continue # Another condition and method for second neighbours and more
            else:
                continue
        tensor = tensor * self.neighboursBoolTensor
        return tensor
    
class squareForest(forestFire):
    def __init__(self, burningThreshold:float, initialForest:np.ndarray, wind:np.ndarray = zeroArray, topography:np.ndarray = zeroArray):
        neighboursBoolTensor = np.ones((4,*initialForest.shape), dtype=bool)
        neighbours = [(-1,0),(1,0),(0,1),(0,-1)]
        super().__init__(burningThreshold, initialForest, neighbours, neighboursBoolTensor, wind, topography)
    
    def plot(self):
        return

class heaxgonalForest(forestFire):
    def __init__(self, burningThreshold:float, initialForest:np.ndarray, wind:np.ndarray = zeroArray, topography:np.ndarray = zeroArray):
        rows,columns = initialForest.shape
        neighboursBoolTensor = hexagonalNeighboursBooleanTensor(columns,rows)
        neighbours = [(-1,0),(1,0),(0,1),(0,-1)]
        super().__init__(burningThreshold, initialForest, neighbours, neighboursBoolTensor, wind, topography)
    
    def plot(self):
        return

class triangularForest(forestFire):
    def __init__(self, burningThreshold:float, initialForest:np.ndarray, wind:np.ndarray = zeroArray, topography:np.ndarray = zeroArray):
        neighboursBoolTensor = np.ones((6,*initialForest.shape), dtype=bool)
        neighbours = [(-1,0),(1,0),(0,1),(0,-1),(1,1),(-1,-1)]
        super().__init__(burningThreshold, initialForest, neighbours, neighboursBoolTensor, wind, topography)
    
    def plot(self):
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