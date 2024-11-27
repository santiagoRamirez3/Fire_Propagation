import numpy as np

def applyOcupation(array:np.ndarray, occuProba:float):
    
    shape = array.shape
    modified_array = np.copy(array)
    occupationMask = (np.random.rand(*shape) > occuProba)
    modified_array[occupationMask] = 0

    return modified_array