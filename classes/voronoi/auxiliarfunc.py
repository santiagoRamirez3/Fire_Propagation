import numpy as np

def applyOcupation(array:np.ndarray, occuProba:float):
    
    shape = array.shape
    modified_array = np.copy(array)
    occupationMask = (np.random.rand(*shape) > occuProba)
    modified_array[occupationMask] = 0

    return modified_array


def log_criteria_niter(x:float):
    n_iter = int(41* np.log(1+x) + 3)
    return n_iter