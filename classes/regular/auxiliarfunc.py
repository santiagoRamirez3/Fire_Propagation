import numpy as np
from scipy.ndimage import label


import matplotlib.pyplot as plt



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

def Apply_occupation_proba(array:np.ndarray, occuProba:float):
    
    shape = array.shape
    modified_array = np.copy(array)
    occupationMask = (np.random.rand(*shape) > occuProba)
    modified_array[occupationMask] = 0
    modified_array[ array.shape[0]//2 , array.shape[1]//2] = 2
    #print(modified_array[45:55,45:55])

    return modified_array
# ===============================================================