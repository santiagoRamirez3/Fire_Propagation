from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

def squarePlot(historicalFire:list):
    return

def hexagonalPlot(historicalFire:list):
    return

def triangularPlot(historicalFire:list):
    return




def plotting(matrix, filename="matrix_plot.png"):
    """
    Grafica una matriz 2D y la guarda en un archivo .png.
    
    :param matrix: Matriz 2D (numpy array) a graficar
    :param filename: Nombre del archivo de salida (.png)
    """
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.colorbar()  # Añade una barra de colores
    plt.title('Visualización de la matriz 2D')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    
    # Guardar la gráfica en un archivo .png
    plt.savefig('media/' + filename, format='png')
    
    # Limpiar la figura para evitar superposiciones en futuras gráficas
    plt.clf()

    