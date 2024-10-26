from classes import simulation
from menu import menu
import numpy as np

if __name__ == '__main__':
    #usrChoice = menu()
    matrix = np.ones((100,100))
    matrix[50,50] = 2
    forest = simulation.squareForest(0.55, matrix)
    forest.propagateFire()
    forest.animate('intento_1')