from classes import simulation, teselado
from menu import menu
import numpy as np

if __name__ == '__main__':
    #usrChoice = menu()
    matrix = np.ones((100,100))
    matrix[50,50] = 2
    forest = simulation.forestFire(0.55, matrix, [(1,0),(-1,0),(0,1),(0,-1)])
    forest.propagateFire()
    print(len(forest.historicalFirePropagation))
    for i, plot in enumerate(forest.historicalFirePropagation):
        teselado.plotting(plot,filename=str(i)+'.png')
    #plot = forest.historicalFirePropagation[-1]
    #i=2
    #teselado.plotting(plot,filename=str(i)+'.png')