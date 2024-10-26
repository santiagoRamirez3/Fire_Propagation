from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

colors = ['white', 'green', 'red', 'black']
ticksLabels = ['No tree', 'Healthy tree', 'Burning tree', 'Burned tree']
customCmap = ListedColormap(colors)
ticksLocation = [0.375, 1.125, 1.875, 2.625]

#=============================================================================================================================================

def squareAnimationPlot(filename:str, historical:list, interval:int) -> None:

    fig, ax = plt.subplots()
    cax = ax.matshow(historical[0], cmap=customCmap, vmin=0, vmax=3)
    cbar = plt.colorbar(cax, ticks=ticksLocation)
    cbar.set_ticklabels(ticksLabels)

    # Función de actualización de la animación
    def update(i):
        cax.set_array(historical[i])  # Actualizar la matriz mostrada
        return [cax]

    # Configuración de la animación
    squareAni = animation.FuncAnimation(
        fig, update, frames=len(historical), interval=interval, blit=True
    )
    # Mostrar la animación
    squareAni.save(filename + ".gif", writer="pillow")
    return

#=============================================================================================================================================

def hexagonalAnimationPlot(filename:str, historical:list, interval:int, size:tuple) -> None:
    m,n = size
    hexagons = generateHexagonalGrid(xmin=0, xmax=n, ymin=0, ymax=m, hex_size=2/3)

    hexagonsColors = colorAssigner(historical[0])
    hex_collection = PolyCollection(hexagons, edgecolors='black', facecolors=hexagonsColors,linewidth=0.2)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_collection(hex_collection)
    ax.set_xlim(0, n)
    ax.set_ylim(0, m)
    ax.set_aspect('equal')

    def update_colors(frame):
        # Generar colores aleatorios para cada fotograma
        hexagonsColors = colorAssigner(historical[frame])
        hex_collection.set_facecolors(hexagonsColors)
    return [hex_collection]

    # Crear la animación
    hexagonalAni = animation.FuncAnimation(fig, update_colors, frames=len(historical), interval=interval, blit=True)
    hexagonalAni.save(filename + ".gif", writer="pillow")
    return

#-------------------------------------------------------------------------------------------------------------------

def hexagonVertices(x_center, y_center, size):
    """Generate the vertices of a hexagon centered at (x_center, y_center) with the given size."""
    angles = np.linspace(0, 2 * np.pi, 7)  # 7 points to complete the hexagon (including the first vertex twice)
    return [(x_center + size * np.cos(angle), y_center + size * np.sin(angle)) for angle in angles]

#-------------------------------------------------------------------------------------------------------------------
def generateHexagonalGrid(xmin, xmax, ymin, ymax, hex_size):
    """
    Generate a grid of hexagons covering the region (xmin, xmax) x (ymin, ymax).
    hex_size is the distance from the center to any vertex of the hexagon.
    """
    hexagons = []
    dx = 3/2 * hex_size  # Horizontal distance between hexagon centers
    dy = np.sqrt(3) * hex_size  # Vertical distance between hexagon centers

    # Loop through grid positions
    y = ymax
    while y >= ymin:
        x = xmin
        while x < xmax:
            hexagons.append(hexagonVertices(x, y, hex_size))
            y += dy
        x += dx
        ymin_offset = dy / 2 if (x-xmin) % (2 * dx) == 0 else -dy / 2  # Offset every other column
        ymin = ymin - ymin_offset

    return hexagons

#-------------------------------------------------------------------------------------------------------------------

def colorAssigner(matrix, colors=colors):
    flatMatrix = matrix.reshape(1)

    for i in [0,1,2,3]:
        flatMatrix[flatMatrix == i] = colors[i]
    return flatMatrix

#=============================================================================================================================================

def triangularAnimationPlot(filename:str, historical:list, interval:int) -> None:
    return

    