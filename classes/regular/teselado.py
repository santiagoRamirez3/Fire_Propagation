from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

colors = ['white', 'green', 'red', 'black']
ticksLabels = ['No tree', 'Healthy tree', 'Burning tree', 'Burned tree']
customCmap = ListedColormap(colors)
ticksLocation = [0.375, 1.125, 1.875, 2.625]

def colorAssigner(matrix, colors=colors):
    flatMatrix = matrix.flatten()
    colorArray = np.empty(len(flatMatrix), dtype='U7')

    for i in [0,1,2,3]:
        colorArray[flatMatrix == i] = colors[i]
    return colorArray

#=============================================================================================================================================

def squareAnimationPlot(filename:str, historical:list, interval:int, p_bond, p_site) -> None:

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(historical[0], cmap=customCmap, vmin=0, vmax=3)
    ax.set_title('Square tessellation simulation', size=20)
    ax.set_xlabel(r'$P_{bond}=$' + str(round(p_bond,2)) + r'  $P_{site}=$' + str(round(p_site,2)), size=15)
    cbar = plt.colorbar(cax, ticks=ticksLocation)
    cbar.set_ticklabels(ticksLabels)
    cbar.set_label('Tree status')

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

#=============================================================================================================================================

def hexagonalAnimationPlot(filename:str, historical:list, interval:int, size:tuple, p_bond, p_site) -> None:
    m,n = size
    hexagons = generateHexagonalGrid(xmin=0, xmax=n, ymin=0, ymax=m, hex_size=2/3)
  
    hexagonsColors = colorAssigner(historical[0])
    hex_collection = PolyCollection(hexagons, edgecolors='black', facecolors=hexagonsColors,linewidth=0.1, cmap=customCmap)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Hexagonal tessellation simulation', size=20)
    ax.set_xlabel(r'$P_{bond}=$' + str(round(p_bond,2)) + r'  $P_{site}=$' + str(round(p_site,2)), size=15)
    ax.add_collection(hex_collection)
    ax.set_xlim(0, n-1)
    ax.set_ylim(1, m+(1/np.sqrt(3)))
    ax.set_aspect('equal')

    # Normalizar los valores (0 a 3 para los colores)
    norm = mcolors.BoundaryNorm(boundaries=[0,0.75,1.5,2.25,3], ncolors=len(colors))

    # Agregar la barra de color
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=customCmap), ax=ax)
    cbar.set_label('Tree status')  # Etiqueta para la barra de color
    cbar.set_ticks(ticksLocation)  # Ubicación de los ticks
    cbar.set_ticklabels(ticksLabels)  # Etiquetas de los ticks
    #cbar = plt.colorbar(hex_collection, ticks=ticksLocation)
    #cbar.set_ticklabels(ticksLabels)

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
    ymax *= (hex_size*np.sqrt(3))
    hexagons = []
    dx = 3/2 * hex_size  # Horizontal distance between hexagon centers
    dy = np.sqrt(3) * hex_size  # Vertical distance between hexagon centers

    # Loop through grid positions
    y = ymax
    row = 1
    while y > ymin:
        x = xmin
        while x < xmax:
            y_offset = (dy / 2) if (x-xmin) % (2 * dx) == 0 else (-dy / 2)  # Offset every other column
            y += y_offset
            hexagons.append(hexagonVertices(x, y, hex_size))
            x += dx
        y = ymax - row*dy
        row += 1

    return hexagons

#=============================================================================================================================================

def triangularAnimationPlot(filename:str, historical:list, interval:int, size:tuple, p_bond, p_site) -> None:
    m,n = size
    triangules = generateTriangularGrid(n,m)
  
    triangulesColors = colorAssigner(historical[0])
    triangule_collection = PolyCollection(triangules, edgecolors='black', facecolors=triangulesColors,linewidth=0.1, cmap=customCmap)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title('Triangular  tessellation simulation', size=20)
    ax.set_xlabel(r'$P_{bond}=$' + str(round(p_bond,2)) + r'  $P_{site}=$' + str(round(p_site,2)), size=15)
    ax.add_collection(triangule_collection)
    ax.set_xlim(0, m * np.sqrt(3))
    ax.set_ylim(0, n - 1)
    ax.set_aspect('equal')

    # Normalizar los valores (0 a 3 para los colores)
    norm = mcolors.BoundaryNorm(boundaries=[0,0.75,1.5,2.25,3], ncolors=len(colors))

    # Agregar la barra de color
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=customCmap), ax=ax)
    cbar.set_label('Tree status')  # Etiqueta para la barra de color
    cbar.set_ticks(ticksLocation)  # Ubicación de los ticks
    cbar.set_ticklabels(ticksLabels)  # Etiquetas de los ticks
    #cbar = plt.colorbar(hex_collection, ticks=ticksLocation)
    #cbar.set_ticklabels(ticksLabels)

    def update_colors(frame):
        triangulesColors = colorAssigner(historical[frame])
        triangule_collection.set_facecolors(triangulesColors)
        return [triangule_collection]

    # Crear la animación
    trianguleAni = animation.FuncAnimation(fig, update_colors, frames=len(historical), interval=interval, blit=True)
    trianguleAni.save(filename + ".gif", writer="pillow")
    return

#-------------------------------------------------------------------------------------------------------------------
def XOR(A,B):
  return ( not(A) and B) or ( A and not(B))

# Function to generate equilateral triangles for a tessellation
def generateTriangularGrid(n,m, size=2.0):
    """
    Generate a triangular grid covering the region (0, n) x (0, m)
    with equilateral triangles. 'size' is the side length of each triangle.
    """
    triangles = []
    width = size * np.sqrt(3) /2  # width of an equilateral triangle
    xmin = 0
    i = 1
    while i <= m:
        y0 = m-i
        for j in range(1,n+1,1):

            # Alternating rows of triangles (even row: pointing up, odd row: pointing down)
            if XOR(i%2,j%2):
                # Triangle pointing right
                x0 = xmin + width*(j)
                triangle = [(x0, y0), (x0 - width, y0 + size/2), (x0 - width, y0 - size/2)]

            else:
                # Triangle pointing left
                x0 = xmin + width*(j-1)
                triangle = [(x0, y0), (x0 + width, y0 + size/2), (x0 + width, y0 - size/2)]
            triangles.append(triangle)
        i += 1

    return triangles