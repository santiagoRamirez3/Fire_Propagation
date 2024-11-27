import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
import matplotlib.animation as animation

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

#========================================================================================
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # Tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # Normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # Finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

#======================================================================================

def generateAnimation(vor,filename:str, historical:list, interval:int, p_bond, p_site):
    regions, vertices = voronoi_finite_polygons_2d(vor)
    polygons = [vertices[region] for region in regions]

    vorColors = colorAssigner(historical[0])
    collection = PolyCollection(polygons, facecolors=vorColors, linewidths=0.1, edgecolor='black', cmap=customCmap)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Voronoi's tessellation simulation", size=20)
    ax.set_xlabel(r'$P_{bond}=$' + str(round(p_bond,2)) + r'  $P_{site}=$' + str(round(p_site,2)), size=15)
    ax.add_collection(collection)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_aspect('equal')

    # Normalizar los valores (0 a 3 para los colores)
    norm = mcolors.BoundaryNorm(boundaries=[0,0.75,1.5,2.25,3], ncolors=len(colors))

    # Agregar la barra de color
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=customCmap), ax=ax)
    cbar.set_label('Estado del árbol')  # Etiqueta para la barra de color
    cbar.set_ticks(ticksLocation)  # Ubicación de los ticks
    cbar.set_ticklabels(ticksLabels)  # Etiquetas de los ticks

    def update_colors(frame):
        # Generar colores aleatorios para cada fotograma
        voronoiColors = colorAssigner(historical[frame])
        collection.set_facecolors(voronoiColors)
        return [collection]

    # Crear la animación
    hexagonalAni = animation.FuncAnimation(fig, update_colors, frames=len(historical), interval=interval, blit=True)
    hexagonalAni.save(filename + ".gif", writer="pillow")
    return