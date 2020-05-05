import pyvista as pv
import numpy as np
from Grid_2D_Equilibrium import grid_mouton
from saute_mouton import mouton_3_corps, sys_TMS, F_TMS, F_TLS
from fonctions_marees import Equilibrium


m_T = 5.9722*1e24  # Terre
m_S = 1.989*1e30  # Soleil
m_L = 7.349*1e22  # Lune
m_M = 0.107*m_T  # Mars


def _cell_bounds(points, bound_position=0.1):
    assert points.ndim == 1, "Only 1D points are allowed"
    diffs = np.diff(points)
    delta = diffs[0] * bound_position
    bounds = np.concatenate([[points[0] - delta], points + delta])
    return bounds


RADIUS = 6371.0
x = np.linspace(-180.0, 180.0, 200)
y = np.linspace(-90, 91, 200)
y_polar = 90.0 - y


# Create arrays of grid cell boundaries, which have shape of (x.shape[0] + 1)
xx_bounds = _cell_bounds(x)
yy_bounds = _cell_bounds(y_polar)
levels = [RADIUS * 1.01]


positions_mars = mouton_3_corps(0, 14 * 24 * 3600, 2000, sys_TMS, F_TMS, slice=2)
positions_lune = mouton_3_corps(0, 14 * 24 * 3600, 2000, sys_TMS, F_TLS, slice=2)
a = grid_mouton(200, Equilibrium, positions_mars, m_M)
b = grid_mouton(200, Equilibrium, positions_lune, m_L)

# Remplir la grille de valeurs scalaires
grid_scalar = pv.grid_from_sph_coords(xx_bounds, yy_bounds, levels)
grid_scalar.cell_arrays["Hauteur de la maree (m)"] = np.array(a[0]).swapaxes(-2, -1).ravel("C")


# Fonction qui calcule les coordonnées et retourne un array (x,y,z)
def points(lon, lat):
    R = RADIUS
    return np.array([1.01*R*np.cos(lat)*np.cos(lon), 1.01*R*np.cos(lat)*np.sin(lon), 1.01*R*np.sin(lat)])


# Crée des points sur la sphère
label = ["Québec", "Paris", "Tokyo", "Brasilia"]
Quebec = points(-1.2428137, 0.8170563)
Paris = points(0.0409942, 0.85265268)
Tokyo = points(2.440659, 0.622259)
Brasilia = points(-0.8365314, -0.2754080)

Points_array = np.vstack((Quebec, Paris, Tokyo, Brasilia))

# Make a plot
p = pv.Plotter()
p.add_mesh(pv.Sphere(radius=RADIUS), color="white", style="surface")
p.show_bounds()
p.add_point_labels(Points_array, label)
p.add_mesh(grid_scalar, clim=[-4.00, 6.00], opacity=0.9, cmap="cividis")


print('Orient the view, then press "q" to close window and produce movie')

# setup camera and close
p.show(auto_close=False)

# Open a gif
p.open_gif("marée_mars_deux_semaines.gif")

# Update Z and write a frame for each updated position
for i in range(len(a)):
    grid_scalar.cell_arrays["Hauteur de la maree (m)"] = np.array(a[i]).swapaxes(-2, -1).ravel("C")
    p.write_frame()

# Close movie and delete object
p.close()