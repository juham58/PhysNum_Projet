import pyvista as pv
import numpy as np
from Grid_2D_Equilibrium import grid_fct, fonc_Equilibrium
import mpl_toolkits

def _cell_bounds(points, bound_position=0.5):
    """
    Calculate coordinate cell boundaries.

    Parameters
    ----------
    points: numpy.array
        One-dimensional array of uniformy spaced values of shape (M,)
    bound_position: bool, optional
        The desired position of the bounds relative to the position
        of the points.

    Returns
    -------
    bounds: numpy.array
        Array of shape (M+1,)

    Examples
    --------
    >>> a = np.arange(-1, 2.5, 0.5)
    >>> a
    array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    >>> cell_bounds(a)
    array([-1.25, -0.75, -0.25,  0.25,  0.75,  1.25,  1.75,  2.25])
    """
    assert points.ndim == 1, "Only 1D points are allowed"
    diffs = np.diff(points)
    delta = diffs[0] * bound_position
    bounds = np.concatenate([[points[0] - delta], points + delta])
    return bounds


# First, create some dummy data

# Approximate radius of the Earth
RADIUS = 6371.0

# Longitudes and latitudes
x = np.linspace(-180.0, 180.0, 200)
y = np.linspace(-90, 90, 200)
y_polar = 90.0 - y  # grid_from_sph_coords() expects polar angle

xx, yy = np.meshgrid(x, y)


fct_Equilibrium = fonc_Equilibrium(384400000.0, 235828.00, 492358.0, 6.4185e23)
Grid_equilibrium = grid_fct(200, fct_Equilibrium)


# Create arrays of grid cell boundaries, which have shape of (x.shape[0] + 1)
xx_bounds = _cell_bounds(x)
yy_bounds = _cell_bounds(y_polar)
# Vertical levels
# in this case a single level slightly above the surface of a sphere
levels = [RADIUS * 1.01]


grid_scalar = pv.grid_from_sph_coords(xx_bounds, yy_bounds, levels)

# And fill its cell arrays with the scalar data
grid_scalar.cell_arrays["Hauteur de la marée (mètres)"] = np.array(Grid_equilibrium[2]).swapaxes(-2, -1).ravel("C")


def points(lon, lat):
    R = RADIUS
    return np.array([1.01*R*np.cos(lat)*np.cos(lon), 1.01*R*np.cos(lat)*np.sin(lon), 1.01*R*np.sin(lat)])

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
p.add_mesh(grid_scalar, clim=[-2.00, 3.50], opacity=0.9, cmap="cividis")
p.show()