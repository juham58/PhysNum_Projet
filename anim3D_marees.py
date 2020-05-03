from fonctions_marees import Equilibrium
import matplotlib.pyplot as plt
import numpy as np


# def f(x, y):
    # return y * np.sin(x)

# Crée une grid NxN
def grid(N, fct):
    lon = np.linspace(-1.0 * np.pi, 1.0 * np.pi, N)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, N)
    theta, delta = np.meshgrid(lon,lat)
    G = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            G[i,j] = fct(theta[i,j], delta[i,j])

    return theta, delta, G

# plt.pcolor(X, Y, Z)
# plt.show()


# Retourne une fonction Equilibrium avec seulement theta et delta comme valeur (1 position et 1 masse)
def fonc_Equilibrium(x_l, y_l, z_l, m_astre):
    return lambda theta, delta: Equilibrium(x_l, y_l, z_l, m_astre, theta, delta)


def mapping_map_to_sphere(lon, lat, radius = 6731000.0):
    # this function maps the points of coords (lon, lat) to points onto the  sphere of radius radius

    lon = np.array(lon, dtype=np.float64)
    lat = np.array(lat, dtype=np.float64)
    xs = radius * np.cos(lon) * np.cos(lat)
    ys = radius * np.sin(lon) * np.cos(lat)
    zs = radius * np.sin(lat)
    return xs, ys, zs


if __name__ == '__main__':
    # fct_Equilibrium is the new Equilibrium(theta, delta)
    fct_Equilibrium = fonc_Equilibrium(384400000.0, 235828.00, 492358.0, 6.4185e23)
    Grid_equilibrium = grid(200, fct_Equilibrium)
    plt.pcolormesh(Grid_equilibrium[0], Grid_equilibrium[1], Grid_equilibrium[2])
    plt.show()