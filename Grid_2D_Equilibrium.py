from fonctions_marees import Equilibrium
import numpy as np
from saute_mouton import mouton_3_corps, sys_TMS, F_TLS, F_TMS


m_T = 5.9722*1e24  # Terre
m_S = 1.989*1e30  # Soleil
m_L = 7.349*1e22  # Lune
m_M = 0.107*m_T  # Mars

# def f(x, y):
    # return y * np.sin(x)

# Crée une grid NxN et sort une grille de champ scalaire p/r à une fonction donnée f(theta, delta)
def grid_fct(N, fct):
    lon = np.linspace(-1.0 * np.pi, 1.0 * np.pi, N)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, N)
    theta, delta = np.meshgrid(lon,lat)
    G = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            G[i,j] = fct(theta[i,j], delta[i,j])

    return G


def grid_mouton(N, equilib, sm, masse_astre):
    lon = np.linspace(-1.0 * np.pi, 1.0 * np.pi, N)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, N)
    liste = []
    for i in range(len(sm["t"])):
        theta, delta = np.meshgrid(lon, lat)
        theta = theta + (2*np.pi/(24*3600))*sm["t"][i]
        G = np.zeros((N, N))
        G = equilib(sm["L"][i, 0], sm["L"][i, 1], sm["L"][i, 2], sm["S"][i, 0], sm["S"][i, 1], sm["S"][i, 2], masse_astre, theta, delta)
        liste.append(G)
    return liste


# Retourne une fonction Equilibrium avec seulement theta et delta comme valeur (1 position et 1 masse)
def fonc_Equilibrium(x_l, y_l, z_l, m_astre):
    return lambda theta, delta: Equilibrium(x_l, y_l, z_l, m_astre, theta, delta)


if __name__ == '__main__':
    positions_mars = mouton_3_corps(0, 31 * 24 * 3600, 2000, sys_TMS, F_TMS, slice=6)
    positions_lune = mouton_3_corps(0, 31 * 24 * 3600, 2000, sys_TMS, F_TLS, slice=6)
    a = grid_mouton(50, Equilibrium, positions_mars, m_M)
