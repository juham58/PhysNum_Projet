from fonctions_marees import Equilibrium
import numpy as np


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

    return theta, delta, G


# Retourne une fonction Equilibrium avec seulement theta et delta comme valeur (1 position et 1 masse)
def fonc_Equilibrium(x_l, y_l, z_l, m_astre):
    return lambda theta, delta: Equilibrium(x_l, y_l, z_l, m_astre, theta, delta)


if __name__ == '__main__':
    # fct_Equilibrium is the new Equilibrium(theta, delta)
    fct_Equilibrium = fonc_Equilibrium(384400000.0, 235828.00, 492358.0, 6.4185e23)
    Grid_equilibrium = grid_fct(200, fct_Equilibrium)