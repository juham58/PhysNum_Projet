from fonctions_etude_marees.fonctions_marees import Equilibrium
import numpy as np
import matplotlib.pyplot as plt
from algorithmes.saute_mouton import sys_TMS_stable, mouton_3_corps, sys_TMS, F_TMS, F_TLS


m_T = 5.9722*1e24  # Terre
m_S = 1.989*1e30  # Soleil
m_L = 7.349*1e22  # Lune
m_M = 0.107*m_T  # Mars


# Retourne une fonction Equilibrium avec seulement theta et delta comme paramètres d'entrée
def fonc_Equilibrium(x_l, y_l, z_l, x_s, y_s, z_s, m_astre):
    return lambda theta, delta: Equilibrium(x_l, y_l, z_l, x_s, y_s, z_s, m_astre, theta, delta)


# Crée une grid NxN de champ scalaire p/r à une fonction donnée f(theta, delta) et la plot en "heatmap"
def grid_fct(N, fct):
    lon = np.linspace(-1.0 * np.pi, 1.0 * np.pi, N)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, N)
    theta, delta = np.meshgrid(lon,lat)
    G = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            G[i,j] = fct(theta[i,j], delta[i,j])

    plt.pcolor(theta, delta, G)
    c = plt.colorbar()
    c.ax.set_ylabel("Hauteur de la marée [m]", rotation=270)
    plt.xlabel("Longitude [rad]")
    plt.ylabel("Latitude [rad]")
    plt.show()


# fonction qui crée une liste d'array de NxN. Chaque array est un champ scalaire pour une différente position d'astres
def grid_mouton(N, equilib, sm, masse_astre):
    lon = np.linspace(-1.0 * np.pi, 1.0 * np.pi, N)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, N)
    liste = []
    somme_moyenne = 0
    for i in range(len(sm["t"])):
        theta, delta = np.meshgrid(lon, lat)
        theta = theta + (2*np.pi/(24*3600))*sm["t"][i]
        G = np.zeros((N, N))
        G = equilib(sm["L"][i, 0], sm["L"][i, 1], sm["L"][i, 2], sm["S"][i, 0], sm["S"][i, 1], sm["S"][i, 2], masse_astre, theta, delta)
        liste.append(G)
        somme_moyenne += G.max() - G.min()
    moyenne = somme_moyenne/len(sm["t"])
    print("Moyenne du marnage: {}".format(moyenne))
    return liste


if __name__ == '__main__':
    x_l = sys_TMS_stable["Mars"]["x"]
    y_l = sys_TMS_stable["Mars"]["y"]
    z_l = sys_TMS_stable["Mars"]["z"]
    x_s = sys_TMS_stable["Soleil"]["x"]
    y_s = sys_TMS_stable["Soleil"]["y"]
    z_s = sys_TMS_stable["Soleil"]["z"]
    fct_eq = fonc_Equilibrium(x_l, y_l, z_l, x_s, y_s, z_s, m_M)
    grid_fct(200, fct_eq)

    # Calcul du marnage moyen pour 28 jours
    positions_mars = mouton_3_corps(0, 28 * 24 * 3600, 2000, sys_TMS_stable, F_TMS, slice=4)
    positions_lune = mouton_3_corps(0, 28 * 24 * 3600, 2000, sys_TMS, F_TLS, slice=4)
    a = grid_mouton(200, Equilibrium, positions_mars, m_M)
    b = grid_mouton(200, Equilibrium, positions_lune, m_L)