import matplotlib.pyplot as plt
import numpy as np
from saute_mouton import mouton_3_corps, sys_TMS_stable, F_TMS, F_TLS, sys_TMS
from fonctions_marees import Equilibrium


m_T = 5.9722*1e24  # Terre
m_S = 1.989*1e30  # Soleil
m_L = 7.349*1e22  # Lune
m_M = 0.107*m_T  # Mars


positions_mars = mouton_3_corps(0, 31 * 24 * 3600, 2000, sys_TMS_stable, F_TMS, slice=2)
positions_lune = mouton_3_corps(0, 31 * 24 * 3600, 2000, sys_TMS, F_TLS, slice=2)


# Fonction qui fait un graphique de la hauteur de la marée à un point précis sur terre selon le temps
def graph_maree_1pt(equilib, sm, masse_astre, teta, deta):
    liste1 = []
    for i in range(len(sm["t"])):
        theta = teta + (2*np.pi/(24*3600))*sm["t"][i]
        G = 0
        G = equilib(sm["L"][i, 0], sm["L"][i, 1], sm["L"][i, 2], sm["S"][i, 0], sm["S"][i, 1], sm["S"][i, 2], masse_astre, theta, deta)
        liste1.append(G)
    liste2 = []
    for i in range(len(sm["t"])):
        liste2.append(sm["t"][i]/(24*3600))
    plt.figure()
    plt.xlabel("temps [jours]")
    plt.ylabel("Hauteur des marées [m]")
    plt.plot(liste2, liste1)
    plt.grid()
    plt.show()

theta_qc = -1.2428137
delta_qc = 0.8170563
# Graphiques de hauteur des marées pour 31 jours à Québec pour la Lune et pour Mars
graph_maree_1pt(Equilibrium, positions_mars, m_M, theta_qc, delta_qc)
graph_maree_1pt(Equilibrium, positions_lune, m_L, theta_qc, delta_qc)
# Graphiques de hauteur des marées pour 31 jours au Méridien de Greenwich pour la Lune et pour Mars
graph_maree_1pt(Equilibrium, positions_mars, m_M, 0, 0)
graph_maree_1pt(Equilibrium, positions_lune, m_L, 0, 0)
