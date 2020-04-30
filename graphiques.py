import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
from saute_mouton import mouton_3_corps


def graph_proximite(t_i, t_f, N, c_init, F, slice):
    # on appelle une fois la fonction pour avoir les array de résultats
    mouton = mouton_3_corps(t_i, t_f, N, c_init, F, slice)

    # puis on fait un graphique des trajectoires
    fig, ax = plt.subplots()
    plt.plot(mouton["t"], mouton["P"], 'r--', linewidth=1, label="Satellite")
    plt.xlabel("Position en x [-]")
    plt.ylabel("Position en y [-]")
    plt.legend(loc=5, prop={'size': 10})
    plt.grid()

    plt.show()


def graph_3_corps(t_i, t_f, N, c_init, F, slice):
    # on appelle une fois la fonction pour avoir les array de résultats
    mouton = mouton_3_corps(t_i, t_f, N, c_init, F, slice)

    # puis on fait un graphique des trajectoires
    fig, ax = plt.subplots()
    ax.add_patch(patches.Circle((0, 0), radius=8993920, alpha=0.6, color="black"))
    plt.plot(mouton["L"][:, 0], mouton["L"][:, 1], 'r--', linewidth=0.1, label="Satellite")
    plt.xlabel("Position en x [-]")
    plt.ylabel("Position en y [-]")
    plt.legend(loc=5, prop={'size': 10})
    plt.grid()

    plt.show()


def graph_3_corps_3d(t_i, t_f, N, c_init, F):
    # on appelle une fois la fonction pour avoir les array de résultats
    mouton = mouton_3_corps(t_i, t_f, N, c_init, F)

    # puis on fait un graphique des trajectoires
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    plt.plot(mouton["L"][:, 0], mouton["L"][:, 1], mouton["L"][:, 2], 'r--', linewidth=0.1, label="Satellite")
    plt.xlabel("Position en x [-]")
    plt.ylabel("Position en y [-]")
    plt.legend(loc=5, prop={'size': 10})
    plt.grid()

    N=20
    stride=1
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    r = 8993920
    ax.plot_surface(r*x, r*y, r*z, linewidth=0.0, cstride=stride, rstride=stride, alpha=0.6, color="black")
    plt.show()

def graph_instab(instab, var="x"):
    plt.figure()
    plt.xlabel("Conndition initiale de {} [m]".format(var))
    plt.ylabel("Écart type de l'excentricité\nde l'orbite de Mars [-]")
    plt.plot(instab[1], instab[0])
    plt.grid()
    plt.show()
