import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from algorithmes.saute_mouton import mouton_3_corps


def anim_3_corps(t_i, t_f, N, c_init, F, slice):
    mouton = mouton_3_corps(t_i, t_f, N, c_init, F, slice)
    fig, ax = plt.subplots()
    #ax.set(xlim=(-5, 5), ylim=(-5, 5))

    ligne_A, = ax.plot(c_init[0][0][0], c_init[0][0][1], 'b-', label="Corps A", zorder=2)
    ligne_B, = ax.plot(c_init[0][1][0], c_init[0][1][1], 'g-', label="Corps B", zorder=3)
    ligne_C, = ax.plot(c_init[0][2][0], c_init[0][2][1], 'r-', label="Corps C", zorder=4)

    anim_ligne_A = lambda i: ligne_A.set_data(mouton["A"][:i, 0], mouton["A"][:i, 1])
    anim_ligne_B = lambda i: ligne_B.set_data(mouton["B"][:i, 0], mouton["B"][:i, 1])
    anim_ligne_C = lambda i: ligne_C.set_data(mouton["C"][:i, 0], mouton["C"][:i, 1])
    anim_titre = lambda i: ax.set_title("Mouvement des trois corps\nà t= {}".format(round(mouton["t"][i], 3)))

    frames_anim = len(mouton["t"])
    graph_anim_A = FuncAnimation(fig, anim_ligne_A, frames=frames_anim, interval=1)
    graph_anim_B = FuncAnimation(fig, anim_ligne_B, frames=frames_anim, interval=1)
    graph_anim_C = FuncAnimation(fig, anim_ligne_C, frames=frames_anim, interval=1)
    graph_anim_titre = FuncAnimation(fig, anim_titre, frames=frames_anim, interval=1)
    plt.legend()
    plt.grid()
    plt.show()


def anim_3_corps_satellite(t_i, t_f, N, c_init, F, slice):
    mouton = mouton_3_corps(t_i, t_f, N, c_init, F, slice)
    fig, ax = plt.subplots()
    ax.set(xlim=(-5e8, 5e8), ylim=(-5e8, 5e8))

    ligne_S, = ax.plot(c_init["Mars"]["x"], c_init["Mars"]["y"], 'r--', linewidth=0.2, label="Satellite", zorder=3)
    cercle = plt.Circle((0, 0), radius=1993920, alpha=0.95, zorder=10, color="maroon")
    ax.add_patch(cercle)

    anim_ligne_S = lambda i: ligne_S.set_data(mouton["L"][:i, 0], mouton["L"][:i, 1])
    anim_lim_roche = lambda i: cercle.set_center((mouton["L"][i, 0], mouton["L"][i, 1]))
    anim_titre = lambda i: ax.set_title("Mouvement du satellite par rapport à la Terre\nà t= {} jours".format(round(mouton["t"][i]/(24*3600), 3)))

    frames_anim = len(mouton["t"])
    graph_anim_S = FuncAnimation(fig, anim_ligne_S, frames=frames_anim, interval=1)
    graph_roche = FuncAnimation(fig, anim_lim_roche, frames=frames_anim, interval=1)
    graph_anim_titre = FuncAnimation(fig, anim_titre, frames=frames_anim, interval=1)
    ax.add_patch(patches.Circle((0, 0), radius=8993920, alpha=0.6, color="black"))
    plt.legend()
    plt.grid()
    plt.show()
