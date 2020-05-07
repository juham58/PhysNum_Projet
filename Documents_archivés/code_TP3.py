import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# définition de la constante gravitationnelle
G = 4*constants.pi**2

# définition des masses
m_A = 1
m_B = 1
m_C = 1

# définition des constantes initiales
r_Ai = np.array([3.3030197, -0.82771837])
r_Bi = np.array([-3.3030197, 0.82771837])
r_Ci = np.array([0.0, 0.0])

v_Ai = np.array([1.587433767, 1.47221479])
v_Bi = np.array([1.587433767, 1.47221479])
v_Ci = np.array([-3.174867535, -2.94442961])


def F(corps, r_A, r_B, r_C):
    if corps == "A":
        return -G*(m_B*((r_A-r_B)/(np.linalg.norm(r_A-r_B)**3))
                   + m_C*((r_A-r_C)/(np.linalg.norm(r_A-r_C)**3)))

    if corps == "B":
        return -G*(m_A*((r_B-r_A)/(np.linalg.norm(r_B-r_A)**3))
                   + m_C*((r_B-r_C)/(np.linalg.norm(r_B-r_C)**3)))

    if corps == "C":
        return -G*(m_A*((r_C-r_A)/(np.linalg.norm(r_C-r_A)**3))
                   + m_B*((r_C-r_B)/(np.linalg.norm(r_C-r_B)**3)))


def mouton_3_corps(t_i, t_f, N, slice=0):
    t_points = np.linspace(t_i, t_f, N)
    rA_arr = np.zeros((len(t_points), 2))
    rB_arr = np.zeros((len(t_points), 2))
    rC_arr = np.zeros((len(t_points), 2))
    h = (t_f-t_i)/N

    # calcul du point v(t+h/2) avec Runge-Kutta d'ordre 4
    k1_A_v = 0.5*h*F("A", r_Ai, r_Bi, r_Ci)
    k1_B_v = 0.5*h*F("B", r_Ai, r_Bi, r_Ci)
    k1_C_v = 0.5*h*F("C", r_Ai, r_Bi, r_Ci)

    k2_A_v = 0.5*h*F("A", r_Ai+0.5*k1_A_v, r_Bi+0.5*k1_B_v, r_Ci+0.5*k1_C_v)
    k2_B_v = 0.5*h*F("B", r_Ai+0.5*k1_A_v, r_Bi+0.5*k1_B_v, r_Ci+0.5*k1_C_v)
    k2_C_v = 0.5*h*F("C", r_Ai+0.5*k1_A_v, r_Bi+0.5*k1_B_v, r_Ci+0.5*k1_C_v)

    k3_A_v = 0.5*h*F("A", r_Ai+0.5*k2_A_v, r_Bi+0.5*k2_B_v, r_Ci+0.5*k2_C_v)
    k3_B_v = 0.5*h*F("B", r_Ai+0.5*k2_A_v, r_Bi+0.5*k2_B_v, r_Ci+0.5*k2_C_v)
    k3_C_v = 0.5*h*F("C", r_Ai+0.5*k2_A_v, r_Bi+0.5*k2_B_v, r_Ci+0.5*k2_C_v)

    k4_A_v = 0.5*h*F("A", r_Ai+0.5*k3_A_v, r_Bi+0.5*k3_B_v, r_Ci+0.5*k3_C_v)
    k4_B_v = 0.5*h*F("B", r_Ai+0.5*k3_A_v, r_Bi+0.5*k3_B_v, r_Ci+0.5*k3_C_v)
    k4_C_v = 0.5*h*F("C", r_Ai+0.5*k3_A_v, r_Bi+0.5*k3_B_v, r_Ci+0.5*k3_C_v)

    v_A_demie = v_Ai + 1/6*(k1_A_v+2*k2_A_v+2*k3_A_v+k4_A_v)
    v_B_demie = v_Bi + 1/6*(k1_B_v+2*k2_B_v+2*k3_B_v+k4_B_v)
    v_C_demie = v_Ci + 1/6*(k1_C_v+2*k2_C_v+2*k3_C_v+k4_C_v)

    # on trouve r(t+1/2h)
    r_A_demie = r_Ai + 0.5*h*v_A_demie
    r_B_demie = r_Bi + 0.5*h*v_B_demie
    r_C_demie = r_Ci + 0.5*h*v_C_demie

    # On définit v(t) et r(t)
    v_A = v_Ai
    v_B = v_Bi
    v_C = v_Ci

    r_A = r_Ai
    r_B = r_Bi
    r_C = r_Ci

    # on enregistre la première rangée des array contenant les positions et t
    rA_arr[0][0], rA_arr[0][1] = r_A[0], r_A[1]
    rB_arr[0][0], rB_arr[0][1] = r_B[0], r_B[1]
    rC_arr[0][0], rC_arr[0][1] = r_C[0], r_C[1]

    # début des calculs par sauts
    for i, t in enumerate(t_points[1:]):

        v_A = v_A + h*F("A", r_A_demie, r_B_demie, r_C_demie)
        v_B = v_B + h*F("B", r_A_demie, r_B_demie, r_C_demie)
        v_C = v_C + h*F("C", r_A_demie, r_B_demie, r_C_demie)

        r_A = r_A + h*v_A_demie
        r_B = r_B + h*v_B_demie
        r_C = r_C + h*v_C_demie

        v_A_demie = v_A_demie + h*F("A", r_A, r_B, r_C)
        v_B_demie = v_B_demie + h*F("B", r_A, r_B, r_C)
        v_C_demie = v_C_demie + h*F("C", r_A, r_B, r_C)

        r_A_demie = r_A_demie + h*v_A
        r_B_demie = r_B_demie + h*v_B
        r_C_demie = r_C_demie + h*v_C

        rA_arr[i+1][0], rA_arr[i+1][1] = r_A[0], r_A[1]
        rB_arr[i+1][0], rB_arr[i+1][1] = r_B[0], r_B[1]
        rC_arr[i+1][0], rC_arr[i+1][1] = r_C[0], r_C[1]

    if slice == 0:
        return {"A": rA_arr, "B": rB_arr, "C": rC_arr, "t": t_points}

    # coupe de moitié les array de résultats un nombre de fois égale à slice
    # permet donc aux animations d'être observées dans un délai raisonnable
    for s in range(slice):
        rA_arr = np.delete(rA_arr, np.s_[1::2], 0)
        rB_arr = np.delete(rB_arr, np.s_[1::2], 0)
        rC_arr = np.delete(rC_arr, np.s_[1::2], 0)
        t_points = np.delete(t_points, np.s_[1::2], 0)
    return {"A": rA_arr, "B": rB_arr, "C": rC_arr, "t": t_points}


# fonction d'affichage des trajectoires des corps à un certain temps et N
def graph_3_corps(t_i, t_f, N):
    # on appelle une fois la fonction pour avoir les array de résultats
    mouton = mouton_3_corps(t_i, t_f, N)

    # puis on fait un graphique des trajectoires
    plt.figure(figsize=(12,6))
    plt.plot(mouton["A"][:, 0], mouton["A"][:, 1], 'b-', label="Corps A,\nr_i={},\nv_i={}".format(np.round(r_Ai, 2), np.round(v_Ai, 2)))
    plt.plot(mouton["B"][:, 0], mouton["B"][:, 1], 'g-', label="Corps B,\nr_i={},\nv_i={}".format(np.round(r_Bi, 2), np.round(v_Bi, 2)))
    plt.plot(mouton["C"][:, 0], mouton["C"][:, 1], 'r-', label="Corps C,\nr_i={},\nv_i={}".format(np.round(r_Ci, 2), np.round(v_Ci, 2)))
    plt.xlabel("Position en x [-]")
    plt.ylabel("Position en y [-]")
    plt.title("Trajectoires des corps A, B, et C pour N={}, t={}".format(N, t_f))
    plt.legend(loc=5, prop={'size': 10})
    plt.grid()
    plt.show()


# fonction d'animation des trajectoires pour N jusqu'à un certain t
def anim_3_corps(t_i, t_f, N, slice):
    mouton = mouton_3_corps(t_i, t_f, N, slice)
    fig, ax = plt.subplots()
    ax.set(xlim=(-5, 5), ylim=(-5, 5))

    ligne_A, = ax.plot(r_Ai[0], r_Ai[1], 'b-', label="Corps A", zorder=2)
    ligne_B, = ax.plot(r_Bi[0], r_Bi[1], 'g-', label="Corps B", zorder=3)
    ligne_C, = ax.plot(r_Ci[0], r_Ci[1], 'r-', label="Corps C", zorder=4)

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
