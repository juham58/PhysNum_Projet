import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import time
from mpl_toolkits import mplot3d


# définition de la constante gravitationnelle
G = 6.67408*1e-11

# définition des masses (kg)
m_T = 5.9722*1e24
m_S = 1.989*1e30
m_L = 7.349*1e22

sys_TSL =[np.array([[-1.281626781937739*1e11, -7.874213977476427*1e10, 1.169622431946918*1e7],
                     [0, 0, 0],
                     [-1.277825169776073*1e11, -7.860393013191698*1e10, -2.207933296514675*1e7]]),
                    np.array([[1.531400682940386*1e4, -2.537562925376570*1e4, 1.173818649602865],
                     [0, 0, 0],
                     [1.497454237339654*1e4, -2.446484169034256*1e4, 3.041640271121793*1e2]])]


sys_TL =[np.array([[0, 0, 0],
                    [3.801612161666349*1e8, 1.382096428472888*1e8, -3.377555728461962*1e4]]),
                    np.array([[0, 0, 0],
                     [-3.394644560073193*1e2, 9.107875634231341*1e2, 2.924258406161673*1e1]])]



def F_TSL(corps, r_A, r_B, r_C):
    if corps == "A":
        return -G*(m_S*((r_A-r_B)/(np.linalg.norm(r_A-r_B)**3))
                   + m_L*((r_A-r_C)/(np.linalg.norm(r_A-r_C)**3)))

    if corps == "B":
        return -G*(m_T*((r_B-r_A)/(np.linalg.norm(r_B-r_A)**3))
                   + m_L*((r_B-r_C)/(np.linalg.norm(r_B-r_C)**3)))

    if corps == "C":
        return -G*(m_T*((r_C-r_A)/(np.linalg.norm(r_C-r_A)**3))
                   + m_S*((r_C-r_B)/(np.linalg.norm(r_C-r_B)**3)))


def F_TL(corps, r_A, r_B):
    if corps == "A":
        return -G*(m_L*((r_A-r_B)/(np.linalg.norm(r_A-r_B)**3)))

    if corps == "B":
        return -G*(m_T*((r_B-r_A)/(np.linalg.norm(r_B-r_A)**3)))


def mouton_3_corps(t_i, t_f, N, c_init, F, slice=0):
    t_debut = time.process_time()
    t_points = np.linspace(t_i, t_f, N)
    rA_arr = np.zeros((len(t_points), 3))
    rB_arr = np.zeros((len(t_points), 3))
    rC_arr = np.zeros((len(t_points), 3))
    h = (t_f-t_i)/N

    r_Ai, r_Bi, r_Ci = c_init[0][0], c_init[0][1], c_init[0][2]
    v_Ai, v_Bi, v_Ci = c_init[1][0], c_init[1][1], c_init[1][2]

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
    rA_arr[0][0], rA_arr[0][1], rA_arr[0][2] = r_A[0], r_A[1], r_A[2]
    rB_arr[0][0], rB_arr[0][1], rB_arr[0][2] = r_B[0], r_B[1], r_B[2]
    rC_arr[0][0], rC_arr[0][1], rC_arr[0][2] = r_C[0], r_C[1], r_C[2]

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

        rA_arr[i+1][0], rA_arr[i+1][1], rA_arr[i+1][2] = r_A[0], r_A[1], r_A[2]
        rB_arr[i+1][0], rB_arr[i+1][1], rB_arr[i+1][2] = r_B[0], r_B[1], r_B[2]
        rC_arr[i+1][0], rC_arr[i+1][1], rC_arr[i+1][2] = r_C[0], r_C[1], r_C[2]

        proximite = np.abs(np.linalg.norm(r_B-r_A))
        if proximite <= 8993.92*1e3:
            print("Limite de Roche", t/(24*3600), "jours", "Distance: ", proximite)

        if proximite >= 1.47146e9:
            print("Hill Sphere: ", t/(24*3600), "jours", "Distance: ", proximite)

    if slice == 0:
        return {"A": rA_arr, "B": rB_arr, "L": rB_arr-rA_arr, "t": t_points}

    # coupe de moitié les array de résultats un nombre de fois égale à slice
    # permet donc aux animations d'être observées dans un délai raisonnable
    for s in range(slice):
        rA_arr = np.delete(rA_arr, np.s_[1::2], 0)
        rB_arr = np.delete(rB_arr, np.s_[1::2], 0)
        rC_arr = np.delete(rC_arr, np.s_[1::2], 0)
        t_points = np.delete(t_points, np.s_[1::2], 0)
    print("temps exec: ", time.process_time()-t_debut)
    return {"A": rA_arr, "B": rB_arr, "L": rB_arr-rA_arr, "t": t_points}


def mouton_2_corps(t_i, t_f, N, c_init, F, slice=0):
    t_debut = time.process_time()
    t_points = np.linspace(t_i, t_f, N)
    rA_arr = np.zeros((len(t_points), 3))
    rB_arr = np.zeros((len(t_points), 3))
    h = (t_f-t_i)/N

    r_Ai, r_Bi = c_init[0][0], c_init[0][1]
    v_Ai, v_Bi = c_init[1][0], c_init[1][1]

    # calcul du point v(t+h/2) avec Runge-Kutta d'ordre 4
    k1_A_v = 0.5*h*F("A", r_Ai, r_Bi)
    k1_B_v = 0.5*h*F("B", r_Ai, r_Bi)

    k2_A_v = 0.5*h*F("A", r_Ai+0.5*k1_A_v, r_Bi+0.5*k1_B_v)
    k2_B_v = 0.5*h*F("B", r_Ai+0.5*k1_A_v, r_Bi+0.5*k1_B_v)

    k3_A_v = 0.5*h*F("A", r_Ai+0.5*k2_A_v, r_Bi+0.5*k2_B_v)
    k3_B_v = 0.5*h*F("B", r_Ai+0.5*k2_A_v, r_Bi+0.5*k2_B_v)

    k4_A_v = 0.5*h*F("A", r_Ai+0.5*k3_A_v, r_Bi+0.5*k3_B_v)
    k4_B_v = 0.5*h*F("B", r_Ai+0.5*k3_A_v, r_Bi+0.5*k3_B_v)

    v_A_demie = v_Ai + 1/6*(k1_A_v+2*k2_A_v+2*k3_A_v+k4_A_v)
    v_B_demie = v_Bi + 1/6*(k1_B_v+2*k2_B_v+2*k3_B_v+k4_B_v)

    # on trouve r(t+1/2h)
    r_A_demie = r_Ai + 0.5*h*v_A_demie
    r_B_demie = r_Bi + 0.5*h*v_B_demie

    # On définit v(t) et r(t)
    v_A = v_Ai
    v_B = v_Bi

    r_A = r_Ai
    r_B = r_Bi

    # on enregistre la première rangée des array contenant les positions et t
    rA_arr[0][0], rA_arr[0][1], rA_arr[0][2] = r_A[0], r_A[1], r_A[2]
    rB_arr[0][0], rB_arr[0][1], rB_arr[0][2] = r_B[0], r_B[1], r_B[2]

    # début des calculs par sauts
    for i, t in enumerate(t_points[1:]):

        v_A = v_A + h*F("A", r_A_demie, r_B_demie)
        v_B = v_B + h*F("B", r_A_demie, r_B_demie)

        r_A = r_A + h*v_A_demie
        r_B = r_B + h*v_B_demie

        v_A_demie = v_A_demie + h*F("A", r_A, r_B)
        v_B_demie = v_B_demie + h*F("B", r_A, r_B)

        r_A_demie = r_A_demie + h*v_A
        r_B_demie = r_B_demie + h*v_B

        rA_arr[i+1][0], rA_arr[i+1][1], rA_arr[i+1][2] = r_A[0], r_A[1], r_A[2]
        rB_arr[i+1][0], rB_arr[i+1][1], rB_arr[i+1][2] = r_B[0], r_B[1], r_B[2]

        proximite = np.abs(np.linalg.norm(r_B-r_A))
        #print(r_A)
        if proximite <= 8993.92*1e3:
            print("Limite de Roche", t/(24*3600), " jours", "Distance: ", proximite)

    if slice == 0:
        return {"A": rA_arr, "B": rB_arr, "L": rB_arr-rA_arr, "t": t_points}

    # coupe de moitié les array de résultats un nombre de fois égale à slice
    # permet donc aux animations d'être observées dans un délai raisonnable
    for s in range(slice):
        rA_arr = np.delete(rA_arr, np.s_[1::2], 0)
        rB_arr = np.delete(rB_arr, np.s_[1::2], 0)
        t_points = np.delete(t_points, np.s_[1::2], 0)

    print("temps exec: ", time.process_time()-t_debut)
    return {"A": rA_arr, "B": rB_arr, "L": rB_arr-rA_arr, "t": t_points}


# fonction d'animation des trajectoires pour N jusqu'à un certain t
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


def anim_2_corps_satellite(t_i, t_f, N, c_init, F, slice):
    mouton = mouton_2_corps(t_i, t_f, N, c_init, F, slice)
    fig, ax = plt.subplots()
    ax.set(xlim=(-4e8, 4e8), ylim=(-4e8, 4e8))

    ligne_S, = ax.plot(c_init[0][1][0], c_init[0][1][1], 'r-', label="Satellite", zorder=3)
    cercle = plt.Circle((0, 0), radius=1993920, alpha=0.95, zorder=10, color="maroon")
    ax.add_patch(cercle)

    anim_ligne_S = lambda i: ligne_S.set_data(mouton["L"][:i, 0], mouton["L"][:i, 1])
    anim_lim_roche = lambda i: cercle.set_center((mouton["L"][i, 0], mouton["L"][i, 1]))
    anim_titre = lambda i: ax.set_title("Mouvement des trois corps\nà t= {} jours".format(round(mouton["t"][i]/(24*3600), 3)))

    frames_anim = len(mouton["t"])
    graph_anim_S = FuncAnimation(fig, anim_ligne_S, frames=frames_anim, interval=1)
    graph_roche = FuncAnimation(fig, anim_lim_roche, frames=frames_anim, interval=1)
    graph_anim_titre = FuncAnimation(fig, anim_titre, frames=frames_anim, interval=1)
    ax.add_patch(patches.Circle((0, 0), radius=8993920, alpha=0.6, color="black"))
    plt.legend()
    plt.grid()
    plt.show()

def anim_3_corps_satellite(t_i, t_f, N, c_init, F, slice):
    mouton = mouton_3_corps(t_i, t_f, N, c_init, F, slice)
    fig, ax = plt.subplots()
    ax.set(xlim=(-4e8, 4e8), ylim=(-4e8, 4e8))

    ligne_S, = ax.plot(c_init[0][1][0], c_init[0][1][1], 'r--', linewidth=0.5, label="Satellite", zorder=3)
    cercle = plt.Circle((0, 0), radius=1993920, alpha=0.95, zorder=10, color="maroon")
    ax.add_patch(cercle)

    anim_ligne_S = lambda i: ligne_S.set_data(mouton["L"][:i, 0], mouton["L"][:i, 1])
    anim_lim_roche = lambda i: cercle.set_center((mouton["L"][i, 0], mouton["L"][i, 1]))
    anim_titre = lambda i: ax.set_title("Mouvement des trois corps\nà t= {} jours".format(round(mouton["t"][i]/(24*3600), 3)))

    frames_anim = len(mouton["t"])
    graph_anim_S = FuncAnimation(fig, anim_ligne_S, frames=frames_anim, interval=1)
    graph_roche = FuncAnimation(fig, anim_lim_roche, frames=frames_anim, interval=1)
    graph_anim_titre = FuncAnimation(fig, anim_titre, frames=frames_anim, interval=1)
    ax.add_patch(patches.Circle((0, 0), radius=8993920, alpha=0.6, color="black"))
    plt.legend()
    plt.grid()
    plt.show()

def anim_2_corps(t_i, t_f, N, c_init, F, slice):
    mouton = mouton_2_corps(t_i, t_f, N, c_init, F, slice)
    fig, ax = plt.subplots()
    #ax.set(xlim=(-5, 5), ylim=(-5, 5))

    ligne_A, = ax.plot(c_init[0][0][0], c_init[0][0][1], 'b-', label="Corps A", zorder=2)
    ligne_B, = ax.plot(c_init[0][1][0], c_init[0][1][1], 'g-', label="Corps B", zorder=3)
    cercle = plt.Circle((0, 0), radius=8993920, alpha=0.6, color="black")
    ax.add_patch(cercle)

    anim_ligne_A = lambda i: ligne_A.set_data(mouton["A"][:i, 0], mouton["A"][:i, 1])
    anim_ligne_B = lambda i: ligne_B.set_data(mouton["B"][:i, 0], mouton["B"][:i, 1])
    anim_lim_roche = lambda i: cercle.set_center((mouton["A"][i, 0], mouton["A"][i, 1]))
    anim_titre = lambda i: ax.set_title("Mouvement des trois corps\nà t= {} jours".format(round(mouton["t"][i]/(24*3600), 3)))

    frames_anim = len(mouton["t"])
    graph_anim_A = FuncAnimation(fig, anim_ligne_A, frames=frames_anim, interval=1)
    graph_anim_B = FuncAnimation(fig, anim_ligne_B, frames=frames_anim, interval=1)
    graph_roche = FuncAnimation(fig, anim_lim_roche, frames=frames_anim, interval=1)
    graph_anim_titre = FuncAnimation(fig, anim_titre, frames=frames_anim, interval=1)
    plt.legend()
    plt.grid()
    plt.show()

def graph_3_corps(t_i, t_f, N, c_init, F, slice):
    # on appelle une fois la fonction pour avoir les array de résultats
    mouton = mouton_3_corps(t_i, t_f, N, c_init, F, slice)

    # puis on fait un graphique des trajectoires
    fig = plt.figure()
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


# anim_3_corps(0, 365.25*24*3600, 60000, sys_TSL, F_TSL, 10)
#graph_3_corps(0, 31*24*3600, 2000, sys_TL, F_TL)
