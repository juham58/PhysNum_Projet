import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from partie_1 import mouton_2_corps, anim_2_corps, anim_2_corps_satellite, anim_3_corps_satellite, graph_3_corps


# définition de la constante gravitationnelle
G = 6.67408*1e-11


# définition des masses (kg)
m_T = 5.9722*1e24  # Terre
m_S = 1.989*1e30  # Soleil
m_L = 7.349*1e22  # Lune
m_M = 0.107*m_T  # Mars

sys_TM = [np.array([[0, 0, 0],
                    [1.801612161666349*1e8, 1.382096428472888*1e8, -3.377555728461962*1e4]]),
          np.array([[0, 0, 0],
                    [-3.394644560073193*1e2, 9.107875634231341*1e2, 2.924258406161673*1e1]])]

sys_TMS = [np.array([[0, 0, 0],
                    [3.801612161666349*1e8, 1.382096428472888*1e8, -3.377555728461962*1e7],
                    [1.274562537709166e11, 7.981279122934923e10, -4.406210892602801*1e6]]),
          np.array([[0, 0, 0],
                    [-3.394644560073193*1e2, 9.107875634231341*1e2, 2.924258406161673*1e1],
                    [-1.532817776260213e4, 2.537026724877958e4, -7.731597224385212e1]])]


def F_TM(corps, r_A, r_B):
    if corps == "A":
        return -G*(m_M*((r_A-r_B)/(np.linalg.norm(r_A-r_B)**3)))

    if corps == "B":
        return -G*(m_T*((r_B-r_A)/(np.linalg.norm(r_B-r_A)**3)))


def F_TMS(corps, r_A, r_B, r_C):
    if corps == "A":
        return -G*(m_M*((r_A-r_B)/(np.linalg.norm(r_A-r_B)**3))
                   + m_S*((r_A-r_C)/(np.linalg.norm(r_A-r_C)**3)))

    if corps == "B":
        return -G*(m_T*((r_B-r_A)/(np.linalg.norm(r_B-r_A)**3))
                   + m_S*((r_B-r_C)/(np.linalg.norm(r_B-r_C)**3)))

    if corps == "C":
        return -G*(m_T*((r_C-r_A)/(np.linalg.norm(r_C-r_A)**3))
                   + m_M*((r_C-r_B)/(np.linalg.norm(r_C-r_B)**3)))


#anim_3_corps_satellite(0, 100*12*31*24*3600, 2000000, sys_TMS, F_TMS, 6)
graph_3_corps(0, 1000*12*31*24*3600, 2000000, sys_TMS, F_TMS, 4)
