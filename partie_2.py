import numpy as np
import pickle
from pathlib import Path
from graphiques import graph_3_corps, graph_instab
from anims import anim_2_corps_satellite, anim_3_corps_satellite
from saute_mouton import mouton_2_corps, mouton_3_corps
np.set_printoptions(threshold=np.inf)

# définition de la constante gravitationnelle
G = 6.67408*1e-11


# définition des masses (kg)
m_T = 5.9722*1e24  # Terre
m_S = 1.989*1e30  # Soleil
m_L = 7.349*1e22  # Lune
m_M = 0.107*m_T  # Mars


# système Terre-Mars
sys_TM = [np.array([[0, 0, 0],
                    [3.801612161666349*1e8, 1.382096428472888*1e8, -3.377555728461962*1e4]]),
          np.array([[0, 0, 0],
                    [-3.394644560073193*1e2, 9.107875634231341*1e2, 2.924258406161673*1e1]])]


# système Terre-Mars-Soleil
sys_TMS = {"Terre": {"x": 0.0, "y": 0.0, "z": 0.0,
                     "vx": 0.0, "vy": 0.0, "vz": 0.0},
           "Mars": {"x": 3.801612161666349*1e8, "y": 1.382096428472888*1e8, "z": -3.377555728461962*1e7,
                    "vx": -3.394644560073193*1e2, "vy": 9.107875634231341*1e2, "vz": 2.924258406161673*1e1},
           "Soleil": {"x": 1.274562537709166e11, "y": 7.981279122934923e10, "z": -4.406210892602801*1e6,
                      "vx": -1.532817776260213e4, "vy": 2.537026724877958e4, "vz": -7.731597224385212e1}}


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


def sauvegarde(t_i, t_f, N, c_init, F, slice=0, var="x", id=0):
    mouton = mouton_3_corps(t_i, t_f, N, c_init, F, slice, var)
    nom_fichier = "TMS_{}_jours_{}_N_{}{}".format(round(t_f/(24*3600)), N, var, id)
    pickle.dump(mouton, open(str(Path.cwd()/"Données"/nom_fichier), "w+b"))


def chargement(nom_fichier):
    data = pickle.load(open(str(Path.cwd()/"Données"/str(nom_fichier)), "rb"))
    return data


# Applique un nombre de perturbations d'un certain ordre (exemple 10^7 m)
# sur une condition initiales var et enregistre les données
def perturbations(t_i, t_f, N, c_init, F, slice=0, var="x", nombre=10, ordre=1e7):
    c_init["Mars"][var] = c_init["Mars"][var] - (nombre/2)*ordre
    for i in range(nombre):
        c_init["Mars"][var] = c_init["Mars"][var] + i*ordre
        sauvegarde(t_i, t_f, N, c_init, F, slice, var, id=i)



# temps en jours, noms des fichiers sans le dernier chiffre
def instabilite(temps, N, nom_fichiers, var="x", nombre_fichiers=10):
    instabilite = np.zeros(nombre_fichiers)
    c_init = []
    for n in range(nombre_fichiers):
        mouton = chargement(nom_fichiers+str(n))
        mois = int(31*N//temps)
        liste_mois = np.split(mouton["P"], np.arange(0, 19999, mois))
        liste_mois.pop(0)
        excentricites = np.zeros(len(liste_mois))
        for m in range(len(liste_mois)):
            a = (np.amax(liste_mois[m])-np.amin(liste_mois[m]))/2
            b = np.sqrt(np.median(liste_mois[m])**2 - (np.amin(liste_mois[m])-a)**2)
            excentricites[m] = np.sqrt(1-(a**2/b**2))
        if mouton["valide"] is False:
            instabilite[n] = np.nan
            c_init.append(mouton["c_init"]["Mars"][var])
        else:
            instabilite[n] = np.std(excentricites)
            c_init.append(mouton["c_init"]["Mars"][var])
    return instabilite, c_init

#print(mouton_3_corps(0, 24*3600, 5, sys_TMS, F_TMS))

#print(instabilite(3720, 20000, "TMS_3720_jours_20000_N_x", nombre_fichiers=30))

perturbations(0, 12*12*31*24*3600, 200000, sys_TMS, F_TMS, nombre=100, ordre=1e5)

graph_instab(instabilite(4464, 200000, "TMS_4464_jours_20000_N_x", nombre_fichiers=100))

#anim_3_corps_satellite(0, 100*12*31*24*3600, 2000000, sys_TMS, F_TMS, 6)
#graph_3_corps(0, 2*12*31*24*3600, 20000, sys_TMS, F_TMS, 0)
