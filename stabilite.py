import numpy as np
import pickle
from pathlib import Path
from scipy.constants import golden_ratio
from graphiques import graph_3_corps, graph_stab, graph_proximite
from anims import  anim_3_corps_satellite
from saute_mouton import mouton_3_corps
np.set_printoptions(threshold=np.inf)


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


sys_TMS_stable = {"Terre": {"x": 0.0, "y": 0.0, "z": 0.0,
                     "vx": 0.0, "vy": 0.0, "vz": 0.0},
           "Mars": {"x": 447459216.1666349, "y": 162077642.84728882, "z": -3.377555728461962*1e7,
                    "vx": -3.394644560073193*1e2, "vy": 9.107875634231341*1e2, "vz": 2.924258406161673*1e1},
           "Soleil": {"x": 1.274562537709166e11, "y": 7.981279122934923e10, "z": -4.406210892602801*1e6,
                      "vx": -1.532817776260213e4, "vy": 2.537026724877958e4, "vz": -7.731597224385212e1}}


#sys_TMS = {"Terre": {"x": 0.0, "y": 0.0, "z": 0.0,
#                     "vx": 0.0, "vy": 0.0, "vz": 0.0},
#           "Mars": {"x": 4.27961*1e8, "y": 1.54219*1e8, "z": -3.377555728461962*1e7,
#                    "vx": -3.394644560073193*1e2, "vy": 9.107875634231341*1e2, "vz": 2.924258406161673*1e1},
#           "Soleil": {"x": 1.274562537709166e11, "y": 7.981279122934923e10, "z": -4.406210892602801*1e6,
#                      "vx": -1.532817776260213e4, "vy": 2.537026724877958e4, "vz": -7.731597224385212e1}}


def F_TM(corps, r_A, r_B):
    # définition de la constante gravitationnelle
    G = 6.67408*1e-11
    if corps == "A":
        return -G*(m_M*((r_A-r_B)/(np.linalg.norm(r_A-r_B)**3)))

    if corps == "B":
        return -G*(m_T*((r_B-r_A)/(np.linalg.norm(r_B-r_A)**3)))

# force sur la Terre, Mars et Soleil
def F_TMS(corps, r_A, r_B, r_C):
    # définition de la constante gravitationnelle
    G = 6.67408*1e-11
    m_T = 5.9722*1e24  # Terre
    m_S = 1.989*1e30  # Soleil
    m_L = 7.349*1e22  # Lune
    m_M = 0.107*m_T  # Mars
    if corps == "A":
        return -G*(m_M*((r_A-r_B)/(np.linalg.norm(r_A-r_B)**3))
                   + m_S*((r_A-r_C)/(np.linalg.norm(r_A-r_C)**3)))

    if corps == "B":
        return -G*(m_T*((r_B-r_A)/(np.linalg.norm(r_B-r_A)**3))
                   + m_S*((r_B-r_C)/(np.linalg.norm(r_B-r_C)**3)))

    if corps == "C":
        return -G*(m_T*((r_C-r_A)/(np.linalg.norm(r_C-r_A)**3))
                   + m_M*((r_C-r_B)/(np.linalg.norm(r_C-r_B)**3)))


def F_TLS(corps, r_A, r_B, r_C):
    # définition de la constante gravitationnelle
    G = 6.67408*1e-11
    if corps == "A":
        return -G*(m_L*((r_A-r_B)/(np.linalg.norm(r_A-r_B)**3))
                   + m_S*((r_A-r_C)/(np.linalg.norm(r_A-r_C)**3)))

    if corps == "B":
        return -G*(m_T*((r_B-r_A)/(np.linalg.norm(r_B-r_A)**3))
                   + m_S*((r_B-r_C)/(np.linalg.norm(r_B-r_C)**3)))

    if corps == "C":
        return -G*(m_T*((r_C-r_A)/(np.linalg.norm(r_C-r_A)**3))
                   + m_L*((r_C-r_B)/(np.linalg.norm(r_C-r_B)**3)))


# Sauvegarde les données produites par la fonction mouton_3_corps
def sauvegarde(t_i, t_f, N, c_init, F, slice=0, var="x", tol_valide=True, id=0):
    mouton = mouton_3_corps(t_i, t_f, N, c_init, F, slice, var)
    nom_fichier = "TMS_{}_jours_{}_N_{}{}".format(round(t_f/(24*3600)), N, var, id)
    pickle.dump(mouton, open(str(Path.cwd()/"Données"/nom_fichier), "w+b"))
    return mouton


# charge les données produites par la fonction mouton_3_corps
# préalablement sauvegardées par la fonction sauvegarde
def chargement(nom_fichier):
    data = pickle.load(open(str(Path.cwd()/"Données"/str(nom_fichier)), "rb"))
    return data


# Applique un nombre de perturbations d'un certain ordre (exemple 10^7 m)
# sur une condition initiales var et enregistre les données
def perturbations(t_i, t_f, N, c_init, F, slice=0, var="x", nombre=10, ordre=1e7):
    c_init["Mars"][var] = c_init["Mars"][var] - (nombre/2)*ordre
    orbites_stables = 0  # on comptera le nombre d'orbites stables trouvés
    for i in range(nombre):
        c_init["Mars"][var] = c_init["Mars"][var] + i*ordre
        data = sauvegarde(t_i, t_f, N, c_init, F, slice, var, tol_valide=False, id=i)

        if data["valide"] is True:
            orbites_stables += 1

        if data["valide"] is False and orbites_stables > 15:
            return i  # retourne le nombre de fichiers créés après au moins 15 orbites stables

    return nombre  # retorune le nombre de fichiers créés si auncun orbite instable



# calcul la stabilité d'un orbite en trouvant l'écart type de l'excentricité
# temps en jours, noms des fichiers sans le dernier chiffre
def stabilite(temps, N, nom_fichiers, var="x", nombre_fichiers=10):
    stabilite = np.zeros(nombre_fichiers)
    c_init = []
    for n in range(nombre_fichiers):
        mouton = chargement(nom_fichiers+str(n))
        mois = int(31*N//temps)
        liste_mois = np.split(mouton["P"], np.arange(0, N-1, mois))
        liste_mois.pop(0)
        excentricites = np.zeros(len(liste_mois))
        for m in range(len(liste_mois)):
            a = (np.amax(liste_mois[m])-np.amin(liste_mois[m]))/2
            b = np.sqrt(np.median(liste_mois[m])**2 - (np.amin(liste_mois[m])-a)**2)
            excentricites[m] = np.sqrt(1-(a**2/b**2))
        if mouton["valide"] is False:
            stabilite[n] = np.nan
            c_init.append(mouton["c_init"]["Mars"][var])
        else:
            stabilite[n] = 1/np.std(excentricites)
            c_init.append(mouton["c_init"]["Mars"][var])
    #print(len(stabilite))
    return stabilite, c_init


# algorithme adaptatif permettant de trouver les conditions initiales en x et y
# qui créent un orbite stable
# en utilisant les fonctions perturbations et stabilite
def recherche_stab(t_i, t_f, N, c_init, F, slice=0, nombre=10, ordre=1e7):
    temps = round((t_f-t_i)/(24*3600))
    nombre_fichiers = nombre
    nombre_x, nombre_y = nombre, nombre
    delta_x, delta_y = np.inf, np.inf
    while delta_x >= 1e4 or delta_y >= 1e4:

        #stabilité en x
        var = "x"
        nom_fichiers = "TMS_{}_jours_{}_N_{}".format(temps, N, var)
        c_init_x = c_init["Mars"]["x"]
        nombre_fichiers_x = perturbations(t_i, t_f, N, c_init, F, slice, var="x", nombre=nombre_x, ordre=ordre)
        stab_data_x = stabilite(temps, N, nom_fichiers, var="x", nombre_fichiers=nombre_fichiers_x)
        graph_stab(stab_data_x, var="x", c_init=c_init_x)
        stab_x = list(stab_data_x[0])
        cond_x = list(stab_data_x[1])
        index_x = stab_x.index(max(stab_x))  # on trouve la position du maximum dans la liste
        c_init["Mars"]["x"] = cond_x[index_x]  # on trouve la condition initiale correspondant au max

        delta_x = np.abs(c_init_x - c_init["Mars"]["x"])
        c_init_x = c_init["Mars"]["x"]
        print("-------\n", "delta_x: ", delta_x, " max: ", cond_x[index_x], "\n-------")

        # stabilité en y
        var = "y"
        nom_fichiers = "TMS_{}_jours_{}_N_{}".format(temps, N, var)
        c_init_y = c_init["Mars"]["y"]
        nombre_fichiers_y = perturbations(t_i, t_f, N, c_init, F, slice, var="y", nombre=nombre_y, ordre=ordre)
        stab_data_y = stabilite(temps, N, nom_fichiers, var="y", nombre_fichiers=nombre_fichiers_y)
        graph_stab(stab_data_y, var="y", c_init=c_init_y)
        stab_y = list(stab_data_y[0])
        cond_y = list(stab_data_y[1])
        index_y = stab_y.index(max(stab_y))  # on trouve la position du maximum dans la liste
        c_init["Mars"]["y"] = cond_y[index_y]  # on trouve la condition initiale correspondant au max

        print(nombre_y)
        delta_y = np.abs(c_init_y - c_init["Mars"]["y"])
        c_init_y = c_init["Mars"]["y"]
        print("-------\n", "delta_y: ", delta_y, " max: ", cond_y[index_y], "\n-------")

    print(c_init, nom_fichiers, nombre_fichiers_y)
    pickle.dump(c_init, open(str(Path.cwd()/"Données"/"c_init"/nom_fichiers), "w+b"))  # on enregistre les données trouvées
    return c_init

#anim_3_corps_satellite(0, 10*12*31*24*3600, 20000, sys_TMS, F_TMS, 6)
#chargement("TMS_372_jours_30000_N_x0")
#print(mouton_3_corps(0, 24*3600, 5, sys_TMS, F_TMS))

#print(stabilite(3720, 20000, "TMS_3720_jours_20000_N_x", nombre_fichiers=30))
#recherche_stab(0, 3*12*31*24*3600, 3000, sys_TMS, F_TMS, nombre=43000, ordre=1e3)
#mouton_3_corps(0, 2*31*24*3600, 2000, sys_TMS, F_TMS, slice=4)

#perturbations(0, 12*31*24*3600, 2000, sys_TMS, F_TMS, nombre=4000, ordre=1e4)
#graph_stab(stabilite(2*31, 4000, "TMS_744_jours_4000_N_x", var="x", nombre_fichiers=180))

#anim_3_corps_satellite(0, 2*12*31*24*3600, 60000, sys_TMS, F_TMS, 4)
#anim_3_corps_satellite(0, 2*12*31*24*3600, 60000, sys_TMS_stable, F_TMS, 4)
graph_3_corps(0, 2*12*31*24*3600, 60000, sys_TMS, F_TMS, 0)
graph_3_corps(0, 2*12*31*24*3600, 60000, sys_TMS_stable, F_TMS, 0)
#graph_proximite(0, 3*12*31*24*3600, 60000, sys_TMS, F_TMS, 0)
#graph_proximite(0, 3*12*31*24*3600, 60000, sys_TMS_stable, F_TMS, 0)
