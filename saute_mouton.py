import numpy as np
import time


def mouton_3_corps(t_i, t_f, N, c_init, F, slice=0, var= "x"):
    t_debut = time.process_time()
    t_points = np.linspace(t_i, t_f, N)
    rA_arr = np.zeros((len(t_points), 3))
    rB_arr = np.zeros((len(t_points), 3))
    rC_arr = np.zeros((len(t_points), 3))
    prox_arr = np.zeros((len(t_points), 1))
    valide = True
    h = (t_f-t_i)/N

    r_Ai = np.array([c_init["Terre"]["x"], c_init["Terre"]["y"], c_init["Terre"]["z"]])
    r_Bi = np.array([c_init["Mars"]["x"], c_init["Mars"]["y"], c_init["Mars"]["z"]])
    r_Ci = np.array([c_init["Soleil"]["x"], c_init["Soleil"]["y"], c_init["Soleil"]["z"]])
    v_Ai = np.array([c_init["Terre"]["vx"], c_init["Terre"]["vy"], c_init["Terre"]["vz"]])
    v_Bi = np.array([c_init["Mars"]["vx"], c_init["Mars"]["vy"], c_init["Mars"]["vz"]])
    v_Ci = np.array([c_init["Soleil"]["vx"], c_init["Soleil"]["vy"], c_init["Soleil"]["vz"]])

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
    proximite = np.abs(np.linalg.norm(r_B-r_A))
    prox_arr[0][0] = proximite

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
        prox_arr[i+1][0] = proximite
        if proximite <= 8993920:
            valide = False

        if proximite >= 1.47146e9:
            valide = False

    #if valide is False:
        #print("Orbite instable")

    if slice == 0:
        #print("temps exec: ", time.process_time()-t_debut)
        return {"A": rA_arr, "B": rB_arr, "L": rB_arr-rA_arr, "P": prox_arr, "t": t_points, "valide": valide, "c_init": c_init, "var": var}

    # coupe de moitié les array de résultats un nombre de fois égale à slice
    # permet donc aux animations d'être observées dans un délai raisonnable
    for s in range(slice):
        rA_arr = np.delete(rA_arr, np.s_[1::2], 0)
        rB_arr = np.delete(rB_arr, np.s_[1::2], 0)
        rC_arr = np.delete(rC_arr, np.s_[1::2], 0)
        t_points = np.delete(t_points, np.s_[1::2], 0)
        prox_arr = np.delete(prox_arr, np.s_[1::2], 0)
    #print("temps exec: ", time.process_time()-t_debut)
    return {"A": rA_arr, "B": rB_arr, "L": rB_arr-rA_arr, "P": prox_arr, "t": t_points, "valide": valide, "c_init": c_init, "var": var}


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
