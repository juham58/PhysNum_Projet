import numpy as np


# V_dl seulement en fonction de la position de la lune
def V_dl(x_l, y_l, z_l):
    return np.arctan(z_l/np.arctan(y_l/x_l))


# Phi_P, attention, ce sont des variables flotantes, car on évalue sur un linspace toutes les valeurs de P
def P_phi(x, y, z):
    return np.arctan(z/np.arctan(y/x))


# Cp en fonction de la position des points V et P
def Cp(Vx, Vy, Px, Py):
    return np.arctan((Py/Px - Vy/Vx)/(1 + (Py/Px)*(Vy/Vx)))


# distance R entre les 2 astres
def R_l(x_l,y_l,z_l):
    return np.sqrt(x_l**2 + y_l**2 + z_l**2)


# C_0(t) calculé
def C_0(R_astres, d_l):
    r = 6371000.0
    return (r/R_astres)**3.0 * (1.5)*np.sin(d_l)**2.0 - 0.5


# C_1(t) calculé
def C_1(R_astres, d_l, C_p):
    return (r/R_astres)**3.0 * (0.75)*np.sin(2.0*d_l)*np.cos(2.0*C_p)


# C_2(t) calculé
def C_2(R_astres, d_l, C_p):
    return (r/R_astres)**3.0 * (0.75)*np.cos(d_l)**2.0 * np.cos(2*C_p)


# fonction équilibrium tide
def Equilibrium(x_l, y_l, z_l, fct_0, fct_1, fct_2, phi_p, masse_astre):
    r = 6371000.0
    m_e = 5.9722e24
    Cte = r*masse_astre/m_e
    C_0 = fct_0(R_l(x_l, y_l, z_l), V_dl(x_l, y_l, z_l))
    C_1 = fct_1(R_l(x_l, y_l, z_l), V_dl(x_l, y_l, z_l), )
    return



