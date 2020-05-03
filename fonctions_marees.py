import numpy as np


r = 6371000.0
# V_dl seulement en fonction de la position de la lune
def V_dl(R_l, z_l):
    return np.arcsin(z_l/R_l)


# Cp en fonction de la position des points V et P
def Cp(x_l, y_l, theta_p):
    return theta_p - np.arctan(y_l/x_l)


# distance R entre les 2 astres
def R_l(x_l,y_l,z_l):
    return np.sqrt(x_l**2 + y_l**2 + z_l**2)


# C_0(t) calculé
def C_0(R_astre, d_l):
    return (r/R_astre)**3.0 * ((1.5)*np.sin(d_l)**2.0 - 0.5)


# C_1(t) calculé
def C_1(R_astres, d_l, C_p):
    return (r/R_astres)**3.0 * (0.75)*np.sin(2.0*d_l)*np.cos(2.0*C_p)


# C_2(t) calculé
def C_2(R_astres, d_l, C_p):
    return (r/R_astres)**3.0 * (0.75)*np.cos(d_l)**2.0 * np.cos(2*C_p)


# fonction équilibrium tide
def Equilibrium(x_l, y_l, z_l, masse_astre, theta, delta):
    m_e = 5.9722e24
    Cte = r*masse_astre/m_e
    R = R_l(x_l, y_l, z_l)
    d_l = V_dl(R, z_l)
    C0 = C_0(R, d_l)
    C_p = Cp(x_l, y_l, theta)
    C1 = C_1(R, d_l, C_p)
    C2 = C_2(R, d_l, C_p)
    return Cte*(C0*(1.5*np.sin(delta)**2.0 - 0.5) + C1*np.sin(2*delta) + C2*np.cos(delta)**2)


if __name__ == '__main__':
    x_l = 384400000.0
    y_l = 235828.00
    z_l = 492358.0
    theta = 0.213
    delta = 0.012
    a = Equilibrium(x_l, y_l, z_l, 6.4185e23, theta, delta)
    print(a)