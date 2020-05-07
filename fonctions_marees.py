import numpy as np
from saute_mouton import sys_TMS_stable


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


# C_2(t) calculé selon la distance R_astre, d_l et C_p
def C_2(R_astre, d_l, C_p):
    return (r/R_astre)**3.0 * (0.75)*np.cos(d_l)**2.0 * np.cos(2*C_p)


# fonction équilibrium tide
def Equilibrium(x_l, y_l, z_l, x_s, y_s, z_s, masse_astre, theta, delta):
    m_e = 5.9722e24
    m_s = 1.989e30
    Cte_l = r*masse_astre/m_e
    Rl = R_l(x_l, y_l, z_l)
    d_l = V_dl(Rl, z_l)
    C0 = C_0(Rl, d_l)
    C_p = Cp(x_l, y_l, theta)
    C1 = C_1(Rl, d_l, C_p)
    C2 = C_2(Rl, d_l, C_p)
    Cte_s = r*m_s/m_e
    Rs = R_l(x_s, y_s, z_s)
    d_s = V_dl(Rs, z_s)
    C0s = C_0(Rs, d_s)
    C_ps = Cp(x_s, y_s, theta)
    C1s = C_1(Rs, d_s, C_ps)
    C2s = C_2(Rs, d_s, C_ps)
    H_astre = Cte_l*(C0*(1.5*np.sin(delta)**2.0 - 0.5) + C1*np.sin(2*delta) + C2*np.cos(delta)**2)
    H_soleil = Cte_s*(C0s*(1.5*np.sin(delta)**2.0 - 0.5) + C1s*np.sin(2*delta) + C2s*np.cos(delta)**2)
    return H_astre + H_soleil


if __name__ == '__main__':
    # Trouvons R_Mars à sa position initiale pour observer son apport en F_g sur la planète
    print(R_l(sys_TMS_stable["Mars"]["x"],sys_TMS_stable["Mars"]["y"],sys_TMS_stable["Mars"]["z"]))
