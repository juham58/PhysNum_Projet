import numpy as np


# V_dl seulement en fonction de la position de la lune
def V_dl(x, y, z):
    return np.arctan(z/np.arctan(y/x))

# Phi_P, attention, ce sont des variables flotantes, car on évalue sur un linspace toutes les valeurs de P
def P_phi(x, y, z):
    return np.arctan(z/np.arctan(y/x))

# Cp en fonction de la position des points V et P
def Cp(Vx, Vy, Px, Py):
    return np.arctan((Py/Px - Vy/Vx)/(1 + (Py/Px)*(Vy/Vx)))


#
def 
