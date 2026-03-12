import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv, spherical_jn, spherical_yn

"""
Calcul du champ magnétique induit.
Selon Vance 2020 (ou 2021), voir le Supplementary. Sinon c'est aussi défini dans Macri et al. 2025
A*exp(-i*phi_A) = [j_n+1(k*R_out)*y_n+1(k*R_in) - j_n+1(k*R_in)*y_n+1(k*R_out)] / [j_n+1(k*R_in)*y_n-1(k*R_out) - j_n-1(k*R_out)*y_n+1(k*R_in)]
A : amplitude of the induced magnetic field (or induction efficiency?)
phi_A : phase delay of the induced magnetic field
j_n and y_n : spherical Bessel functions of the first and second kind of degree n.
k = sqrt(i*omega*mu_0*sigma_o), omega = 2pi/T_J (T_J=9.93hours), mu_0 the magnetic permeability of free space, sigma_o conductivity of the ocean.
R_out and R_in : outer and inner radii of the ocean layer.

I have to deal when Q=NaN. This seems to happen when the conductivity is larger than 1. This may be explained that when sigma>1, k*R >> 1 which leads to
j(k*r_out)*y(k*r_in)-j(k*r_in)*y(k*r_out) ~ 0.
"""

def spherical_jn_complex(n, z):
    # j_n(z) = sqrt(pi/(2z)) * J_{n+0.5}(z)
    return np.sqrt(np.pi / (2*z)) * jv(n + 0.5, z)
    
def spherical_yn_complex(n, z):
    # j_n(z) = sqrt(pi/(2z)) * J_{n+0.5}(z)
    return np.sqrt(np.pi / (2*z)) * yv(n + 0.5, z)

def induced_field(sigma_ocean,r_out,r_in):
    omega = 2*np.pi/(9.93*3600) #rad/s^-1
    mu_0 = 1.256637e-6 #kg.m.s^-2.A^-2
    sigma_o = sigma_ocean #S.m^-1 = kg^-1.m^-3.s^3.A^2
    k = np.sqrt(1j*omega*mu_0*sigma_o) #k is in m^-1. So I will need my radii R_in and R_out in m.
    Q = (spherical_jn(2,k*r_out)*spherical_yn(2,k*r_in)-spherical_jn(2,k*r_in)*spherical_yn(2,k*r_out)) / (spherical_jn(2,k*r_in)*spherical_yn(0,k*r_out)-spherical_jn(0,k*r_out)*spherical_yn(2,k*r_in))
    return np.abs(Q),-np.angle(Q)
