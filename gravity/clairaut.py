import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d

"""
Résolution de l'équation de Clairaut pour le calcul des aplatissements
d'un ellipsoïde triaxial en rotation synchrone.

Équation (59) de Baland+2015:
d²α/dr0² + (6/r0)(ρ/ρ̄)(dα/dr0) - (6/r0²)(1 - ρ/ρ̄)α = 0

avec la condition aux limites (60):
dα/dr0(R) = (1/R)[25/4 * q_r - 2α(R)]
"""

def solve_clairaut(r_array, rho_array, q_r):
    R = r_array[-1]

    #éviter exactement r=0
    eps = 1e-6 * R
    r = r_array.copy()
    r[0] = eps

    # --- Densité moyenne ---
    #see my Obsedian notes in MOONS.
    integrand = rho_array * r_array**2
    integral = cumulative_trapezoid(integrand, r_array, initial=0)
    rho_bar = np.zeros_like(r_array)
    rho_bar[1:] = 3 * integral[1:] / r_array[1:]**3
    rho_bar[0] = rho_array[0]  #évite division par 0 au centre

    rho_func = interp1d(r, rho_array, kind='linear')
    rhobar_func = interp1d(r, rho_bar, kind='linear')

    # --- ODE ---
    def ode(r, y):
        alpha = y[0]
        dalpha = y[1]

        rho = rho_func(r)
        rhobar = rhobar_func(r)

        d2alpha = np.zeros_like(r)
        mask = r > eps #loin du centre: équation complète.
        d2alpha[mask] = (-(6 * rho[mask] / (r[mask] * rhobar[mask])) * dalpha[mask] + (6 / r[mask]**2) * (1 - rho[mask] / rhobar[mask]) * alpha[mask])
        d2alpha[~mask] = 0.0 #au centre pour éviter la division par 0.

        return np.vstack((dalpha, d2alpha))

    # --- BC ---
    def bc(ya, yb):
        bc_center = ya[1]  # alpha'(0)=0
        bc_surface = yb[1] - (1/R)*(25/4*q_r - 2*yb[0])
        return np.array([bc_center, bc_surface])

    #initial guess
    y_init = np.zeros((2, r.size))
    y_init[0] = 1e-4

    sol = solve_bvp(ode, bc, r, y_init, tol=1e-6, max_nodes=20000)
    if not sol.success:
        raise RuntimeError("La résolution BVP a échoué")
    return sol.x, sol.y[0]
