import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii

#Constants
G = 6.67430e-8

class SatelliteModel:
    def __init__(self, name, radius, layers, nlayers):
        self.name = name
        self.radius = radius
        self.layers = layers
        self.nlayers = nlayers
        """
        Next line will need work on the distribution of the layers.
        """
        #self.r = np.linspace(1, radius, nlayers)
        #self.r = np.logspace(0, np.log10(radius), nlayers)
        
        # Initialisations
        self.mass = np.zeros(nlayers)
        self.gravity = np.zeros(nlayers)
        self.pressure = np.zeros(nlayers)

    def perso_density_profile(self):
        """
        Obsolete if we use an EOS.
        This will need to be updated once we use EOSs to calculate density in each nlayer.
        """
        rho = np.zeros_like(self.r)
        for layer in self.layers:
            rho = np.where(self.r <= layer["radius"], layer["density"], rho)
        return rho
        
    def integrate_structure_iterate(self,max_iter):
        model = self.read_guess_model()
        self.r = np.array(model['R_CM'])[::-1]
        self.dr = self.diff_r()
        #P = np.array(model['P_CGS'])[::-1]
        P = np.ones_like(self.r) * 1e12
        P_surf = 1e6
        for i in range(max_iter):  # Fixed-point iteration
            rho = self.polytrope(P)
            shell_volumes = 4 * np.pi * self.r**2 * self.dr
            shell_masses = shell_volumes * rho
            mass = np.cumsum(shell_masses)
            gravity = np.zeros_like(self.r)
            gravity[1:] = G * mass[:-1] / self.r[1:]**2
            dP = -rho * gravity * self.dr
            P_new = np.zeros_like(P)
            P_new[:-1] = np.cumsum(dP[::-1])[-2::-1]
            P_new = P_new - P_new[0] + P_surf
            #print(P_new)
            print("- - -")
            if np.allclose(P, P_new, rtol=1e-4):
                break
            P = P_new
            print(f"Iter {i}, max(P): {P.max():.2e}, max(rho): {rho.max():.2e}")
        # Final assignment
        self.pressure = P
        self.rho = rho
        self.mass = mass
        self.gravity = gravity
        
    def mass_conservation(self):
        shell_volumes = 4 * np.pi * self.r**2 * self.dr
        shell_masses = shell_volumes * rho
        self.mass = np.cumsum(shell_masses)
        
    def read_guess_model(self):
        model = ascii.read("JupiterModel/jup_howard23.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        return model
        
    def diff_r(self):
        dr = np.diff(self.r, prepend=self.r[0])
        #dr = np.diff(self.r)
        return dr

    def polytrope(self, pressure):
        n = 1
        K = 1.96e11
        rho = (pressure / K) ** (n / (n + 1))
        return rho

    def plot(self):
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.r / 1e5, self.rho)
        plt.title("Density (g/cc)")
        plt.xlabel("Radius (km)")
        plt.subplot(1, 3, 2)
        plt.plot(self.r / 1e5, self.pressure/1e9)
        plt.title("Pressure (kbar)")
        plt.xlabel("Radius (km)")
        plt.subplot(1, 3, 3)
        plt.plot(self.r / 1e5, self.mass / 1e24)
        plt.title("Cumulated mass ($10^{24}$ g)")
        plt.xlabel("Radius (km)")
        plt.suptitle(f"Internal structure of {self.name}")
        plt.tight_layout()
        plt.show()

