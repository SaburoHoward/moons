import numpy as np
import matplotlib.pyplot as plt

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
        self.r = np.linspace(1, radius, nlayers)
        #self.r = np.logspace(0, np.log10(radius), nlayers)
        self.dr = self.r[1] - self.r[0]
        
        # Computes the density profile
        #self.rho = self._build_density_profile()
        
        # Initialisations
        self.mass = np.zeros(nlayers)
        self.gravity = np.zeros(nlayers)
        self.pressure = np.zeros(nlayers)

    def _build_density_profile(self):
        """
        Obsolete if we use an EOS.
        This will need to be updated once we use EOSs to calculate density in each nlayer.
        """
        rho = np.zeros_like(self.r)
        for layer in self.layers:
            rho = np.where(self.r <= layer["radius"], layer["density"], rho)
        return rho

    def integrate_structure(self):
        """
        dr will not be constant anymore if I use sthing else than linspace.
        """
        shell_volumes = 4 * np.pi * self.r**2 * self.dr
        shell_masses = shell_volumes * self.rho
        # Cumulated mass
        self.mass = np.cumsum(shell_masses)
        
        # Initialise with zeros so that g(0)=0
        self.gravity = np.zeros_like(self.r)
        self.gravity[1:] = G * self.mass[:-1] / self.r[1:]**2
        
        # I integrate from the surface to the center
        dP = -self.rho * self.gravity * self.dr
        self.pressure = np.zeros_like(self.rho)
        # Cumulative inverted integration
        self.pressure[:-1] = np.cumsum(dP[::-1])[-2::-1]
        
        P_surf = 1e6
        self.pressure = self.pressure - self.pressure[0] + P_surf
        guess_ini = self.pressure
        return guess_ini
        
    def integrate_structure_iterate(self):
        P = np.ones_like(self.r) * 1e6 # Initial guess for pressure profile
        #P = self.integrate_structure()
        print(P)
        P_surf = 1e6
        for _ in range(1):  # Fixed-point iteration
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
            print(P_new)
            print("- - -")
            if np.allclose(P, P_new, rtol=1e-4):
                break
            P = P_new
        # Final assignment
        self.pressure = P
        self.rho = rho
        self.mass = mass
        self.gravity = gravity

    def polytrope(self, pressure):
        n = 1
        K = 1.96
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

