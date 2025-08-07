import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

#Constants
G = 6.67430e-8

class SatelliteModel:
    def __init__(self, name, radius, layers, nlayers, distribution_type):
        """
        r : radii, from center to surface
        m : masses, ---
        pressure : pressures, from surface to center
        temperature : temperatures, ---
        rho : densities, ---
        """
        self.name = name
        self.radius = radius
        self.layers = layers
        self.nlayers = nlayers
        
        # Call to select the radius mesh
        self.radius_distrib(distribution_type)
        
        # Initialisations
        self.mass = np.zeros(nlayers)
        self.gravity = np.zeros(nlayers)
        self.pressure = np.zeros(nlayers)
        self.temperature = np.zeros(nlayers)
        self.rho = np.zeros(nlayers)
        
    def integrate_structure_iterate(self,max_iter,rtol,debug,P_surf,T_surf):
        """
        function which solves mass conservation and hydrostatic equilibrium, and iterates to converge to a solution.
        """
        #model = self.read_guess_model()
        #self.r = np.array(model['R_CM'])[::-1]
        #self.pressure = np.array(model['P_CGS'])[::-1]
        self.dr = self.diff_r()
        self.pressure = np.ones_like(self.r) * 1e12
        mtot_prev = None
        
        for iter in range(max_iter):
            self.get_density() #call the EOS to compute rho
            self.mass_conservation()
            self.hydrostatic_eq(P_surf)
            self.heat_transport(T_surf)
            
            #Check convergence on total mass
            try:
                if mtot_prev is not None:
                    relerr_mtot = (self.mass[-1] - mtot_prev) / mtot_prev
                    if debug:
                        print(f"Iter {iter}, relative error on M_tot: {relerr_mtot:.2e}")
                    if abs(relerr_mtot) < rtol:
                        print(f"Convergence on total mass reached after {iter} iterations.")
                        break
            except Exception as e:
                print(f"Une erreur est survenue : {e}")
            mtot_prev = self.mass[-1]
            
            if debug:
                print(f"Iter {iter}, max(P): {P.max():.2e}, max(rho): {rho.max():.2e}")
        
    def mass_conservation(self):
        """
        dm/dr = 4 * pi * r^2 * rho
        """
        shell_volumes = 4 * np.pi * self.r**2 * self.dr
        shell_masses = shell_volumes * self.rho
        self.mass = np.cumsum(shell_masses)
        
    def hydrostatic_eq(self,P_surf):
        """
        dP/dr = - rho * g
        """
        #self.gravity[1:] = G * self.mass[:-1] / self.r[1:]**2 #POURQUOI self.mass[:-1] ???
        self.gravity[1:] = G * self.mass[1:] / self.r[1:]**2
        dP = -self.rho * self.gravity * self.dr
        self.pressure[-1] = 0
        self.pressure[:-1] = np.cumsum(dP[::-1])[-2::-1]
        self.pressure = self.pressure - self.pressure[0] + P_surf
        
    def heat_transport(self,T_surf):
        """
        dT/dP = T/P * nabla_T
        """
        grada = self.nabla_T()
        grada = grada[:]
        interp_nabla = interp1d(self.pressure[:],grada, bounds_error=False, fill_value=0.3)
        def dT_dP(p,t):
            #return t / p * interp_nabla(p)
            return t / p * 0.3
        P_eval = self.pressure[:]
        sol = solve_ivp(dT_dP,(P_eval[0],P_eval[-1]),np.array([T_surf]),t_eval=P_eval)
        assert sol.success, "Temperature integration failed..."
        T_out = sol.y[0]
        self.temperature = T_out
        #print(self.temperature)
        
    def nabla_T(self):
        """
        calculation of nabla_T for heat transport equation.
        """
        grada = np.ones_like(self.pressure) * 0.3
        return grada
        
    def read_guess_model(self):
        """
        to start from an existing model.
        """
        model = ascii.read("JupiterModel/jup_howard23.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        return model
        
    def diff_r(self):
        """
        calculates the radius difference between two layers of the model.
        """
        dr = np.diff(self.r, prepend=self.r[0])
        #dr = np.diff(self.r, prepend=self.r[0] - (self.r[1] - self.r[0]))
        return dr
        
    def get_density(self):
        rho = np.zeros_like(self.pressure)
        for i, layer in enumerate(self.layers):
            n = layer["n"]
            K = layer["K"]
            r_max = layer["radius"]
            if i == 0:
                r_min = 0  # core
            else:
                r_min = self.layers[i - 1]["radius"]
            # find indices between r_min and r_max
            mask = (self.r  >= r_min) & (self.r  <= r_max)
            indices = np.where(mask)[0]
            j_indices = self.nlayers - indices - 1
            P_sel = self.pressure[j_indices]
            """
            c'est ici qu'il faudra faire l'appel à l'eos pour avoir la densité
            """
            rho[j_indices] = self.polytrope(P_sel, n, K)
        self.rho = rho

    def polytrope(self, pressure, n, K):
        return (pressure / K) ** (n / (n + 1))

    def radius_distrib(self,distribution_type):
        """
        defines the selected radii mesh
        """
        if distribution_type == "linear":
            self.r = np.linspace(0, self.radius, self.nlayers)
        elif distribution_type == "log":
            self.r = np.logspace(0, np.log10(self.radius), self.nlayers)
        elif distribution_type == "exp":
            mesh_surf_amp = 1e5 #increase to have more layers near surface
            mesh_surf_width = 5e-2 #controls the width of the refined region
            f = np.linspace(0, 1, self.nlayers)
            refinement = 1 + mesh_surf_amp * f * np.exp((f - 1) / mesh_surf_width) #densité de points dans l'espace normalisé.
            r_norm = np.cumsum(1 / refinement) #1/refinement = espacement local entre 2 points. Cumsum pour cumul des espacements locaux.
            r_norm -= r_norm[0] #to start at 0
            r_norm /= r_norm[-1] #to get r[-1]=1
            self.r = r_norm * self.radius
        else:
            self.r = np.linspace(1, self.radius, self.nlayers)
            print('default is linear and is probably bad')
        print(f"Radius mesh is : {distribution_type}")
        #self.plot_radius_distrib(distribution_type)
        
    def moment_of_inertia(self):
        """
        for a sphere.
        """
        I = np.sum(4 * np.pi * self.r**4 * self.rho * self.dr)
        return I
        
    def perso_density_profile(self):
        """
        Obsolete if we use an EOS.
        """
        rho = np.zeros_like(self.r)
        for layer in self.layers:
            rho = np.where(self.r <= layer["radius"], layer["density"], rho)
        return rho

    def plot(self):
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.r / 1e5, self.rho)
        plt.title("Density (g/cc)")
        plt.xlabel("Radius (km)")
        plt.subplot(1, 3, 2)
        plt.plot(self.pressure/1e6,self.temperature)
        plt.title("Temperature (K)")
        plt.xlabel("Pressure (bar)")
        plt.subplot(1, 3, 3)
        plt.plot(self.r / 1e5, self.mass / 1e24)
        plt.title("Cumulated mass ($10^{24}$ g)")
        plt.xlabel("Radius (km)")
        plt.suptitle(f"Internal structure of {self.name}")
        plt.tight_layout()
        plt.show()

    def plot_radius_distrib(self,distribution_type):
        plt.figure(figsize=(10,10))
        plt.scatter(np.linspace(1,self.nlayers,self.nlayers),self.r,label=distribution_type,s=1)
        plt.legend()
        plt.show()
