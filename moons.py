import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from eos.analytical_eos import polytrope, hm1989_rocks_vec
from eos.mixture import linear_mixing, pure_eos

#Constants
G = 6.67430e-8

class SatelliteModel:
    def __init__(self, name, mass, layers, nlayers, distribution_type):
        """
        r : radii, from center to surface
        m : masses, ---
        pressure : pressures, from surface to center
        temperature : temperatures, ---
        rho : densities, ---
        """
        self.name = name
        self.mass = mass
        self.layers = layers
        self.nlayers = nlayers
        
        # Call to select the radius mesh
        self.mass_distrib(distribution_type)
        
        # Initialisations
        self.r = np.zeros(nlayers) #radii
        self.gravity = np.zeros(nlayers)
        self.p = np.zeros(nlayers) #pressures
        self.t = np.zeros(nlayers) #temperatures
        self.rho = np.zeros(nlayers) #densities
        self.deltarho = np.zeros(nlayers) #to evaluate the density differences between two time steps
        self.s = np.zeros(nlayers) #entropies
        self.gradad = np.zeros(nlayers)
        self.svpk = [] #to stock the interpolation functions from EOSs. Name comes from CEPAM.
        
    def integrate_structure_iterate(self,max_iter,rtol,debug,P_surf,T_surf,min_iterations=10):
        """function which solves mass conservation and hydrostatic equilibrium, and iterates to converge to a solution."""
        #model = self.read_guess_model()
        #self.temperature = np.array(model['T_K'])
        self.dm = self.diff_m()
        
        self.p = np.ones_like(self.m) * 1e12
        self.t = np.ones_like(self.m) * 1e4
        
        rtot_prev = None
        
        for iter in range(max_iter):
            print(f"- - - Iteration {iter} - - -")
            self.get_density(iter) #call the EOS to compute rho
            self.mass_conservation()
            self.hydrostatic_eq(P_surf)
            self.heat_transport(T_surf)
            
            try:
                if rtot_prev is not None:
                    relerr_rtot = (self.r[-1] - rtot_prev) / rtot_prev
                    if debug:
                        print(f"Iter {iter}, relative error on R_tot: {relerr_rtot:.2e}")
                    if abs(relerr_rtot) < rtol and iter >= min_iterations:
                        print(f"Convergence on total radius reached after {iter} iterations.")
                        break
            except Exception as e:
                print(f"Une erreur est survenue : {e}")
            rtot_prev = self.r[-1]
            
            if debug:
                print(f"Iter {iter}, max(P): {P.max():.2e}, max(rho): {rho.max():.2e}")
        self.save_output()
        
    def mass_conservation(self):
        """dm/dr = 4 * pi * r^2 * rho"""
        q = np.zeros_like(self.rho)
        q[0] = 0.
        q[1:] = 3.*self.dm/4/np.pi/self.rho[1:]
        self.r = np.cumsum(q)**(1./3)

    def hydrostatic_eq(self,P_surf):
        """dP/dr = - rho * g"""
        dp = G * self.m[1:]*self.dm/4./np.pi/self.r[1:]**4
        self.p[-1] = P_surf
        self.p[:-1] = P_surf + np.cumsum(dp[::-1])[::-1]

    def heat_transport(self,T_surf):
        """dT/dP = T/P * nabla_T"""
        gradt = self.gradad
        #gradt = self.nabla_T()
        mask = (gradt == 0.)
        isoT_indices = np.where(mask)[0] #I check where gradt=0 and save the indices to further set the temperature
        interp_nabla = interp1d(self.p,gradt)
        def dT_dP(p,t):
            return t / p * interp_nabla(p)
        P_eval = self.p[:][::-1]
        sol = solve_ivp(dT_dP,(P_eval[0],P_eval[-1]),np.array([T_surf]),t_eval=P_eval)
        assert sol.success, "Temperature integration failed..."
        T_out = sol.y[0][::-1]
        self.t = T_out
        self.t[isoT_indices]=np.max(T_out)
        
    def nabla_T(self):
        """calculation of nabla_T for heat transport equation."""
        grada = np.ones_like(self.p) * 0.3
        return grada
        
    def diff_m(self):
        """calculates the radius difference between two layers of the model."""
        #dm = np.diff(self.m, prepend=self.m[0]) #besoin du prepend???
        dm = np.diff(self.m)
        return dm
        
    def get_density(self,iter):
        """
        ici on fait l'initialisation et on récupère interp_rho et interp_s pour chaque EOS de chaque couche.
        On stocke ça dans svpk. svpk=[(interp_rho_elem0_layer0,interp_S_elem0_layer0),(interp_rho_elem1_layer0,interp_S_elem1_layer0),...]
        """
        if iter==0:
            for i, layer in enumerate(self.layers):
                layer_svpk = []
                layer_svpk_gradad = []
                if layer["eos"]=="mixture":
                    for j in range(layer["nbelem"]):
                        interp_rho,interp_s = pure_eos(layer["files"][j],"cubic")
                        layer_svpk.append([interp_rho,interp_s])
                    self.svpk.append(layer_svpk)
                else:
                    self.svpk.append(["None","None"]) #to avoid indices problemes when using different layer["eos"] in one structure.
        
        rho = np.zeros_like(self.p)
        gradad = np.zeros_like(self.p)
        s = np.zeros_like(self.p)
        for i, layer in enumerate(self.layers):
            m_max = layer["mass"]
            if i == 0:
                m_min = 0  # core
            else:
                m_min = self.layers[i - 1]["mass"]
            # find indices between m_min and m_max
            mask = (self.m  >= m_min) & (self.m  <= m_max)
            indices = np.where(mask)[0]
            #j_indices = self.nlayers - indices - 1
            P_sel = self.p[indices]
            T_sel = self.t[indices]
            if layer["eos"]=="polytrope":
                n = layer["n"]
                K = layer["K"]
                rho[indices] = polytrope(P_sel, n, K)
            if layer["eos"]=="hm1989_rocks":
                rho[indices] = hm1989_rocks_vec(P_sel, layer["roches"])
            if layer["eos"]=="mixture":
                rho[indices],s[indices],gradad[indices] = linear_mixing(np.log10(P_sel),np.log10(T_sel),layer["nbelem"],layer["mass_fractions"],self.svpk[i])
                if layer["T_struct"]=="isotherm":
                    gradad[indices]=0.
        self.rho = rho
        self.s = s
        self.gradad = gradad

    def mass_distrib(self,distribution_type):
        """defines the selected radii mesh"""
        if distribution_type == "linear":
            self.m = np.linspace(0, self.mass, self.nlayers)
        elif distribution_type == "log":
            self.m = np.logspace(0, np.log10(self.mass), self.nlayers)
        elif distribution_type == "exp":
            mesh_surf_amp = 1e5 #increase to have more layers near surface
            mesh_surf_width = 5e-2 #controls the width of the refined region
            f = np.linspace(0, 1, self.nlayers)
            refinement = 1 + mesh_surf_amp * f * np.exp((f - 1) / mesh_surf_width) #densité de points dans l'espace normalisé.
            m_norm = np.cumsum(1 / refinement) #1/refinement = espacement local entre 2 points. Cumsum pour cumul des espacements locaux.
            m_norm -= m_norm[0] #to start at 0
            m_norm /= m_norm[-1] #to get r[-1]=1
            self.m = m_norm * self.mass
        else:
            self.m = np.linspace(1, self.mass, self.nlayers)
            print('default is linear and is probably bad')
        print(f"Mass mesh is : {distribution_type}")
        #self.plot_mass_distrib(distribution_type)
        
    def moment_of_inertia(self):
        """
        for a sphere.
        """
        I = np.sum(4 * np.pi * self.r**4 * self.rho * self.dr)
        return I
        
    def perso_density_profile(self):
        """Obsolete if we use an EOS."""
        rho = np.zeros_like(self.m)
        for layer in self.layers:
            rho = np.where(self.m <= layer["mass"], layer["density"], rho)
        return rho
        
    def read_guess_model(self):
        """to start from an existing model."""
        #model = ascii.read("JupiterModel/jup_howard23.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        #model = ascii.read("JupiterModel/Jupiter_Z0/jup_pureH-He.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        #model = ascii.read("JupiterModel/Jupiter_Z0/jup_RockyCore1percent.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        #model = ascii.read("JupiterModel/Polytrope/jup.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        return model
        
    def save_output(self):
        output = Table([(self.r / self.r[-1])[::-1],(self.m / self.m[-1])[::-1],self.r[::-1],self.m[::-1],
                        self.p[::-1],self.t[::-1],self.rho[::-1],self.s[::-1],self.gradad[::-1]],
                names=('R_RTOT','M_MTOT','R_CM','M_G','P_CGS','T_K','RHO_GCC','S_CGS','GRADAD'))
        output.write("model.csv",overwrite=True)

    def plot(self):
        jup_ref = self.read_guess_model()
        fig = plt.figure(figsize=(14, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.r / 1e5, self.rho,label='MOONS')
        plt.plot(jup_ref['R_CM']/1e5,jup_ref['RHO_GCC'],label='CEPAM')
        plt.legend()
        plt.title("Density (g/cc)")
        plt.xlabel("Radius (km) (depth)")
        plt.subplot(1, 3, 2)
        plt.plot(self.p/1e6,self.t)
        plt.plot(jup_ref['P_CGS']/1e6,jup_ref['T_K'])
        plt.title("Temperature (K)")
        plt.xlabel("Pressure (bar)")
        plt.subplot(1, 3, 3)
        plt.plot(self.r / 1e5, self.m / 1e24)
        plt.plot(jup_ref['R_CM']/1e5,jup_ref['# M_MTOT']*1.89861120e+6)
        plt.title("Cumulated mass ($10^{24}$ g)")
        plt.xlabel("Radius (km)")
        plt.suptitle(f"Internal structure of {self.name}")
        plt.tight_layout()
        plt.show()
        #fig.savefig("Figures/CMS19_comparison.pdf",bbox_inches="tight")

    def plot_mass_distrib(self,distribution_type):
        plt.figure(figsize=(10,10))
        plt.scatter(np.linspace(1,self.nlayers,self.nlayers),self.m,label=distribution_type,s=1)
        plt.legend()
        plt.show()
