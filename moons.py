import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import erf

from eos.analytical_eos import polytrope, hm1989_rocks_vec
from eos.mixture import linear_mixing, pure_eos

from gravity.gravitational_harmonics import Pn, C
from gravity.clairaut import solve_clairaut

#Constants
G = 6.673848e-8

class SatelliteModel:
    def __init__(self, name, mass, Prot, layers, nlayers, distribution_type):
        """
        r : radii, from center to surface
        m : masses, ---
        pressure : pressures, from surface to center
        temperature : temperatures, ---
        rho : densities, ---
        """
        self.name = name
        self.mass = mass
        self.Prot = Prot
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
        
    def integrate_structure_iterate(self,max_iter,rtol,debug,P_surf,T_surf,from_scratch,save,min_iterations=10):
        """function which solves mass conservation and hydrostatic equilibrium, and iterates to converge to a solution."""
        #model = self.read_guess_model()
        #self.m = model["# M_MTOT"].data[::-1]*self.mass
        #self.temperature = np.array(model['T_K'])
        #print(self.m)
        self.dm = self.diff_m()
        if from_scratch:
            self.p = np.ones_like(self.m) * 1e12
        else:
            # To converge faster, we start directly from a previously calculated polytrope.
            polytrope = self.start_from_poly()
            self.p = polytrope['P_CGS'].data[::-1]
    
        self.t = np.ones_like(self.m) * 1e4
        rtot_prev = None
        
        for iter in range(max_iter):
            #print(f"- - - Iteration {iter} - - -")
            self.get_density(iter) #call the EOS to compute rho
            self.mass_conservation()
            self.hydrostatic_eq(P_surf)
            #self.heat_transport(T_surf)
            
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
        if save:
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
        if isoT_indices.size == 0:
            interp_nabla = interp1d(self.p,gradt,bounds_error=False,fill_value="extrapolate")
        else:
            interp_nabla = interp1d(self.p[isoT_indices[-1]+1:],gradt[isoT_indices[-1]+1:],bounds_error=False,fill_value="extrapolate")
        def dT_dP(p,t):
            return t / p * interp_nabla(p)
        P_eval = self.p[:][::-1]
        if isoT_indices.size != 0:
            P_eval = P_eval[:self.nlayers-isoT_indices[-1]-1]
        sol = solve_ivp(dT_dP,(P_eval[0],P_eval[-1]),np.array([T_surf]),t_eval=P_eval)
        assert sol.success, "Temperature integration failed..."
        T_out = sol.y[0][::-1]
        if isoT_indices.size == 0:
            self.t = T_out
        else:
            self.t[isoT_indices[-1]+1:] = T_out
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
        This returns not only density, but also entropy and adiabatic gradient.
        I first initialise to get interp_rho et interp_s for each tabulated eos in the different layers.
        This is saved in svpk. svpk=[(interp_rho_elem0_layer0,interp_S_elem0_layer0),(interp_rho_elem1_layer0,interp_S_elem1_layer0),...]
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
            if layer["eos"]=="constant_density":
                rho[indices] = layer["constant_rho"]
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
        elif distribution_type == "debras":
            beta = 6.0 / self.nlayers
            i = np.arange(self.nlayers)
            lambdas = 1.0 - (np.exp(i * beta) - 1.0) / (np.exp(self.nlayers * beta) - 1.0)
            self.m = lambdas[::-1] * self.mass
        elif distribution_type == "erf":
            p1 = 2.7
            p2 = 0.6
            xi = np.linspace(-p1 + p2, p1 + p2, self.nlayers)
            mesh = 0.5 * (1 + erf(xi))
            mesh = (mesh - mesh[0]) / (mesh[-1] - mesh[0])
            self.m = mesh * self.mass
        else:
            self.m = np.linspace(1, self.mass, self.nlayers)
            print('default is linear and is probably bad')
        #print(f"Mass mesh is : {distribution_type}")
        
    def moment_of_inertia(self):
        """
        for a sphere.
        """
        R_cm = self.r[-1]
        I = np.trapz((8/3)*np.pi * self.r**4 * self.rho, x=self.r)
        return R_cm,I
        
    def get_gravity(self):
        J = C(2,0,self.mass,self.r,self.rho)
        return J
        
    def call_clairaut(self):
        omega = 2*np.pi/self.Prot/24/3600
        q_r = omega**2*self.r[-1]**3/G/self.mass
        r_sol, alpha_sol = solve_clairaut(self.r/self.r[-1], self.rho, q_r)
        J2 = (alpha_sol[-1]-(5/4)*q_r)*2/3
        return J2
        
    def read_guess_model(self):
        """to start from/or compare to an existing model."""
        #model = ascii.read("JupiterModel/Polytrope/jup_1000.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        #model = ascii.read("JupiterModel/jup_howard23.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        #model = ascii.read("JupiterModel/Jupiter_Z0/jup_pureH-He.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        #model = ascii.read("JupiterModel/Jupiter_Z0/jup_RockyCore1percent.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        model = ascii.read("JupiterModel/Jupiter_Z0/jup_RockyCore10percent.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
        return model
        
    def start_from_poly(self):
        poly = Table.read("model_ini/polytrope_jup_1000.csv",format='csv')
        return poly
        
    def save_output(self):
        output = Table([(self.r / self.r[-1])[::-1],(self.m / self.m[-1])[::-1],self.r[::-1],self.m[::-1],
                        self.p[::-1],self.t[::-1],self.rho[::-1],self.s[::-1],self.gradad[::-1]],
                names=('R_RTOT','M_MTOT','R_CM','M_G','P_CGS','T_K','RHO_GCC','S_CGS','GRADAD'))
        output.write("output/model.csv",overwrite=True)

    def plot(self):
        jup_ref = self.read_guess_model()
        fig = plt.figure(figsize=(14, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.r / 1e5, self.rho,label='MOONS')
        #plt.plot(jup_ref['R_CM']/1e5,jup_ref['RHO_GCC'],label='CEPAM')
        plt.legend()
        plt.title("Density (g/cc)")
        plt.xlabel("Radius (km) (depth)")
        plt.subplot(1, 3, 2)
        plt.plot(self.p/1e6,self.t,lw=1)
        #plt.plot(jup_ref['P_CGS']/1e6,jup_ref['T_K'])
        plt.title("Temperature (K)")
        plt.xlabel("Pressure (bar)")
        #plt.xscale('log')
        plt.subplot(1, 3, 3)
        #plt.plot(self.r / 1e5, self.m / 1e24)
        #plt.plot(jup_ref['R_CM']/1e5,jup_ref['# M_MTOT']*1.89861120e+6)
        #plt.title("Cumulated mass ($10^{24}$ g)")
        #plt.xlabel("Radius (km)")
        plt.scatter(self.r/self.r[-1],self.m/self.m[-1],s=2)
        #plt.scatter(jup_ref['R_RTOT'],jup_ref['# M_MTOT'],s=1,alpha=0.2)
        plt.suptitle(f"Internal structure of {self.name}")
        plt.tight_layout()
        #plt.show()
        #fig.savefig("Figures/CMS19_comparison.pdf",bbox_inches="tight")
