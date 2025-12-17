import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import erf

def read_guess_model():
    """to start from/or compare to an existing model."""
    #model = ascii.read("JupiterModel/jup_howard23.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
    #model = ascii.read("JupiterModel/Jupiter_Z0/jup_pureH-He.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
    #model = ascii.read("JupiterModel/Jupiter_Z0/jup_RockyCore1percent.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
    model = ascii.read("JupiterModel/Polytrope/jup_1000.csv",format="csv",guess=False,fast_reader={'exponent_style': 'D'})
    return model

def mass_distrib(distribution_type,mass,nlayers,p1,p2):
    """defines the selected radii mesh"""
    if distribution_type == "linear":
        m = np.linspace(0, mass, nlayers)
    elif distribution_type == "log":
        m = np.logspace(0, np.log10(mass), nlayers)
    elif distribution_type == "exp":
        mesh_surf_amp = p1 #increase to have more layers near surface
        mesh_surf_width = p2 #controls the width of the refined region
        f = np.linspace(0, 1, nlayers)
        refinement = 1 + mesh_surf_amp * f * np.exp((f - 1) / mesh_surf_width) #densité de points dans l'espace normalisé.
        m_norm = np.cumsum(1 / refinement) #1/refinement = espacement local entre 2 points. Cumsum pour cumul des espacements locaux.
        m_norm -= m_norm[0] #to start at 0
        m_norm /= m_norm[-1] #to get r[-1]=1
        m = m_norm * mass
    elif distribution_type == "erf":
        xi = np.linspace(-p1 + p2, p1 + p2, nlayers)
        mesh = 0.5 * (1 + erf(xi))
        m = (mesh - mesh[0]) / (mesh[-1] - mesh[0])
    elif distribution_type == "debras":
        beta = 6.0 / nlayers
        i = np.arange(nlayers)
        lambdas = 1.0 - (np.exp(i * beta) - 1.0) / (np.exp(nlayers * beta) - 1.0)
        m = lambdas[::-1] * mass
    elif distribution_type == "chatGPT":
        a = 5
        s = np.linspace(0, 1, nlayers)
        m = np.tanh(a * s) / np.tanh(a)
    elif distribution_type == "claude":
        beta = 5.0
        xi = np.linspace(-1, 1, nlayers)
        m = 0.5 * (1 + np.tanh(beta * xi) / np.tanh(beta))
    elif distribution_type == "claude2":
        xi = np.linspace(0, 1, nlayers)
        if p1 == 0 and p2 == 0:
            m = xi
        elif p1 == 0:
            m = 1 - 0.5 * (1 + np.tanh(p2 * (1 - 2*xi)) / np.tanh(p2))
        elif p2 == 0:
            m = 0.5 * (1 + np.tanh(p1 * (2*xi - 1)) / np.tanh(p1))
        else:
            t = 2 * xi - 1  # [-1, 1]
            tanh_0 = np.tanh(p1 * t) / np.tanh(p1)
            tanh_1 = np.tanh(p2 * t) / np.tanh(p2)
            weight = 0.5 * (1 + t)
            combined_tanh = (1 - weight) * tanh_0 + weight * tanh_1
            
            m = 0.5 * (1 + combined_tanh)
    else:
        m = np.linspace(1, mass, nlayers)
        print('default is linear and is probably bad')
    return m/m[-1]

nlayers=1000
mass = 1.8982532e30
fig = plt.figure(figsize=(8,8))
#plt.scatter(np.linspace(1,nlayers,nlayers),mass_distrib("linear",mass,nlayers,0,0),label="linear",s=1)
#plt.scatter(np.linspace(1,nlayers,nlayers),mass_distrib("log",mass,nlayers,0,0),label="log",s=1)
#plt.scatter(np.linspace(1,nlayers,nlayers),mass_distrib("exp",mass,nlayers,1e5,5e-2),label="exp",s=1)
#plt.scatter(np.linspace(1,nlayers,nlayers),mass_distrib("exp_surf_cent",mass,nlayers,1e5,5e-2),label="exp_surf_cent",s=1)
#plt.scatter(np.linspace(1,nlayers,nlayers),mass_distrib("debras",mass,nlayers,0,0),label="debras",s=1)
#plt.scatter(np.linspace(1,nlayers,nlayers),mass_distrib("chatGPT",mass,nlayers,1e3,5e-2),label="chatGPT",s=1)
#plt.scatter(np.linspace(1,nlayers,nlayers),mass_distrib("claude",mass,nlayers,1e3,5e-2),label="claude",s=1)
#plt.scatter(np.linspace(1,nlayers,nlayers),mass_distrib("claude2",mass,nlayers,7.5,2.5),label="claude2",s=1)
model = read_guess_model()
plt.scatter(np.linspace(1,nlayers,nlayers),model["# M_MTOT"].data[::-1],label="cepam",s=1)
plt.scatter(np.linspace(1,nlayers,nlayers),mass_distrib("erf",mass,nlayers,2.5,0.6),label="erf",s=1)
plt.scatter(np.linspace(1,nlayers,nlayers),mass_distrib("erf",mass,nlayers,2.7,0.5),label="erf2",s=1)
plt.xlim(0,100)
plt.ylim(0,0.05)
plt.legend()
plt.show()
