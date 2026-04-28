import numpy as np
import matplotlib.pyplot as plt
import time
import emcee
from collections import OrderedDict
from moons import SatelliteModel

# Initial parameters - - - - - - - - - -  I need to put them in a separate file and read it from here.

G = 6.673848e-8
GM = 3202.72e15 #km^3/s^2 en cm^3/s^2
Mobject = GM/G
Name = "Europa"
P_surf=1e6
T_surf=110.
Prot = 3.55 #days
#sigma_ocean = 1 #S.m^-1
rho_Ih = 920. #kg/m3
#rho_core = 5
#rho_rock = 3
#rho_ocean = 1.1

param_priors = OrderedDict([
    ('mnoyau', {'type': 'uniform','bounds': (0.0, 0.49)}),
    #('rho_core', {'type': 'gaussian','mu': 5.,'sigma': 0.05}),
    ('rho_core', {'type': 'uniform','bounds': (5.15, 8.)}),
    ('mmanteau', {'type': 'uniform','bounds': (0.7, 0.96)}),
    ('rho_mantle', {'type': 'uniform','bounds': (2.5, 4.)}),
    ('log_sigma_ocean', {'type': 'uniform','bounds': (-3, 1)}),
    ('P_Ih', {'type': 'uniform','bounds': (14., 206.)}),
])

# Data
data = OrderedDict([
('R_cm', [1565.e5, 8e5]),
('NMoI', [0.346, 0.005]),
('J2', [435.5e-6, 8.2e-6]),
('C22', [131.5e-6, 2.5e-6]),
('A', [0.97, 0.02]),
])

# - - - - - - - - - - - - - - - -
# Forward model
# - - - - - - - - - - - - - - - -
def forward_model(params):
    params_dict = {name: val for name, val in zip(param_priors.keys(), params)}
    Mcore = params_dict['mnoyau']
    # épaisseurs des couches, et non les interfaces...
    #m2 = params_dict['mmanteau']
    #Mmantle = Mcore + (1-Mcore)*m2
    Mmantle = params_dict['mmanteau']
    rho_rock = params_dict['rho_mantle']
    rho_core = params_dict['rho_core']
    sigma_ocean = 10**params_dict['log_sigma_ocean']
    P_Ih = params_dict['P_Ih']
    
    layers = [
        {"name": "Noyau", "mass": Mobject*Mcore, "eos": "constant_density","constant_rho":rho_core,"T_struct":"isentrope"},
        {"name": "Manteau", "mass": Mobject*Mmantle, "eos": "constant_density","constant_rho":rho_rock,"T_struct":"isentrope"},
        {"name": "Ocean", "mass": Mobject, "eos": "h2o_phasediag","rho_Ih":rho_Ih,"P_Ih":P_Ih,"T_struct":"isentrope"},
    ]
    object = SatelliteModel(Name, Mobject, Prot, sigma_ocean, layers, nlayers=2000, distribution_type='erf')
    object.integrate_structure_iterate(max_iter=100,rtol=1e-5,debug=False,P_surf=P_surf,T_surf=T_surf,from_scratch=False,save=False)

    R_cm, MoI = object.moment_of_inertia()
    NMoI = MoI/(Mobject*(R_cm)**2)
    J2 = object.call_clairaut()
    C22 = J2*3/10
    A, phi, R_Ih, D_ocean = object.call_mag_ind()
    model_dict = {'R_cm':R_cm, 'NMoI':NMoI, 'J2':J2 ,'C22':C22,'A':A}
    return model_dict

# - - - - - - - - - - - - - - - -
# Likelihood
# - - - - - - - - - - - - - - - -
def log_likelihood(params, data):
    #print(params)
    model = forward_model(params)
    logL = 0.0
    for key, (obs, sigma) in data.items():
        if key not in model:
            raise KeyError(f"forward_model ne renvoie pas '{key}'")
        mod = model[key]
        if not np.isfinite(mod):
            #print("un NaN apparaît !")
            return -np.inf, model
        logL += -0.5 * ((mod - obs)**2 / sigma**2)
    return logL, model

# - - - - - - - - - - - - - - - -
# Priors
# - - - - - - - - - - - - - - - -
def log_prior(params, param_priors):
    logp = 0.0
    for value, (name, prior) in zip(params, param_priors.items()):
        if prior["type"] == "uniform":
            low, high = prior["bounds"]
            if not (low <= value <= high):
                return -np.inf
        elif prior["type"] == "gaussian":
            mu = prior["mu"]
            sigma = prior["sigma"]
            logp += -0.5 * ((value - mu)/sigma)**2
        else:
            raise ValueError(f"Type de prior inconnu : {prior['type']}")
    return logp

# - - - - - - - - - - - - - - - -
# Posterior
# - - - - - - - - - - - - - - - -
def log_posterior(params, param_priors, data):
    lp = log_prior(params, param_priors)
    if not np.isfinite(lp):
        return -np.inf, None
    logL, data_to_return = log_likelihood(params, data)
    return lp + logL, data_to_return

# - - - - - - - - - - - - - - - -
# Running the MCMC
# - - - - - - - - - - - - - - - -
def generate_initial_walkers(nwalkers, param_priors):
    p0 = []
    for i in range(nwalkers):
        walker = []
        for name, prior in param_priors.items():
            if prior["type"] == "uniform":
                low, high = prior["bounds"]
                walker.append(np.random.uniform(low, high))
            elif prior["type"] == "gaussian":
                walker.append(np.random.normal(prior["mu"], prior["sigma"]))
        p0.append(walker)
    return np.array(p0)

ndim = len(param_priors)
nwalkers = 16
N_burnin = 400
N_prod = 4000
p0 = generate_initial_walkers(nwalkers, param_priors)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,args=(param_priors, data))

print(" - - - - - Burn-in with", N_burnin, "iterations")
p0 = sampler.run_mcmc(p0, N_burnin, progress=True)[0]
sampler.reset()
print(" - - - - - Production with", N_prod, "iterations")
sampler.run_mcmc(p0, N_prod, progress=True)
samples = sampler.get_chain(flat=True)
data_final = sampler.get_blobs(flat=True)
np.savez("output/mcmc_results.npz",params=samples,blobs=data_final)
print("Data has been saved.")

