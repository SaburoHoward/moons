import numpy as np
import matplotlib.pyplot as plt
import time
import emcee
from collections import OrderedDict
from moons import SatelliteModel

# Initial parameters - - - - - - - - - -  I need to put them in a separate file and read it from here.
Mplanet = 1.8982532e30
Name = "Jupiter"
P_surf=1e6
T_surf=165.

param_priors = OrderedDict([
    ('mnoyau', {'type': 'uniform','bounds': (0.0, 0.1)}),
#    ('mmanteau', {'type': 'gaussian','mu': 0.3,'sigma': 0.05}),
#    ('Rnoyau', {'type': 'uniform','bounds': (1e5, 2e5)}),
])

# Data
data = OrderedDict([
('R_cm', [69911e5, 6e5]),
#('NMoI', [0.263, 0.001]),
#('J2', [71492e5, 6e5])
])

# - - - - - - - - - - - - - - - -
# Forward model
# - - - - - - - - - - - - - - - -
def forward_model(params):
    params_dict = {name: val for name, val in zip(param_priors.keys(), params)}
    Mcore = params_dict['mnoyau']
    layers = [{"name": "Manteau", "mass": Mplanet*Mcore, "eos": "hm1989_rocks","roches": True,"T_struct":"isotherm"},
              {"name": "Envelope", "mass": Mplanet, "eos":"mixture","nbelem":2,"files":["Chabrier2019-H.csv","Chabrier2019-He.csv"],"mass_fractions":[0.73,0.27],"T_struct":"isentrope"}]
    jupiter = SatelliteModel(Name, Mplanet, layers, nlayers=1000, distribution_type='exp')
    jupiter.integrate_structure_iterate(max_iter=50,rtol=1e-4,debug=False,P_surf=P_surf,T_surf=T_surf)

    R_cm, MoI = jupiter.moment_of_inertia()
    NMoI = MoI/(Mplanet*(R_cm)**2)
    model_dict = {'R_cm':R_cm, 'NMoI':NMoI}
    return model_dict

# - - - - - - - - - - - - - - - -
# Likelihood
# - - - - - - - - - - - - - - - -
def log_likelihood(params, data):
    model = forward_model(params)
    logL = 0.0
    for key, (obs, sigma) in data.items():
        if key not in model:
            raise KeyError(f"forward_model ne renvoie pas '{key}'")
        mod = model[key]
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
nwalkers = 4
N_burnin = 10
N_prod = 10
p0 = generate_initial_walkers(nwalkers, param_priors)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,args=(param_priors, data))

print(" - - - - - Burn-in with", N_burnin, "iterations")
p0 = sampler.run_mcmc(p0, N_burnin, progress=True)[0]
sampler.reset()
print(" - - - - - Production with", N_prod, "iterations")
sampler.run_mcmc(p0, N_prod, progress=True)
samples = sampler.get_chain(flat=True)
data_final = sampler.get_blobs(flat=True)
np.savez("mcmc_results.npz",params=samples,blobs=data_final)
print("Data has been saved.")

