import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import erf
import corner

data = np.load("output/mcmc_results.npz", allow_pickle=True)

params = data["params"]
R_cm = np.array([b["R_cm"] for b in data["blobs"]])
NMoI = np.array([b["NMoI"] for b in data["blobs"]])

samples = np.column_stack([params[:, 0],params[:, 1],params[:, 2],
R_cm/1e5,NMoI])

labels = [
    r"$M_{\rm core}$",
    r"$\rho_{\rm rocks}$",
    r"$M_{\rm mantle}$",
    r"$R$ (km)",
    r"NMoI"
]

fig = corner.corner(
    samples,
    labels=labels,
    show_titles=True,
    title_fmt=".3g",
    quantiles=[0.16, 0.5, 0.84]
)

plt.show()
