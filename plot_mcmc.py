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
J2 = np.array([b["J2"] for b in data["blobs"]])
C22 = np.array([b["C22"] for b in data["blobs"]])
#A = np.array([b["A"] for b in data["blobs"]])

Mcore = params[:, 0]
#rho_core = params[:, 1]
Mmantle = Mcore + (1-Mcore)*params[:, 1]
#rho_mantle = params[:, 3]
Mocean = Mmantle + (1-Mmantle)*params[:, 2]
#rho_ocean = params[:, 5]

#samples = np.column_stack([Mcore,rho_core,Mmantle,rho_mantle,Mocean,rho_ocean,
#R_cm/1e5,NMoI,J2*1e6,C22*1e6])
samples = np.column_stack([Mcore,Mmantle,Mocean,
R_cm/1e5,NMoI,J2*1e6,C22*1e6])

labels = [
    r"$M_{\rm core}$",
#    r"$\rho_{\rm core}$ (gcc)",
    r"$M_{\rm mantle}$",
#    r"$\rho_{\rm mantle} (gcc)$",
    r"$M_{\rm ocean}$",
#    r"$\rho_{\rm ocean} (gcc)$",
#    r"$\sigma_{\rm ocean} (S/m)$",
    r"$R$ (km)",
    r"NMoI",
    r"$J_2 \times 10^6$",
    r"$C_{22} \times 10^6$",
#    r"A",
]

fig = corner.corner(
    samples,
    plot_contours=True,
    plot_datapoints=True,
    #smooth=True,
    plot_density=False,
    fill_contours=True,
    labels=labels,
    labelpad=0.1,
    show_titles=True,
    title_fmt=".3g",
    quantiles=[0.16, 0.5, 0.84],
    color = 'cornflowerblue'
)

plt.show()
fig.savefig("Figures/X_mcmc_ganymede.pdf", bbox_inches="tight")
