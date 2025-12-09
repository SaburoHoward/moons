import numpy as np
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import RectBivariateSpline as rbs

def pure_eos(file,method):
    """
    Reads the table of an EOS (in Cepam format: LOGP,LOGT,LOGRHO,LOGS)
    And returns the table, and the interpolation functions for LOGRHO and LOGS
    I now use RBS instead of RGI for computation time reasons.
    """
    table = Table.read("eos/data/"+file, format='csv')
    P_grid = np.unique(table["LOGP"])
    T_grid = np.unique(table["LOGT"])
    rho_flat = np.array(table["LOGRHO"])
    rho_grid = rho_flat.reshape(len(P_grid), len(T_grid))
    interp_rho = rbs(P_grid,T_grid,rho_grid)
    s_flat = np.array(table["LOGS"])
    s_grid = s_flat.reshape(len(P_grid), len(T_grid))
    interp_s = rbs(P_grid,T_grid,s_grid)
    return interp_rho,interp_s
    
def linear_mixing(P_list,T_list,nbelem,mass_fractions,svpk):
    """Returns density and entropy of the mixture"""
    rho_mixt = np.zeros_like(P_list)
    s_mixt = np.zeros_like(P_list)
    st_mixt = np.zeros_like(P_list)
    sp_mixt = np.zeros_like(P_list)
    for i in range(nbelem):
        interp_rho,interp_s = svpk[i]
        rho = interp_rho(P_list,T_list,dx=0, dy=0,grid=False)
        s = interp_s(P_list,T_list,dx=0, dy=0,grid=False)
        sp = interp_s(P_list,T_list,dx=1, dy=0,grid=False) #dlogS/dlogP
        st = interp_s(P_list,T_list,dx=0, dy=1,grid=False) #dlogS/dlogT
        if mass_fractions[i] > 0.:
            rho_mixt += mass_fractions[i]/(10**rho)
            s_mixt += mass_fractions[i]*(10**s)
            st_mixt += mass_fractions[i]*(10**s)*st
            sp_mixt += mass_fractions[i]*(10**s)*sp
    rho_mixt = 1./rho_mixt #linear mixing on 1/rho, not rho.
    gradad = -sp_mixt/st_mixt
    return rho_mixt,s_mixt,gradad
