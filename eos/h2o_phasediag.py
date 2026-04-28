import numpy as np
import matplotlib.pyplot as plt
from seafreeze import seafreeze as sf
from scipy.optimize import brentq
from astropy.table import Table
from scipy.interpolate import RectBivariateSpline as rbs

def melt_T_Ih(T):
    """
    Wagner et al. 2011
    Range of validity: 251.165 - 273.16 K
    """
    a1 = 0.119539337e7
    a2 = 0.808183159e5
    a3 = 0.333826860e4
    b1 = 3.
    b2 = 0.257500e2
    b3 = 0.103750e3
    Tt = 273.16
    Pt = 611.657e-6 #MPa
    theta = T/Tt
    Pmelt = Pt*(1+a1*(1-theta**b1)+a2*(1-theta**b2)+a3*(1-theta**b3))
    return Pmelt

def melt_T_III(T):
    """
    Wagner et al. 2011
    Range of validity: 251.165 - 256.164 K
    P va de 208.566 à 345.421
    """
    Tt = 251.165
    Pt = 208.566 #MPa
    theta = T/Tt
    Pmelt = Pt*(1-0.299948*(1-theta**60))
    return Pmelt
    
def melt_T_V(T):
    """
    Wagner et al. 2011
    Range of validity: 256.164 - 273.31 K
    P va de 350.1 à 631.46
    """
    Tt = 256.164
    Pt = 350.1 #MPa
    theta = T/Tt
    Pmelt = Pt*(1-1.18721*(1-theta**8))
    return Pmelt
    
def melt_T_VI(T):
    """
    Wagner et al. 2011
    Range of validity: 273.31 - 355 K
    P va de 632.4 à 2213.364
    """
    Tt = 273.31
    Pt = 632.4 #MPa
    theta = T/Tt
    Pmelt = Pt*(1-1.07476*(1-theta**4.6))
    return Pmelt
    
def T_melt_from_P(P,phase):
    """
    This function does the opposite. It returns the melting temperature at a given pressure using an optimization method
    """
    if phase == 'Ih':
        T_min = 251.165
        T_max = 273.16
        f = lambda T: melt_T_Ih(T) - P
    if phase == 'III':
        T_min = 251.165
        T_max = 256.164
        f = lambda T: melt_T_III(T) - P
    if phase == 'V':
        T_min = 256.164 #255.786, j'ai testé à la main et trouvé cette valeur pour obtenir Pmin=345.21MPa
        T_max = 273.31
        f = lambda T: melt_T_V(T) - P
    if phase == 'VI':
        T_min = 273.31
        T_max = 355
        f = lambda T: melt_T_VI(T) - P
    return brentq(f, T_min, T_max)
    
def sf_create_interp(file,phase):
    """
    Create interpolators from the SeaFreeze tables I created.
    For water1, it provides rho, alpha and Cp.
    For other phases, it only provides rho.
    """
    table = Table.read("eos/data/"+file, format='csv')
    P_grid = np.unique(table["P_MPa"])
    T_grid = np.unique(table["T_K"])
    rho_flat = np.array(table["density_kg/m3"])
    rho_grid = rho_flat.reshape(len(P_grid), len(T_grid))
    interp_rho = rbs(P_grid,T_grid,rho_grid)
    if phase == 'water1':
        alpha_flat = np.array(table["alpha_K-1"])
        alpha_grid = alpha_flat.reshape(len(P_grid), len(T_grid))
        interp_alpha = rbs(P_grid,T_grid,alpha_grid)
        Cp_flat = np.array(table["Cp_Jkg-1K-1"])
        Cp_grid = Cp_flat.reshape(len(P_grid), len(T_grid))
        interp_Cp = rbs(P_grid,T_grid,Cp_grid)
        return interp_rho, interp_alpha, interp_Cp
    else:
        return interp_rho
    
def Tprofile_and_density(p_Ih, pressure, rho_Ih,svpk_h2o):
    """
    1. Pick a point on the Ice Ih melting line. Pressure should be between 13.0386 and 208.5667 MPa.
    2. Get the melting temperature at this pressure.
    3. You are now in the liquid adiabatic ocean. Call seafreeze to get alpha,rho,cp at (P,T)
    4. Calculate the temperature at the next layer using Ti+1=Ti+(Pi+1-Pi)*alpha*Ti/(rho*Cp)
    5. At every layer, check if Ti+1 is still higher than the melting temp
    """
    hit_HP = False #booléen pour savoir quand on atteint une glace HP et on sort de l'océan
    density = np.zeros_like(pressure)
    temperature = np.zeros_like(pressure)
    indices_Ih = pressure <= p_Ih * 1e7 #MPa en cgs
    density[indices_Ih] = rho_Ih
    indices_sup = (pressure > p_Ih * 1e7)
    P_sel = pressure[indices_sup]
    es_gibt_ozean = True
    if P_sel.size == 0: #problème quand le manteau se rapproche trop de la surface...
        print("P_Ih (MPa) =",p_Ih, "is probably deeper than M_mantle")
        es_gibt_ozean = False
    else:
        d_sel = density[indices_sup]
        T_sel = temperature[indices_sup]
        new_p_Ih = P_sel[-1]/1e7 #je prends une valeur de mon modèle proche de p_Ih, pas exactement p_Ih...
        t_melt = T_melt_from_P(new_p_Ih,'Ih')
        t_curr = t_melt
        p_curr = new_p_Ih
        
        interp_rho_w1, interp_alpha_w1, interp_Cp_w1 = svpk_h2o['water1']
        interp_rho_III = svpk_h2o['III']
        interp_rho_V = svpk_h2o['V']
        interp_rho_VI = svpk_h2o['VI']
        
        T_sel[-1] = t_melt
        PT = np.empty((1,), dtype='object')
        for i in range(len(P_sel) - 1, 0, -1):
            if hit_HP: #on a atteint une glace HP
                PT[0] = (p_curr, t_curr)
                p_next = P_sel[i-1]/1e7
                if 208.566 < p_curr <= 345.421:
                    phase ='III'
                    interp_rho = interp_rho_III
                #elif 350.1 < p_curr < 631.46:
                elif 345.421 < p_curr <= 632.4: #j'élargis la V pour que ce soit continu
                    phase = 'V'
                    interp_rho = interp_rho_V
                elif 632.4 < p_curr < 2213.364:
                    phase = 'VI'
                    interp_rho = interp_rho_VI
                elif p_curr > 208.566:
                    #print("probably in between, P=",p_curr)
                    phase = 'V' #faut peut être changer les bornes pour que ça soit continu. C'EST FAIT.
                                #J'ai choisi V... donc rho à III/V et à V/VI est prescrit par V.
                    interp_rho = interp_rho_V
                #out_sf = sf.seafreeze(PT,phase)
                #rho = out_sf.rho
                rho = interp_rho(p_curr,t_curr,dx=0, dy=0,grid=False)
                if phase == 'III' and p_next > 345.421:
                    #car si prochaine P correspond à HP V, il faut utiliser la courbe de fusion de HP V
                    #t_next = T_melt_from_P(p_next,'V')
                    t_next = t_curr #ouais mais isotherme c'est plus simple
                elif phase == 'V' and p_next > 631.46:
                    #t_next = T_melt_from_P(p_next,'VI')
                    t_next = t_curr
                elif 345.421 < p_next < 350.1 or 631.46 < p_next < 632.4:
                    t_next = t_curr #si on est entre les phases, on est isotherme pour simplifier
                else:
                    t_next = T_melt_from_P(p_next,phase)
                d_sel[i] = rho
                T_sel[i-1] = t_next
                p_curr = p_next
                t_curr = t_next
                
            if not hit_HP: #on est dans l'océan
                PT[0] = (p_curr, t_curr)
                #out_sf = sf.seafreeze(PT,'water1')
                #alpha, rho, Cp = out_sf.alpha, out_sf.rho, out_sf.Cp
                rho = interp_rho_w1(p_curr,t_curr,dx=0, dy=0,grid=False)
                alpha = interp_alpha_w1(p_curr,t_curr,dx=0, dy=0,grid=False)
                Cp = interp_Cp_w1(p_curr,t_curr,dx=0, dy=0,grid=False)
                #print((approx_rho-rho)/rho,(approx_alpha-alpha)/alpha,(approx_Cp-Cp)/Cp)
                d_sel[i] = rho
                p_next = P_sel[i-1]/1e7
                t_next = t_curr + (P_sel[i-1]-P_sel[i])*1e-7*1e6*alpha*t_curr/rho/Cp #1e6 car 1MPa=1e6 J/m3
                T_sel[i-1] = t_next #t_next[0] if we use out_sf=sf.seafreeze...
                p_curr = p_next
                t_curr = t_next #same as 2 lines above
                if 208.566 < p_curr < 345.421:
                    if t_curr <= T_melt_from_P(p_curr,'III'):
                        #print("transition to HP III at (P,T)=",p_curr,T_melt_from_P(p_curr,'III'))
                        hit_HP = True
                        t_curr = T_melt_from_P(p_curr,'III')
                        T_sel[i-1] = t_curr
                elif 350.1 < p_curr < 631.46:
                #elif 345.421 < p_curr <= 632.4:
                    if t_curr <= T_melt_from_P(p_curr,'V'):
                        #print("transition to HP V at (P,T)=",p_curr,T_melt_from_P(p_curr,'V'))
                        hit_HP = True
                        t_curr = T_melt_from_P(p_curr,'V')
                        T_sel[i-1] = t_curr
                elif 632.4 < p_curr < 2213.364:
                    if t_curr <= T_melt_from_P(p_curr,'VI'):
                        #print("transition to HP VI at (P,T)=",p_curr,T_melt_from_P(p_curr,'VI'))
                        hit_HP = True
                        t_curr = T_melt_from_P(p_curr,'VI')
                        T_sel[i-1] = t_curr
                #elif p_curr > 208.566:
                #    print("in between, P=",p_curr) #mais pas grave car on continue dans l'océan
        if len(d_sel)>1: #il y avait un soucis lorsqu'il n'y a qu'UNE seule couche dans l'océan.
            d_sel[0] = d_sel[1]
        else:
            d_sel[0] = rho_Ih
        density[indices_sup]=d_sel
        temperature[indices_sup]=T_sel
        #for i, (rho, T, P) in enumerate(zip(density, temperature,pressure)):
        #    print(i, P/1e7, T, rho)
    return density/1e3,temperature,es_gibt_ozean
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""
#print(dir(sf))
#print(sf.phases.keys())
P = np.arange(0, 1400, 2)
T = np.arange(200, 300, 0.5)
PT = np.array([P, T], dtype='object')
out = sf.whichphase(PT)
dens_Ih = sf.seafreeze(PT,'Ih').rho
dens_II = sf.seafreeze(PT,'II').rho
dens_III = sf.seafreeze(PT,'III').rho
dens_V = sf.seafreeze(PT,'V').rho
dens_VI = sf.seafreeze(PT,'VI').rho
dens_water1 = sf.seafreeze(PT,'water1').rho
TT, PP = np.meshgrid(T,P)

fig = plt.figure()
# - - - PLOT THE DENSITY IN THE DIFFERENT PHASES - - -
for p in sf.phasenum2phase.keys():
    pi = (out == p)
    if p == 1:
        #plt.scatter(TT[pi], PP[pi],c=dens_Ih[pi], label=sf.phasenum2phase[p],cmap='viridis',vmin=920,vmax=940)
        plt.scatter(TT[pi], PP[pi],c=dens_Ih[pi], label=sf.phasenum2phase[p],cmap='GnBu',vmin=920,vmax=1400,s=1)
        plt.text(220,100,"Ih",c='k')
    if p == 2:
        #plt.scatter(TT[pi], PP[pi],c=dens_II[pi], label=sf.phasenum2phase[p],cmap='viridis',vmin=1185,vmax=1220)
        plt.scatter(TT[pi], PP[pi],c=dens_II[pi], label=sf.phasenum2phase[p],cmap='GnBu',vmin=920,vmax=1400,s=1)
        plt.text(210,380,"II",c='k')
    if p == 3:
        #plt.scatter(TT[pi], PP[pi],c=dens_III[pi], label=sf.phasenum2phase[p],cmap='viridis',vmin=1150,vmax=1170)
        plt.scatter(TT[pi], PP[pi],c=dens_III[pi], label=sf.phasenum2phase[p],cmap='GnBu',vmin=920,vmax=1400,s=1)
        plt.text(246,300,"III",c='k')
    if p == 5:
        #plt.scatter(TT[pi], PP[pi],c=dens_V[pi], label=sf.phasenum2phase[p],cmap='viridis',vmin=1245,vmax=1280)
        plt.scatter(TT[pi], PP[pi],c=dens_V[pi], label=sf.phasenum2phase[p],cmap='GnBu',vmin=920,vmax=1400,s=1)
        plt.text(250,550,"V",c='k')
    if p == 6:
        #plt.scatter(TT[pi], PP[pi],c=dens_VI[pi], label=sf.phasenum2phase[p],cmap='viridis',vmin=1350,vmax=1400)
        plt.scatter(TT[pi], PP[pi],c=dens_VI[pi], label=sf.phasenum2phase[p],cmap='GnBu',vmin=920,vmax=1400,s=1)
        plt.text(250,1000,"VI",c='k')
    if p == 0:
        #plt.scatter(TT[pi], PP[pi],c=dens_water1[pi], label=sf.phasenum2phase[p],cmap='viridis',vmin=1000,vmax=1230)
        plt.scatter(TT[pi], PP[pi],c=dens_water1[pi], label=sf.phasenum2phase[p],cmap='GnBu',vmin=920,vmax=1400,s=1)
        plt.text(275,380,"Liquid",c='k')

# - - - PLOT THE MELTING LINES FROM WAGNER+2011 - - -
T_Ih = np.arange(251.165,273.16,1)
plt.plot(T_Ih,melt_T_Ih(T_Ih),c='k',lw=1)
T_III = np.arange(251.165,256.164,0.1)
plt.plot(T_III,melt_T_III(T_III),c='k',lw=1)
T_V = np.arange(256.164,273.31,0.1)
plt.plot(T_V,melt_T_V(T_V),c='k',lw=1)
T_VI = np.arange(273.31,355,0.1)
plt.plot(T_VI,melt_T_VI(T_VI),c='k',lw=1)

plt.xlim(200,300)
plt.ylim(0,1400)
cbar = plt.colorbar(label='Density (kg/m³)')
cbar.ax.invert_yaxis()
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (MPa)')
plt.gca().invert_yaxis()
plt.title('$H_{2}O$ phase diagram')
plt.show()
"""
