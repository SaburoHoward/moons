import numpy as np
import matplotlib.pyplot as plt
from moons import SatelliteModel
import time
import csv

#Merci de commencer par les couches les plus INTERNES. Rayon en cm et densité en g/cc.

G = 6.673848e-8
GM = 3202.72e15 #km^3/s^2 en cm^3/s^2

Prot = 3.55 #days
M_europa = GM/G
#sigma_ocean = 1e-1 #S.m^-1
sigma_ocean = 10**(-1.55)
rho_Ih = 920. #kg/m3
P_Ih = np.random.uniform(14, 206) #J'évite d'être trop près des limites, on sait jamais...)
P_Ih = 180
#P_Ih = 104
T_surf = 110 #on sait pas pour Europa je crois...

layers_europa = [
    {"name": "Noyau", "mass": M_europa*0.3, "eos": "constant_density","constant_rho":6.,"T_struct":"isentrope"},
    {"name": "Manteau", "mass": M_europa*0.93, "eos": "constant_density","constant_rho":3.5,"T_struct":"isentrope"},
    {"name": "Ocean", "mass": M_europa, "eos": "h2o_phasediag","rho_Ih":rho_Ih,"P_Ih":P_Ih,"T_struct":"isentrope"},
#    {"name": "Ice shell", "mass": M_europa, "eos": "constant_density","constant_rho":0.9,"T_struct":"isentrope"},
]

t0 = time.time()
europa = SatelliteModel("Europa", M_europa, Prot, sigma_ocean, layers_europa, nlayers=2000, distribution_type='erf')
europa.integrate_structure_iterate(max_iter=100,rtol=1e-5,debug=False,P_surf=1e6,T_surf=T_surf,from_scratch=False,save=True)
R_cm, MoI = europa.moment_of_inertia()
print("Radius =",R_cm/1e5,"km")
print("NMoI =",MoI/(M_europa*(R_cm)**2))
J2 = europa.call_clairaut()
print("J2*1e6 =",J2*1e6)
print("C22*1e6 =",J2*1e6*3/10)
A, phi, R_Ih, D_ocean = europa.call_mag_ind()
print("A =",A,"phi =",phi)
print("D_Ih =",R_cm/1e5-R_Ih,"km,", "D_ocean =",D_ocean,"km")

t1 = time.time()
print("- - - Temps écoulé :", t1 - t0, "s")
#ganymede.plot()
headers = ["Radius_km", "NMoI", "J2e6", "C22e6", "A", "Phi", "D_Ih_km","D_ocean_km"]
data = [R_cm/1e5,MoI/(M_europa*(R_cm)**2),J2*1e6,J2*1e6*3/10,A,phi,R_cm/1e5-R_Ih,D_ocean]
with open("output/model_js.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerow(data)
