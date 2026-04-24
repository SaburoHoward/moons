import numpy as np
import matplotlib.pyplot as plt
from moons import SatelliteModel
import time
import csv

#Merci de commencer par les couches les plus INTERNES. Rayon en cm et densité en g/cc.

G = 6.673848e-8
GM = 9887.83e15 #km^3/s^2 en cm^3/s^2

Prot = 7.15 #days
M_ganymede = GM/G
sigma_ocean = 1e-1 #S.m^-1
rho_Ih = 920. #kg/m3
P_Ih = np.random.uniform(14, 206) #J'évite d'être trop près des limites, on sait jamais...)

layers_ganymede = [
    {"name": "Noyau", "mass": M_ganymede*0.1, "eos": "constant_density","constant_rho":5.,"T_struct":"isentrope"},
    {"name": "Manteau", "mass": M_ganymede*0.75, "eos": "constant_density","constant_rho":3.,"T_struct":"isentrope"},
    {"name": "Ocean", "mass": M_ganymede, "eos": "h2o_phasediag","rho_Ih":rho_Ih,"P_Ih":P_Ih,"T_struct":"isentrope"},
#    {"name": "Ice shell", "mass": M_ganymede, "eos": "constant_density","constant_rho":0.9,"T_struct":"isentrope"},
]

t0 = time.time()
ganymede = SatelliteModel("Ganymede", M_ganymede, Prot, sigma_ocean, layers_ganymede, nlayers=2000, distribution_type='erf')
ganymede.integrate_structure_iterate(max_iter=100,rtol=1e-5,debug=False,P_surf=1e6,T_surf=110.,from_scratch=False,save=True)
R_cm, MoI = ganymede.moment_of_inertia()
print("Radius =",R_cm/1e5,"km")
print("NMoI =",MoI/(M_ganymede*(R_cm)**2))
J2 = ganymede.call_clairaut()
print("J2*1e6 =",J2*1e6)
print("C22*1e6 =",J2*1e6*3/10)
A, phi, R_Ih, D_ocean = ganymede.call_mag_ind()
print("A =",A,"phi =",phi)
print("D_Ih =",R_cm/1e5-R_Ih,"km,", "D_ocean =",D_ocean,"km")

t1 = time.time()
print("- - - Temps écoulé :", t1 - t0, "s")
#ganymede.plot()
headers = ["Radius_km", "NMoI", "J2e6", "C22e6", "A", "Phi", "D_Ih_km","D_ocean_km"]
data = [R_cm/1e5,MoI/(M_ganymede*(R_cm)**2),J2*1e6,J2*1e6*3/10,A,phi,R_cm/1e5-R_Ih,D_ocean]
with open("output/model_js.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerow(data)
