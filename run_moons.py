import numpy as np
import matplotlib.pyplot as plt
from moons import SatelliteModel
import time

#Merci de commencer par les couches les plus INTERNES. Rayon en cm et densité en g/cc.
#Veillez à ce que la masse de la dernière couche soit similaire à la masse totale de la planète...

#Jupiter w/ a polytrope
layers_jupiter = [
    {"name": "Noyau", "mass": 1.898e28, "eos": "polytrope", "n": 1., "K": 2.003565e12},
    {"name": "Manteau", "mass": 1.898e29, "eos": "polytrope", "n": 1, "K": 2.003565e12},
    {"name": "Envelope", "mass": 1.8982532e30, "eos": "polytrope", "n": 1., "K": 2.003565e12},
]

#Jupiter as a pure H-He ball (CMS19 EOS)

layers_jupiter = [
    {"name": "Noyau", "mass": 1.898e28, "eos": "mixture","nbelem":2,"files":["Chabrier2019-H.csv","Chabrier2019-He.csv"],"mass_fractions":[0.73,0.27]},
    {"name": "Manteau", "mass": 1.898e29, "eos": "mixture","nbelem":2,"files":["Chabrier2019-H.csv","Chabrier2019-He.csv"],"mass_fractions":[0.73,0.27]},
    {"name": "Envelope", "mass": 1.8982532e30, "eos": "mixture","nbelem":2,"files":["Chabrier2019-H.csv","Chabrier2019-He.csv"],"mass_fractions":[0.73,0.27]},
]


t0 = time.time()
jupiter = SatelliteModel("Jupiter", 1.8982532e30, layers_jupiter, nlayers=1000, distribution_type='exp')
jupiter.integrate_structure_iterate(max_iter=50,rtol=1e-4,debug=False,P_surf=1e6,T_surf=165.)
#MoI = jupiter.moment_of_inertia()
#print(MoI/(1.898e30*(71492e5)**2))
t1 = time.time()
print("Temps écoulé :", t1 - t0, "s")
jupiter.plot()
