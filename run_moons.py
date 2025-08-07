import numpy as np
import matplotlib.pyplot as plt
from moons import SatelliteModel

#Merci de commencer par les couches les plus externes. Rayon en cm et densité en g/cc.
layers_europe = [
    {"name": "Glace", "radius": 1560e5, "density": 1},
    {"name": "Manteau", "radius": 1460e5, "density": 3},
    {"name": "Noyau", "radius": 400e5, "density": 5},
]

#europe = SatelliteModel("Europa", 1560e5, layers_europe, nlayers=100)
#europe.integrate_structure()
#europe.integrate_structure_iterate()
#europe.plot()

layers_jupiter = [
    {"name": "Envelope", "radius": 71492e5, "density": 0.01},
    {"name": "Noyau", "radius": 10000e5, "density": 10},
]

jupiter = SatelliteModel("Jupiter", 71492e5, layers_jupiter, nlayers=1014, distribution_type='exp')
jupiter.integrate_structure_iterate(max_iter=40,rtol=1e-4,debug=False,P_surf=1e6,T_surf=165.)
#MoI = jupiter.moment_of_inertia()
#print(MoI/(1.898e30*(71492e5)**2))
jupiter.plot()
