import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def polytrope(pressure, n, K):
    """
    Polytrope used to reproduce Jupiter
    """
    return (pressure / K) ** (n / (n + 1))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def hm1989_rocks(pressure,roches):
    """
    Analytical formula from Hubbard & Marley 1989.
    I copied the routine from CEPAM (etat_noy.f, pro_rock.f and pro_ice.f).
    Since HM1989 is giving P(rho), we get rho(P) using a Newton method to do the root-finding.
    """
    prec=1.e-8
    err=1.e0
    pmbar = pressure/1.e12
    if roches:
        rho1 = 20.
        while err > prec:
            rho = rho1 - (pro_rock(rho1)-pmbar)/dpro_rock(rho1)
            err = abs((rho-rho1)/rho1)
            rho1 = rho
    else:
        print("Using the icy analytical formula from HM1989")
        rho1 = 10.
        while err > prec:
            rho = rho1 - (pro_ice(rho1)-pmbar)/dpro_ice(rho1)
            err = abs((rho-rho1)/rho1)
            rho1 = rho
    return rho
    
def hm1989_rocks_vec(pressures, roches):
    """
    Needed because I am giving an array of pressure values, not only one single P value.
    """
    pressures = np.atleast_1d(pressures)
    results = np.zeros_like(pressures)
    for i, p in enumerate(pressures):
        results[i] = hm1989_rocks(p, roches)
    return results

def pro_rock(rho):
    return (rho**4.40613)*np.exp(-6.57876-0.176368*rho+0.00202239*rho*rho)
    
def dpro_rock(rho):
    return (4.40613/rho-0.176368+0.00202239*2*rho)*(rho**4.40613)*np.exp(-6.57876-0.176368*rho+0.00202239*rho*rho)
    
def pro_ice(rho):
    return (rho**3.71926)*np.exp(-2.75591-0.271321*rho+0.00700925*rho*rho)
    
def dpro_ice(rho):
    return (3.71926/rho-0.271321+0.00700925*2*rho)*(rho**3.71926)*np.exp(-2.75591-0.271321*rho+0.00700925*rho*rho)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
