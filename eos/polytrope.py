import numpy as np

def polytrope(pressure, n, K):
    return (pressure / K) ** (n / (n + 1))
