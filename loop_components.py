import numpy as np
from scipy.optimize import curve_fit, fsolve

def _quad_model(x, a, b):
    """Model for curve fitting: y = a*x^2 + b"""
    return a * x**2 + b

def _pumpmaker(pumpsheet):
    # Fit the pumpsheet data to the quadratic model
    popt, _ = curve_fit(_quad_model, pumpsheet[:, 0], pumpsheet[:, 1])
    # Return a lambda function that calculates the pump pressure using fitted parameters
    return lambda lpm: _quad_model(lpm, *popt)

# BOSCHE PCE (mk11, free one)
# datasheet: https://www.bosch-ibusiness.com/en-2/products/product-categories/auxiliary-pumps-and-valves/pce-0392024058/pce-0392024058.pdf
boschePCE = _pumpmaker(np.array([
    [0, 0.76],
    [6.66, 0.75],
    [13.3, 0.73],
    [20, 0.67],
    [26.7, 0.5],
]))
boschePCEpower = lambda lpm: 6 * 12 if lpm > 33.3 else (3 + 0.09 * lpm) *12;

def emrax228(lpm):
    a = 0.5 / (6 * 6)
    return a * lpm * lpm

def cm200dx(lpm):
    a = 0.3 / (12 * 12)
    return a * lpm * lpm


print("note: blackice 280mm data from 2.21.2026");
def blackice_280(lpm):
    a = 0.07 * 0.07
    return a * lpm * lpm


