import numpy as np
from scipy.optimize import curve_fit, fsolve

def _quad_model(x, a, b):
    """Model for curve fitting: y = a*x^2 + b"""
    return a * x**2 + b

def _flow_fit(pumpsheet):
    # Fit the pumpsheet data to the quadratic model
    popt, _ = curve_fit(_quad_model, pumpsheet[:, 0], pumpsheet[:, 1])
    # Return a lambda function that calculates the pump pressure using fitted parameters
    return lambda lpm: _quad_model(lpm, *popt)

class Pump:
    """
    Base class that calculates static mathematical attributes for all of 
    its subclasses automatically upon their creation.
    """
    # Fallback base attributes
    flow = lambda: 0
    power = lambda: 0
    
# datasheet: https://www.bosch-ibusiness.com/en-2/products/product-categories/auxiliary-pumps-and-valves/pce-0392024058/pce-0392024058.pdf
class boschePCE1(Pump):
    flow = _flow_fit(np.array([
        [0, 0.76],
        [6.66, 0.75],
        [13.3, 0.73],
        [20, 0.67],
        [26.7, 0.5],
    ]))
    power = lambda lpm: 6 * 12 if lpm > 33.3 else (3 + 0.09 * lpm) *12;

pump_list = [boschePCE1]
