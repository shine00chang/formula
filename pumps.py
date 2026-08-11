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
    power = lambda lpm: 6 * 12 if lpm > 26.66 else (3 + 3 * lpm / 26.66) *12;

pump_list = [boschePCE1]


# datasheet: https://www.bosch-ibusiness.com/en-2/products/product-categories/auxiliary-pumps-and-valves/pce-039202453f/pce-039202453f.pdf
class boschePCE2(Pump):
    flow = _flow_fit(np.array([
        [0, 0.78],
        [12.5, 0.8],
        [41.6, 0.28]
    ]))
    power = lambda lpm: 5 * 12 if lpm > 13.3 else (3 + 2 * lpm / 13.3) * 12;


# https://www.topsflo.com/special-pumps/electric-auto-circulation-pump.html
# https://daviescraig.com.au/product/ebp40-electric-booster-pump-12v-9040
class ebp40(Pump):
    flow = _flow_fit(np.array([
        [0, 0.86],
        [5, 0.8],
        [10, 0.73],
        [15, 0.62],
        [20, 0.52]
    ]))
    power = lambda lpm: 36 + lpm * 12 / 30


# https://www.brushlessdcpump.com/product/vp62a/
class vp62a(Pump):
    flow = _flow_fit(np.array([
        [0, 0.9],
        [7.5, 0.75],
        [17, 0.6]
    ]))
    power = _flow_fit(np.array([
        [0, 38],
        [5, 42],
        [10, 46]
    ]))


class dc56e(Pump):
    flow = _flow_fit(np.array([
        [0, 0.1],
        [10, 0.8],
        [17.5, 0.6]
    ]))
    power = lambda x: 4.5 * 12

pump_list = [boschePCE1, boschePCE2, ebp40, vp62a, dc56e]
