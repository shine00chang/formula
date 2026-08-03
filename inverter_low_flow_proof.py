import math
import numpy as np
import matplotlib.pyplot as plt
import loop_helper as helper  # Must be present in the same directory

def lpm_to_m3s(lpm):
    """Helper to convert LPM to cubic meters per second."""
    return np.array(lpm) / 60000.0

def in_to_m(inches):
    """Helper to convert inches to meters."""
    return inches * 0.0254

def Nu(Re):
    Pr = 6.0  # water at 25C
    # Use np.where to handle numpy arrays for Re > 10000 condition
    return np.where(Re > 10000, 0.023 * (Re ** 0.8) * (Pr ** 0.4), 0.0)

def Re_calc(lpm, D):
    nu_water = 0.9 * 1e-6  # kinematic viscosity of water
    v = lpm_to_m3s(lpm) / (D * D * math.pi / 4)
    return v * D / nu_water

def h_conv(lpm, D):
    k_water = 0.61  # thermal conductivity of water at 25C
    return Nu(Re_calc(lpm, D)) * k_water / D

def pressure(lpm, L, D):
    # Ratio of areas for contraction/expansion
    R = (D * D * math.pi / 4) / (in_to_m(3/8)**2 * math.pi / 4)
    Kcontract = 0.5 - 0.5 * R
    Kexpand = (1 - R) ** 2

    # Use line continuation backslashes or parentheses for multi-line addition
    p = (helper.p_major(lpm, L, D) +
         helper.p_minor(lpm, Kcontract, D) +
         helper.p_minor(lpm, Kexpand, D))
    
    return p

def configs(D_in):
    # Replaced MATLAB struct/array syntax with a Python dictionary
    D = in_to_m(D_in)
    L = in_to_m(12)
    return {
        'Q': 1000,
        'D': D,
        'L': L,
        'W': in_to_m(1),
        'A_cross': D * D * math.pi / 4,
        'A_surface': D * math.pi * L
    }


fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax2.set_ylabel('dT (K)', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim([0, 40])

# Filled the empty list with sample diameters to loop through
for d in [0.25, 0.375, 0.5]:
    cfg = configs(d)
    Q = cfg['Q']
    D = cfg['D']
    L = cfg['L']
    A_surface = cfg['A_surface']

    # Using np.arange instead of MATLAB's 0:0.2:20 syntax
    # Starting at 0.2 to prevent zero-flow calculation errors
    x = np.arange(0.2, 20.2, 0.2) 
    
    # Vectorized operations replace arrayfun
    h = h_conv(x, D)
    k = h * A_surface
    
    # Assuming loop_helper functions are vectorized or accept arrays
    p = pressure(x, L, D) / 1e5  # Convert Pa to bar
    
    # Plot Pressure on left axis (b-)
    ax1.plot(x, p, 'b-', label='pressure [bar]')
    ax1.set_xlabel('flow (LPM)')
    ax1.set_ylabel(f'pressure @ ID {d:.3f}', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Protect against Division by Zero when Re <= 10000 (h_conv = 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        dT = np.where(k > 0, Q / k, np.nan)
        
    ax2.plot(x, dT, 'r-', label=f'dT @ ID {d:.3f}"')

   
ax1.axvline(12, linestyle='--', color='blue')
ax2.axhline(30, linestyle='--', color='red')
ax1.legend()
ax2.legend()
plt.title(f'{configs(0)['L']:.3f}m Straight Tube Cold Plate @ {configs(0)['Q']} W Load')
plt.show()
