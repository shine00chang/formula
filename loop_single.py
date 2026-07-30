import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

# Assuming these are imported from your local environment
from loop_components import emrax228, cm200dx, blackice_280, boschePCE, boschePCEpower
from loop_helper import solve_system, draw_system
from radiator_helper import sim_rad_in_loop
import radiator_springs

# Heat load for radiator maths

def system(lpm):
    return cm200dx(lpm) + emrax228(lpm) + blackice_280(lpm)

def pump(lpm):
    return boschePCE(lpm) * 2

def pump_power(lpm):
    return boschePCEpower(lpm) * 2

# 280mm Radiator Geometry Helper (replaces undefined radiator() call)
def radiator_280_geometry():
    L = 0.28          # Length (m)
    W = 0.14          # Width (m)
    D = 0.046         # Thickness/Depth (m)
    tubeB = 0.0013    # Tube width (m)
    tubeN = 16        # Number of tubes
    
    # Frontal cross-sectional area for airflow (m^2)
    Across = L * (W - tubeN * tubeB)
    return Across

# Generate x-axis hydraulic data (0 to 30 LPM with 0.1 increments)
x = np.arange(0, 30.1, 0.1)
system_op_lpm = solve_system(x, pump, system)
pump_op_power = pump_power(system_op_lpm)

Q_cold_op = sim_rad_in_loop(radiator_springs.blackice_280, 1500 * 1.25, system_op_lpm * (1.0 / 60000.0), 50.0, 20.0)
fan_op_power = Q_cold_op /  (0.0004719474 * 300) * 3.6 * 12 # assuming linear. need to make a fan libray at some point
system_op_power = pump_op_power + fan_op_power

print("===")
print(f"At hydraulic operating point ({system_op_lpm:.2f} LPM)")
print(f"pump system consumes {pump_op_power:.2f} W, fan consumes {fan_op_power:.2f} W. Total {system_op_power:.2f} W")
print("===")

# ---------------------------------------------------------
# Figure 1: Component Impedance Curves
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(x, emrax228(x), label='Motor EMRAX 228')
plt.plot(x, cm200dx(x), label='Inverter CM200DX')
plt.plot(x, blackice_280(x), label='Radiator BlackIce 280mm')
plt.plot(x, boschePCE(x), 'g-', label='Pump BOSCHE PCE (Single)')
plt.plot(x, [boschePCEpower(val) for val in x], linestyle=':', label='Pump Power (W)')

plt.xlabel('Flow Rate (LPM)')
plt.ylabel('Pressure Drop (bar) / Power (W)')
plt.title('Component Hydraulic & Power Curves')
plt.ylim([0, 5])
plt.grid(True)
plt.legend()

# ---------------------------------------------------------
# Figure 2: System Operating Point
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
draw_system(ax, x, pump, system, system_op_lpm)
ax.axvline(6, color='b', linestyle='--', label='Motor Min Spec')
ax.axvline(12, color='g', linestyle='--', label='Inverter Min Spec')
ax.set_title('Hydraulic System Operating Point')
ax.legend()

plt.tight_layout()
plt.show()
