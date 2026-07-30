import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

from loop_components import emrax228, cm200dx, blackice_280, boschePCE, boschePCEpower
from loop_helper import solve_system, draw_system
from radiator_helper import sim_rad_in_loop
import radiator_springs

def system1(lpm):
    return cm200dx(lpm) + blackice_280(lpm);

def pump1(lpm):
    return boschePCE(lpm);

def pump_power1(lpm):
    return boschePCEpower(lpm);

def system2(lpm):
    return emrax228(lpm) + blackice_280(lpm);

def pump2(lpm):
    return boschePCE(lpm);

def pump_power2(lpm):
    return boschePCEpower(lpm);

# Generate x-axis data (0 to 30 with 0.1 increments)
x = np.arange(0, 30, 0.1)
system1_op_lpm = solve_system(x, pump1, system1);
system2_op_lpm = solve_system(x, pump2, system2);

Q_cold_op = sim_rad_in_loop(radiator_springs.blackice_280, 500 * 1.25, system1_op_lpm * (1.0 / 60000.0), 50.0, 20.0)
fan_op_power = Q_cold_op /  (0.0004719474 * 300) * 3.6 * 12 # assuming linear. need to make a fan libray at some point
pump_op_power = pump_power1(system1_op_lpm) 
system_op_power = pump_op_power + fan_op_power
total_power = system_op_power

print("=== System 1 ===")
print(f"At hydraulic operating point ({system1_op_lpm:.2f} LPM)")
print(f"pump system consumes {pump_op_power:.2f} W, fan consumes {fan_op_power:.2f} W. Total {system_op_power:.2f} W")
print("===")

Q_cold_op = sim_rad_in_loop(radiator_springs.blackice_280, 1000 * 1.25, system2_op_lpm * (1.0 / 60000.0), 50.0, 20.0)
fan_op_power = Q_cold_op /  (0.0004719474 * 300) * 3.6 * 12 # assuming linear. need to make a fan libray at some point
pump_op_power = pump_power2(system2_op_lpm) 
system_op_power = pump_op_power + fan_op_power
total_power += system_op_power

print("=== System 2 ===")
print(f"At hydraulic operating point ({system2_op_lpm:.2f} LPM)")
print(f"pump system consumes {pump_op_power:.2f} W, fan consumes {fan_op_power:.2f} W. Total {system_op_power:.2f} W")
print("===")

print(f"Total power consumed: {total_power:.2f} W")
print("===")

# ---------------------------------------------------------
# Figure 1: Component Curves
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(x, emrax228(x), label='Motor EMRAX 228')
plt.plot(x, cm200dx(x), label='Inverter CM200DX')
plt.plot(x, blackice_280(x), label='Radiator BlackIce 280mm')
plt.plot(x, boschePCE(x), 'g-', label='Pump BOSCHE PCE')
plt.plot(x, list(map(boschePCEpower,x)), label='Pump power BOSCHE PCE')

plt.xlabel('LPM')
plt.ylabel('Pressure (bar)')
plt.title('Component Curves')
plt.ylim([0, 5])
plt.grid(True)
plt.legend()

# ---------------------------------------------------------
# Figure 2: System1 Curves
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
draw_system(ax1, x, pump1, system1, system1_op_lpm)
ax1.axvline(12, color='g', linestyle='--', label='inverter spec')
ax1.set_title("System 1: Inverter")
ax1.legend()
 
# ---------------------------------------------------------
# Figure 3: System2 Curves
# ---------------------------------------------------------
draw_system(ax2, x, pump2, system2, system2_op_lpm)
ax2.axvline(6, color='b', linestyle='--', label='motor spec')
ax2.set_title("System 2: Motor")
ax2.legend()
 
plt.tight_layout()
plt.show();
