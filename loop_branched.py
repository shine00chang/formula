import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

from loop_components import emrax228, cm200dx, blackice_280, boschePCE, boschePCEpower
from radiator_helper import sim_rad_in_loop
import radiators

def solve_branches(x, pump, main, branch1, branch2):

    choking_cutoff = 0;

    def solve_lpm1(lpm2):
        def f(x):
            return branch1(x) - branch2(lpm2)
        
        try:
            # fsolve returns an array of roots; we take the first one. 
            # Using 10.0 as an initial guess.
            lpm1 = fsolve(f, 10.0)[0]
            # In this physics context, we only care about positive flow roots
            if lpm1 < 0:
                lpm1 = 0
            return lpm1
        except Exception:
            choking_cutoff = lpm2
            return 0.0

    # Pre-allocate arrays
    branch1_lpm = np.zeros_like(x)
    branch2_lpm = x.copy()
    system_lpm = np.zeros_like(x)
    system_p = np.zeros_like(x)
    system_op_lpm = 0.0

    # Calculate system values across the range
    for i in range(len(x)):
        lpm2 = branch2_lpm[i]
        branch1_lpm[i] = solve_lpm1(lpm2)
        system_lpm[i] = lpm2 + branch1_lpm[i]
        system_p[i] = branch2(lpm2) + main(system_lpm[i])
        
        if system_p[i] < pump(system_lpm[i]):
            system_op_lpm = system_lpm[i]

    return (system_lpm, system_p, system_op_lpm, branch1_lpm, branch2_lpm, choking_cutoff)

def branch1(lpm):
    return emrax228(lpm)

def branch2(lpm):
    return blackice_280(lpm)

def pump(lpm):
    return boschePCE(lpm) * 2;

def pump_power(lpm):
    return boschePCEpower(lpm) * 2;


# Generate x-axis data (0 to 30 with 0.1 increments)
x = np.arange(0, 30.1, 0.1)
(system_lpm, system_p, system_op_lpm, branch1_lpm, branch2_lpm, choking_cutoff) = solve_branches(x, pump, cm200dx, branch1, branch2)

Q_cold_op = sim_rad_in_loop(radiators.blackice_280, 1500 * 1.25, system_op_lpm * (1.0 / 60000.0), 50.0, 20.0)
fan_op_power = Q_cold_op /  (0.0004719474 * 300) * 3.6 * 12 # assuming linear. need to make a fan libray at some point
pump_op_power = pump_power(system_op_lpm) 
system_op_power = pump_op_power + fan_op_power
total_power = system_op_power

print("===")
print(f"At hydraulic operating point ({system_op_lpm:.2f} LPM)")
print(f"pump system consumes {pump_op_power:.2f} W, fan consumes {fan_op_power:.2f} W. Total {system_op_power:.2f} W")
print("===")

# ---------------------------------------------------------
# Figure 1: Component Curves
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(x, emrax228(x), label='Motor EMRAX 228')
ax1.plot(x, cm200dx(x), label='Inverter CM200DX')
ax1.plot(x, blackice_280(x), label='Radiator BlackIce 280mm')
ax1.plot(x, boschePCE(x), 'g-', label='Pump BOSCHE PCE')

ax2 = ax1.twinx()
ax2.plot(x, list(map(boschePCEpower,x)), 'g--', label='Pump power BOSCHE PCE')
ax2.set_ylabel('Power (W)')
ax2.legend(loc='upper right')

ax1.set_xlabel('LPM')
ax1.set_ylabel('Pressure (bar)')
ax1.set_title('Component Curves')
ax1.set_ylim([0, 5])
ax1.grid(True)
ax1.legend()

# ---------------------------------------------------------
# Figure 2: System Curves & Branch Flows
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: System Curves
ax1.plot(x, branch1(x), 'b-', label='1: Motor')
ax1.plot(x, branch2(x), 'r-', label='2: Radiator')
ax1.plot(system_lpm, system_p, 'g-', label='System')
ax1.plot(x, pump(x), 'g-', label='Pumps: 2 Series')
ax1.axvline(choking_cutoff, color='purple', linestyle=':', label='Choking Point')

ax1.set_xlabel('LPM')
ax1.set_ylabel('Pressure (bar)')
ax1.set_title('System Curves')
ax1.set_ylim([0, 5])
ax1.set_xlim([0, 30])
ax1.grid(True)
ax1.legend()


# Subplot 2: Branch Flows
ax2.plot(system_lpm, branch1_lpm, 'b-', label='branch 1')
ax2.plot(system_lpm, branch2_lpm, 'r-', label='branch 2')
ax2.axhline(6, color='b', linestyle='--', label='motor spec')
ax2.axvline(12, color='g', linestyle='--', label='inverter spec')
ax2.axvline(system_op_lpm, color='k', linestyle='--', label='operating point')

ax2.set_xlabel('LPM of System')
ax2.set_ylabel('LPM of Branch')
ax2.set_title('Branch Flows')
ax2.set_ylim([0, 30])
ax2.set_xlim([0, 30])
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
