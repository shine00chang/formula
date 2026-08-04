import numpy as np
import matplotlib.pyplot as plt

# physical constants
def rhowater(): return 1000.0
def rhoair(): return 1.225
def cwater(): return 4184.0
def cair(): return 1000.0
def viscositywater(): return 0.7 * 1e-6
def viscosityair(): return 1.6 * 1e-5
def kwater(): return 0.6
def kair(): return 0.026

# design requirements, constraints
def parameters():
    T0 = 273.0 + 25.0       # Starting temperature of coolant
    Tair = 273.0 + 30.0     # Ambient temperature of air
    qmotor = 1000.0 * 0.66
    qinverter = 500.0
    qload = qmotor + qinverter       # Heat load
    Qwater = 6.0 * 1.67 * 1e-5       # Flow rate of water (from LPM)
    V = 0.5 * 0.001         # Volume of water (from liters)
    return T0, Tair, qload, qmotor, qinverter, Qwater, V

# radiator geometry
# Using Nemesis Black Ice GTR 240. Sourced from Luke's Mk10 Code
def radiator():
    # Radiator size
    L = 0.24
    W = 0.12
    D = 0.046
    finN = 520      # number of fins between two tubes
    finH = 0.147 * 0.0254 # gap between a fin and the one above it
    
    # NOTE: radiator splits flow into many parallel tubes, before rejoining. 
    tubeA = D       # longer side of tube cross section
    tubeB = 0.0013  # shorter side of tube cross section
    tubeL = 0.24    # length of tube
    tubeN = 14      # number of tubes
    
    # Area the water is convecting through
    Awater = 2.0 * (tubeA + tubeB) * tubeL * tubeN
        
    # Area the air is flowing through, required for air velocity
    Across = L * (W - tubeN * tubeB)
    
    # Area the air is convecting through
    Aair = 2.0 * tubeN * L * D + 2.0 * tubeN * finN * finH * D
    
    return Awater, Aair, Across, tubeA, tubeB, tubeN, D

# See: https://www.lamar.edu/engineering/_files/documents/mechanical/dr.-fan-publications/2012/No%2031%202012%20ASEE%20Radiator%20Final.pdf
def effectivenessNTU(Qwater, Qair):
    Awater, Aair, Across, tubeA, tubeB, tubeN, D = radiator()

    Depth = D
    
    Ch = Qwater * rhowater() * cwater()
    Cc = Qair * rhoair() * cair()
    C = min(Ch, Cc)
    Cr = min(Cc, Ch) / max(Cc, Ch)
    
    # hwater
    Dhydraulic = 4.0 * tubeA * tubeB / (2.0 * tubeA + 2.0 * tubeB)
    vwater = Qwater / (tubeA * tubeB * tubeN)
    Rewater = vwater * Dhydraulic / viscositywater()
    Nuwater = 5.6 # refer to table 8.1 in 105D
    hwater = Nuwater * kwater() / Dhydraulic
    
    # hair
    vair = Qair / Across
    Reair = vair * Depth / viscosityair()
    Prair = 0.7
    Nuair = 0.664 * (Reair**0.5) * (Prair**0.33)
    hair = Nuair * kair() / Depth
    uair = 0.99 # TODO: though from Luke's code, this radiator gives about 0.99 efficiency.
    
    # overall heat transfer coefficient
    UA = 1.0 / ( 1.0/(uair * hair * Aair) + 1.0/(hwater * Awater) )
    NTU = UA / C
    
    # Cross flow, both fluids unmixed
    e = 1.0 - np.exp( (np.exp(-(NTU**0.78) * Cr) - 1) / (Cr * (NTU**-0.22)) )
    return e, NTU, hwater, hair, Nuair, Reair

# the plotted function.
# for some CFM, outputs the inlet, outlet, and effectiveness.
def f(cfm):
    T0, Tair, qload, qmotor, qinverter, Qwater, V = parameters()
    Qair = cfm * 0.00047
    Ch = Qwater * rhowater() * cwater()
    Cc = Qair * rhoair() * cair()
    Cmin = min(Ch, Cc)
    
    # q = (Thi - Tair) * Cmin * e
    # Thi = Tair + q / (Cmin * e)
    e, NTU, hwater, hair, Nuair, Reair = effectivenessNTU(Qwater, Qair)
    
    Thi = Tair + qload  / (Cmin * e)
    Tho = Thi - qload / Ch
    
    if cfm == 300:
        print(f"At 300 CFM -> hwater: {hwater:.2f}, hair: {hair:.2f}, Nuair: {Nuair:.2f}, Reair: {Reair:.2f}")
        
    return e, Thi, Tho, NTU


# ---------------------------------------------------------------------
# Main Execution and Plotting
# ---------------------------------------------------------------------

T0, Tair, qload, qmotor, qinverter, Qwater, V = parameters()
cfms = np.arange(100, 701, 1) # 100 to 700 inclusive

# Pre-allocate lists for outputs
e_list, Thi_list, Tho_list, NTU_list = [], [], [], []

for cfm in cfms:
    e, Thi, Tho, NTU = f(cfm)
    e_list.append(e)
    Thi_list.append(Thi)
    Tho_list.append(Tho)
    NTU_list.append(NTU)

# Convert to numpy arrays for element-wise math
e_arr = np.array(e_list)
Thi_arr = np.array(Thi_list)
Tho_arr = np.array(Tho_list)
NTU_arr = np.array(NTU_list)

Tmotoro = Thi_arr + qmotor / (Qwater * rhowater() * cwater())
Tinverteri = (Tmotoro + Tho_arr) / 2.0
Tinvertero = Tinverteri + qinverter / (2.0 * Qwater * rhowater() * cwater())

# --- Figure 1: Main System Output ---
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel("Air Flow (CFM)")
ax1.set_ylabel('Temperature (C)')
ax1.set_ylim([0, 70])
ax1.set_yticks(np.arange(0, 75, 5))
ax1.grid(True)

# Plot Left Axis (Temperatures)
l1, = ax1.plot(cfms, Thi_arr - 273, 'r-', label='inlet & motor inlet')
l2, = ax1.plot(cfms, Tho_arr - 273, 'b-', label='outlet')
l3, = ax1.plot(cfms, Tinverteri - 273, 'c-', label='inverter inlet')
l4 = ax1.axhline(50, color='r', linestyle='--', label='motor requirement')
l5 = ax1.axhline(45, color='k', linestyle='--', label='DCDC requirement')
l6 = ax1.axhline(40, color='b', linestyle='--', label='DCDC water requirement')

# Plot Right Axis (Effectiveness)
ax2 = ax1.twinx()
ax2.set_ylabel('Effectiveness')
ax2.set_ylim([0, 1])
l7, = ax2.plot(cfms, e_arr, 'g-', label='effectiveness')

# Combine legends from both axes
lines = [l1, l2, l3, l4, l5, l6, l7]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='best')

plt.title(f"Radiator Steady State for Load = {qload:.1f} W")
plt.tight_layout()

# --- Figure 2: Effectiveness ---
fig2, ax_ntu = plt.subplots(figsize=(8, 4))
ax_ntu.plot(cfms, NTU_arr, color='tab:blue')
ax_ntu.set_xlabel("Air Flow (CFM)")
ax_ntu.set_ylabel("NTU")
ax_ntu.set_title("NTU vs Air Flow")
ax_ntu.grid(True)

plt.show()
