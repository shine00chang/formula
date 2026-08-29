# lines needed:
# "ts" "INV.COOL_T" "INV.MOT_T" "INV.CMD.TORQUE_COMMAND" "INV.ALL.CONTROL_BOARD_TEMP" "INV.ALL.COOLANT_TEMP" "INV.ALL.HOT_SPOT_TEMP" "FL_TSHMU_FLOW1" "FL_TSHMU_FLOW2" "INV.RPM" "INV.TQ_FB"

import csv
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from ntu import ntu_method
import radiators

def func(filename, time_len):
    print(f"using file:\t'{filename}'...")

    filepath = f'datasets/{filename}'
    
    with open(filepath, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        headers = next(reader)

    # Display available columns
    print("Found columns: ")
    print(headers)

    print(f"reading dataset from '{filepath}'...")
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1, filling_values=0)

    t = data[:, 0]
    d = data[:, 1:]
    
    # Time trimming
    print(f"dataset starts at time {datetime.fromtimestamp(t[0])}")
    t = t - t[0]
    
    if np.any(t > time_len):
        end = np.argmax(t > time_len)
    else:
        end = len(t)
        
    print(f"for time limit of {time_len}, using {end} of {t.shape[0]} lines")
    t = t[0:end]
    d = d[:end, :]

    # downsampling
    dt = np.mean(np.diff(t))
    print(f"raw's average time step: {dt}")
    downsample = 50
    print(f"downsampling by {downsample}...")
    t = t[::downsample]
    d = d[::downsample, :]
    dt = np.mean(np.diff(t))
    print(f"new's average time step: {dt}")

    # Extract coolant and motor temperatures
    try:
        T_coolant = d[:, headers.index("INV.COOL_T") - 1] 
        T_motor = d[:, headers.index("INV.MOT_T") - 1] 
        flow_rad = d[:, headers.index("FL_TSHMU_FLOW2") - 1]   
        flow_inv = d[:, headers.index("FL_TSHMU_FLOW1") - 1] 
        torque_rq = d[:, headers.index("INV.CMD.TORQUE_COMMAND") - 1] 
        torque_fb = d[:, headers.index("INV.TQ_FB") - 1] 
        rpm = - d[:, headers.index("INV.RPM") - 1] 
    except ValueError as e:
        print(f"Error finding required column headers: {e}")
        return

    # Physical Constants
    c_water = 4185      # J/(kg*K)
    rho_water = 1000    # kg/m^3
    c_air = 1005        # J/(kg*K)
    rho_air = 1.225     # kg/m^3

    # Flow rates (L/min) & NTU placeholders
    rad = None          
    Q_hot = 0.1         
    Q_cold = [0.1] * len(T_coolant)

    dT = savgol_filter(T_coolant, window_length=3001, polyorder=3, deriv=1, delta=dt)

    Q_rad = flow_rad / 60000
    Q_inv = flow_inv / 60000
    Q_motor = Q_inv - Q_rad 
    Q_air = 0.1

    # Calculate heat capacity rates
    C_water = rho_water * c_water * Q_rad
    C_air = rho_air * c_air * Q_air
    C_min = np.minimum(C_air, C_water)

    # Evaluate effectiveness
    e_rad = np.array([ntu_method(radiators.blackice_280, Q_hot, Q_cold_i)[0] for Q_cold_i in Q_cold])
    q_rad = e_rad * C_min * (T_coolant - 25)

    dT_rad = [q_rad[i] / (Q_inv[i] * rho_water * c_water) if Q_rad[i] != 0 else 0 for i in range(0, len(Q_rad))]
    dT_in = dT + dT_rad

    q_motor = dT_in * Q_inv * rho_water * c_water
    q_inv   = np.zeros(q_motor.shape)

    # Power math
    power = rpm * np.abs(torque_rq) * 2 * np.pi / 60
    efficiency = 1 - ((q_motor * 1.5 + q_inv) / power)

    # ==========================================
    # Plot 1: Temperatures with secondary y-axis
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    T_clean = savgol_filter(T_coolant, window_length=3001, polyorder=3)
    ax1.plot(t, T_clean, label=r"$T_{coolant,\ clean}$", color="blue", linewidth=1)
    ax1.plot(t, T_coolant, label=r"$T_{coolant}$", color="blue", linestyle="--", linewidth=1)
    ax1.plot(t, T_motor, label=r"$T_{motor}$", color="red", linewidth=1)
    ax1.set_ylabel(r"$Temperature\ [^\circ C]$")
    ax1.set_xlabel(r"$Time\ [s]$")

    ax2 = ax1.twinx()
    ax2.plot(t, dT, label=r"$\Delta T$", color="cyan", linestyle="--", linewidth=1)
    ax2.set_ylabel(r"$\Delta T\ [^\circ C/step]$")
    ax2.set_ylim([-2, 2])
    
    fig.suptitle(r"$Coolant\ and\ Motor\ Temperatures\ vs.\ Time$")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # ==========================================
    # Plot 2: Heat Outputs
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, q_rad, label=r"$Q_{rad}$", linewidth=1)
    ax.plot(t, q_motor, label=r"$Q_{motor}$", color="red", linewidth=1)
    ax.plot(t, q_inv, label=r"$Q_{inv}$", color="orange", linewidth=1)
    ax.axhline(1000, label=r"$Q_{motor,\ expected}$", linestyle="--", color="red", linewidth=1)
    ax.axhline(470, label=r"$Q_{inv,\ expected}$", linestyle="--", color="orange", linewidth=1)
    
    ax.set_ylabel(r"$Heat\ Output\ [W]$")
    ax.set_xlabel(r"$Time\ [s]$")
    ax.set_ylim([0, 2000])
    ax.legend(loc="upper right")

    ax2 = ax.twinx()
    ax2.plot(t, dT, label=r"$\Delta T$", color="cyan", linestyle="--", linewidth=1)
    ax2.plot(t, dT_in, label=r"$\Delta T_{in}$", color="green", linestyle="--", linewidth=1)
    ax2.set_ylabel(r"$\Delta T\ [^\circ C/step]$")
    ax2.set_ylim([-1, 1])
    ax2.legend(loc="upper left")

    ax.set_title(r"$Estimated\ Heat\ Outputs\ and\ Temperature\ Gradients$")

    # ==========================================
    # Plot 3: Torque and RPM
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, torque_rq, label=r"$\tau_{request}$", color="green", linewidth=1)
    ax.set_ylabel(r"$Torque\ Request\ [Nm]$")
    ax.set_xlabel(r"$Time\ [s]$")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(t, rpm, label=r"$\omega_{motor}$", color="blue", linewidth=1)
    ax2.set_ylabel(r"$Motor\ Speed\ [RPM]$")
    ax2.legend(loc="upper right")
    
    ax.set_title(r"$Torque\ Request\ and\ Motor\ RPM\ vs.\ Time$")

    # ==========================================
    # Plot 4: Power and Efficiency
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, power, label=r"$Power_{out}$", color="red", linewidth=1)
    ax.set_ylim([0, 80000])
    ax.set_ylabel(r"$Power\ [W]$")
    ax.set_xlabel(r"$Time\ [s]$")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(t, efficiency, label=r"$\eta_{efficiency}$", color="green", linewidth=1)
    ax2.axhline(0.96, label=r"$\eta_{target}$", color="green", linestyle="--", linewidth=1)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel(r"$Efficiency$")
    ax2.legend(loc="upper right")
    
    ax.set_title(r"$Motor\ Power\ Output\ and\ Efficiency\ vs.\ Time$")

func("real_heat_8_9_26.csv", 2000)
plt.show()
