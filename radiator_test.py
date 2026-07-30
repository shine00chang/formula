import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import radiators


# Physical constants
def rhowater(): return 1000.0
def rhoair(): return 1.225
def cwater(): return 4184.0
def cair(): return 1000.0
def viscositywater(): return 0.7 * 1e-6
def viscosityair(): return 1.6 * 1e-5
def kwater(): return 0.6
def kair(): return 0.026


def generate_fake_calibration_data():
    os.makedirs('datasets', exist_ok=True)
    filename = 'datasets/heater_calibration_fake_2000ml.csv'
    start_temp = 23.0
    initial_rate = 0.24  # deg/s at t = 0
    final_rate = 0.10    # deg/s at t = 60
    duration = 60        # seconds
    
    rate_decay = (initial_rate - final_rate) / duration
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for t in range(duration + 1):
            temp = start_temp + (initial_rate * t) - (0.5 * rate_decay * (t ** 2))
            writer.writerow([t, round(temp, 2)])

    print(f"fake calibration data made at:\t{filename}")


def generate_fake_test_data():
    os.makedirs('datasets', exist_ok=True)
    filename = 'datasets/heater_test_2000ml_3ms_air_25Cair_8LPM.csv'
    start_temp = 23.0
    initial_rate = 0.20   # deg/s at t = 0
    final_rate = 0.0      # deg/s at t = 60
    duration = 200        # seconds
    
    rate_decay = (initial_rate - final_rate) / duration
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for t in range(duration + 1):
            temp = start_temp + (initial_rate * t) - (0.5 * rate_decay * (t ** 2))
            writer.writerow([t, round(temp, 2)])

    print(f"fake test data made at:\t{filename}")


def calibrate(*, plot=False):
    filename = 'heater_calibration_fake_2000ml.csv'
    data = np.loadtxt(f'datasets/{filename}', delimiter=',')
    print(f"calibrating using file:\t'{filename}'...")

    t = data[:, 0]
    T = data[:, 1]

    res = re.search(r'([0-9]*)ml', filename)
    if res is None:
        print(f"ERROR: could not parse milliliter count in file name {filename}. needs to match pattern ([0-9]*)ml")
        return None

    m = int(res.group(1)) / 1000.0  # in kg

    # q' = m * c * T'
    T_dot = np.gradient(T, t)
    q_dot = m * cwater() * T_dot

    T_x = np.linspace(round(np.min(T)), round(np.max(T)), 100)
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
        
    f_q_dot = lambda T_in: q_dot[find_nearest(T, T_in)]

    if not plot:
        return f_q_dot
        
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(t, T, label='T', color="red")
    ax1.set_ylabel('$T_{water}$ (°C)', color="red")
    ax1.tick_params(axis='y', labelcolor="red")

    ax2.plot(t, T_dot, label='dT/dt', color="blue")
    ax2.set_ylabel('$\\frac{dT}{dt}$ (°C/s)', color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")

    ax1.set_xlabel('time (s)')
    ax1.set_title("Temperature vs. Time")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax3.plot(T_x, [f_q_dot(val) for val in T_x], label='$q_{heater}$', color='red')
    ax3.set_ylabel('$q_{heater}$ (W)', color="red")
    ax3.set_xlabel('$T_{water}$ (°C)')
    ax3.set_title("Real Heat Input vs. Temperature")
    ax3.legend()

    fig.tight_layout()
    return f_q_dot


# Radiator geometry
def radiator():
    L = 0.24
    W = 0.12
    D = 0.046
    finN = 520            # number of fins between two tubes
    finH = 0.147 * 0.0254 # gap between a fin and the one above it
    
    tubeA = D             # longer side of tube cross section
    tubeB = 0.0013        # shorter side of tube cross section
    tubeL = 0.24          # length of tube
    tubeN = 14            # number of tubes
    
    Awater = 2.0 * (tubeA + tubeB) * tubeL * tubeN
    Across = L * (W - tubeN * tubeB)
    Aair = 2.0 * tubeN * L * D + 2.0 * tubeN * finN * finH * D
    
    return Awater, Aair, Across, tubeA, tubeB, tubeN, D


def get_heats(T_hot, dT_hot, T_air, m_water, Q_water_m3s, Q_air_m3s):
    Awater, Aair, Across, tubeA, tubeB, tubeN, D = radiator()

    C_water = cwater() * rhowater() * Q_water_m3s
    C_air = cair() * rhoair() * Q_air_m3s
    C_min = min(C_air, C_water)

    # Net heat rejected by radiator = Heat input from heater minus thermal energy stored in water
    q_out = q_in(T_hot) - m_water * cwater() * dT_hot
    q_ideal = C_min * (T_hot - T_air)

    e = q_out / q_ideal if q_ideal > 0 else 0.0
    return q_out, e, C_min

def func():
    filename = 'heater_test_2000ml_3ms_air_25Cair_8LPM.csv'
    data = np.loadtxt(f'datasets/{filename}', delimiter=',')
    print(f"pulling test data from file:\t'{filename}'...")

    t = data[:, 0]
    T_hot = data[:, 1]
    dT_hot = np.gradient(T_hot, t)

    res = re.search(r'([0-9]*)ml', filename)
    if res is None:
        print(f"ERROR: could not parse milliliter count in file name {filename}.")
        return
    m_water = int(res.group(1)) / 1000.0  # in kg 

    res = re.search(r'([0-9\.]*)ms_air', filename)
    if res is None:
        print(f"ERROR: could not parse air speed in file name {filename}.")
        return
    v_air = float(res.group(1))  # in m/s 

    res = re.search(r'([0-9]*)LPM', filename)
    if res is None:
        print(f"ERROR: could not parse LPM in file name {filename}.")
        return
    LPM = float(res.group(1))  # in LPM 

    res = re.search(r'([0-9]*)Cair', filename)
    if res is None:
        print(f"ERROR: could not parse air temp in file name {filename}.")
        return
    T_air = float(res.group(1))  # Keep in °C to match T_hot

    print("== extracted parameters ==")
    print(f"m_water:\t{m_water} kg")
    print(f"air flow:\t{v_air} m/s")
    print(f"water flow:\t{LPM} LPM")
    print(f"air temp:\t{T_air} °C")

    Awater, Aair, Across, tubeA, tubeB, tubeN, D = radiator()

    # Convert flow rates to SI units (m^3/s)
    Q_water_m3s = LPM / (1000.0 * 60.0)
    Q_air_m3s = v_air * Across

    q_out = np.zeros(len(T_hot))
    e = np.zeros(len(T_hot))
    C_min_val = 0.0
    
    for i in range(len(T_hot)):
        q_out[i], e[i], C_min_val = get_heats(T_hot[i], dT_hot[i], T_air, m_water, Q_water_m3s, Q_air_m3s)

    e_ntu = ntu_method(radiators.blackice280, Q_water_m3s, Q_air_m3s)[0]
    
    # Correct NTU heat rejection formula: q = epsilon * C_min * (T_hot_in - T_cold_in)
    q_out_ntu = [e_ntu * C_min_val * (T - T_air) for T in T_hot]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(t, T_hot, label='$T_{water}$', color="red")
    ax1.set_ylabel('$T_{water}$ (°C)', color="red")
    ax1.tick_params(axis='y', labelcolor="red")

    ax2.plot(t, dT_hot, label='dT/dt', color="blue", linestyle='--')
    ax2.set_ylabel('$\\frac{dT}{dt}$ (°C/s)', color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")
    ax2.set_xlabel('time (s)')
    ax1.set_title("Temperature vs. Time")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax3.plot(t, q_out, label='$q_{out, real}$', color='blue')
    ax3.plot(t, q_out_ntu, label='$q_{out, NTU}$', color='blue', linestyle='--')
    ax3.set_ylabel('Heat Rejection (W)', color="blue")
    ax3.tick_params(axis='y', labelcolor="blue")
    ax3.set_xlabel('time (s)')

    ax4 = ax3.twinx()
    ax4.plot(t, e, label='$eff_{real}$', color='green')
    ax4.axhline(e_ntu, label='$eff_{NTU}$', color='green', linestyle='--')
    ax4.set_ylabel('Effectiveness ($\\epsilon$)', color="green")
    ax4.tick_params(axis='y', labelcolor="green")
    ax4.set_ylim([0, 1])

    ax3.set_title("Measured HX Effectiveness vs. NTU Method")
    ax3.legend(loc="upper left")
    ax4.legend(loc="upper right")

    fig.tight_layout()


generate_fake_calibration_data()
q_in = calibrate(plot=True)
generate_fake_test_data()
func()

plt.show()
