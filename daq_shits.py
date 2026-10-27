# lines needed: 1-11

import csv
import math 
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from ntu import ntu_method
import radiators
import motor_efficiency
from inverter_loss import inverter_loss


import numpy as np


def smooth_stairs(data, factor=None):
    """data_smooth = smooth_stairs(data_stairs)

    data   : 1D np.ndarray, uniformly sampled sample-and-hold (staircase) signal
    factor : optional int, MCU samples per sensor sample (e.g. 60). If None,
             sensor updates are detected as the points where the value changes.

    Returns a 1D np.ndarray of the same length, linearly interpolated between
    sensor updates.
    """
    x = np.asarray(data, dtype=float)
    n = np.arange(len(x))              # sample index; uniform spacing, so no timestamps needed

    if factor is None:
        knots = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
    else:
        knots = np.arange(0, len(x), factor)
    if knots[-1] != len(x) - 1:        # keep the last point so the curve spans the whole array
        knots = np.r_[knots, len(x) - 1]

    return np.interp(n, knots, x[knots])


def func(filename, time_range):
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
    
    if np.any(t > time_range[1]):
        end = np.argmax(t > time_range[1])
    else:
        end = len(t)
        
    if np.any(t > time_range[0]):
        start = np.argmax(t > time_range[0])
    else:
        start = 0

    print(f"for time limits {time_range}, using [{start}, {end}) of {t.shape[0]} lines")
    t = t[start:end]
    d = d[start:end, :]
    t_len = t[-1] - t[0]

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
        cv_avg = d[:, headers.index("BMS.AVG_CV") - 1] 
        T_avg_raw = d[:, headers.index("BMS.AVG_T") - 1] 
        T_high = d[:, headers.index("BMS.HI_T") - 1] 
        current = d[:, headers.index("BMS.I") - 1]   
        voltage = d[:, headers.index("BMS.V") - 1] 
        rpm = -d[:, headers.index("INV.RPM") - 1] 
        torque_cmd = d[:, headers.index("INV.TQ_CMD") - 1] 
        torque_fb = d[:, headers.index("INV.TQ_FB") - 1] 

        T_avg = smooth_stairs(T_avg_raw)
    except ValueError as e:
        print(f"Error finding required column headers: {e}")
        return


    # Physical Constants
    c_water = 4185      # J/(kg*K)
    rho_water = 1000    # kg/m^3
    c_air = 1005        # J/(kg*K)
    rho_air = 1.225     # kg/m^3

    # NOTE: Mk11 numbers!
    pack_heat_capacity = 0.048 * 600 * 970
    moduleIR = 0.017 / 6
    pack_resistance = moduleIR * 100 * 1.2
    

    # Pack Data (V & I, not heats) 
    print("== pack data  ==");
    power = current * voltage
    current_avg = np.mean(current)
    current_rms = np.sqrt(np.mean(current**2))
    print(f'current avg:\t{current_avg:.2f}A')
    print(f'current rms:\t{current_rms:.2f}A')
    print(f'power avg:\t{np.mean(power)/1000:.2f} kW')

    pack_heat_expected = current * current * pack_resistance;
    print(f'pack heat expected:\t{np.mean(pack_heat_expected):.2f}W')
    
    
    # Motor Heat 
    print("== motor heat ==")
    motor_eff = motor_efficiency.build();
    for i in range(len(t)):
        if np.isnan(motor_eff(np.abs(rpm[i]), np.abs(torque_cmd[i]))):
            print(rpm[i], torque_cmd[i])
    motor_heat_expected = power * (1-motor_eff(np.abs(rpm), np.abs(torque_cmd)))
    print(f'motor heat expected:\t{np.nanmean(motor_heat_expected):.2f}W (cmd)')

    # Inverter Heat 
    print("== inverter heat ==")
    inverter_heat_expected = [inverter_loss(voltage[i], current[i]) for i in range(len(t))]
    print(f'inverter heat expected:\t{np.nanmean(inverter_heat_expected):.2f}W')

    
    # Cell Heat
    print("== cell heat ==");
    d_T = T_avg[-1] - T_avg[0]
    cell_heat = d_T * pack_heat_capacity / t_len
    print(f'cell del temp:\t{d_T:.2f}T\tover {t_len:.2f}s')
    print(f'cell heat :\t{cell_heat:.2f}W\tfrom calorimetry')

    dT_avg = np.diff(T_avg, prepend=1) / dt
    dT_avg[0] = 0
    cell_heat = dT_avg * pack_heat_capacity
    print(f'cell heat :\t{np.mean(cell_heat):.2f}W\tfrom cell avg T data (transient)')

    # ==========================================
    # Plot 1: Temperatures with secondary y-axis
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), nrows=2, ncols=1)
    
    ax1.plot(t, T_avg_raw, label=r"$T_{avg raw}$", color="red", linestyle=":", linewidth=1)
    ax1.plot(t, T_avg, label=r"$T_{avg}$", color="red", linewidth=1)
    ax1.plot(t, T_high, label=r"$T_{high}$", color="red", linestyle="--", linewidth=1)
    ax1.set_ylabel(r"$Temperature\ [^\circ C]$")
    ax1.set_xlabel(r"$Time\ [s]$")

    ax1.legend(loc="upper left")

    ax2.plot(t, dT_avg, label=r"$\frac{dT_{avg}}{dt}$", color="red", linewidth=1)
    ax1.set_ylabel(r"$Temperature\ [^\circ C]$")
    ax1.set_xlabel(r"$Time\ [s]$")
    ax1.set_title(r"$Cell Heat$")

    ax2.legend(loc="upper left")

    # ==========================================
    # Plot 2: Voltage, Current, Power Curves 
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), nrows=2, ncols=1)

    ax1.plot(t, current, label=r"$current$", color="blue", linewidth=0.5)
    ax1.set_ylabel(r"$current\ [A]$")
    ax1.set_xlabel(r"$Time\ [s]$")

    ax3 = ax1.twinx()
    ax3.plot(t, voltage, label=r"$voltage$", color="green", linewidth=0.5)
    ax3.set_ylabel(r"$voltage$")
    
    ax1.set_title(r"$pack stats$")
    ax1.legend(loc="upper left")
    ax3.legend(loc="upper right")

    ax2.plot(t, power / 1000, color="green", label=r"$voltage\ &\ current$", linestyle="--", linewidth=0.5);
    ax2.plot(t, np.abs(rpm) * np.abs(torque_cmd) * 2 * math.pi / 60 / 1000, color="green", label=r"$inv data$", linestyle="-", linewidth=0.5);
    ax2.set_ylabel(r"$power\ [kW]$")
    ax2.set_xlabel(r"$Time\ [s]$")
    ax2.set_title(r"$pack power$")

    ax2.legend(loc="upper left")

    # ==========================================
    # Plot 3: heats
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(t, motor_heat_expected, label=r"$motor$", color="red", linestyle=":", linewidth=0.5)
    ax1.plot(t, inverter_heat_expected, label=r"$inverter$", color="orange", linestyle=":", linewidth=0.5)
    ax1.plot(t, pack_heat_expected, color="blue", linestyle=":", linewidth=0.5)
    ax1.set_ylabel(r"$heat\ [W]$")
    ax1.set_xlabel(r"$Time\ [s]$")
    ax1.set_title(r"$heats$")

    ax1.legend(loc="upper left")

    # ==========================================
    # Plot 3: RPM & Torque 
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(t, rpm, label=r"$rpm$", color="red", linestyle="-", linewidth=0.5)
    ax1.set_ylabel(r"$RPM$")
    ax1.set_xlabel(r"$Time\ [s]$")

    ax2 = ax1.twinx()
    ax2.plot(t, torque_cmd, label=r"$torque_{cmd}$", color="blue", linestyle=":", linewidth=0.5)
    ax2.set_ylabel(r"$Torque\ [Nm]$")

    ax1.set_title(r"$wheel$")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")


func("real_pack_8_9_26_driver1.csv", [40,180])
#func("real_pack_test.csv", [40,100])
plt.show()
