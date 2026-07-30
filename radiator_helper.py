import numpy as np
import matplotlib.pyplot as plt
from ntu import ntu_method

# Physical constants for heat capacity calculations
RHO_AIR = 1.225      # kg/m^3
C_AIR = 1000.0       # J/(kg*K)
RHO_WATER = 1000.0   # kg/m^3
C_WATER = 4184.0     # J/(kg*K)

# Fundamentally is a function for finding operating Q_cold given system and requirement 
# If plotting, produces effectiveness & T_hot over Q_cold plot.
# Q in m3/s, T in C, q in Watts
def sim_rad_in_loop(rad, q_in, Q_hot, T_hot_req, T_cold, *, plot=True):
    Awater, Aair, Across, tubeA, tubeB, tubeN, D = rad()

    # Sweep air velocity from 0.1 to 6 m/s
    v_air = np.linspace(0.5, 6.0, 100)
    Q_cold = v_air * Across                     # Volumetric airflow (m^3/s)
    cfm = Q_cold * 2118.88                      # Convert m^3/s to CFM

    # Calculate heat capacity rates
    C_water = RHO_WATER * C_WATER * Q_hot
    C_air = RHO_AIR * C_AIR * Q_cold
    C_min = np.minimum(C_air, C_water)

    # Evaluate effectiveness (assuming ntu_method returns effectiveness epsilon)
    e = np.array([ntu_method(rad, Q_hot, Q_cold_i)[0] for Q_cold_i in Q_cold])
    
    # Calculate steady-state coolant inlet temperature: T_in = T_ambient + q_in / (epsilon * C_min)
    T_steady = T_cold + (q_in / (e * C_min))

    # Find the minimum airflow required to keep coolant at or below T_hot_req
    valid_indices = np.where(T_steady <= T_hot_req)[0]
    if len(valid_indices) > 0:
        op_idx = valid_indices[0]
    else:
        # If target temperature is unreachable, default to maximum airflow
        op_idx = np.argmin(T_steady)
        print(f"WARNING: Target temperature {T_hot_req}°C unreachable. Minimum achieved: {T_steady[op_idx]:.1f}°C")

    Q_cold_op = Q_cold[op_idx]
    cfm_op = cfm[op_idx]
    
    print(f"For {T_hot_req}°C coolant target (at {q_in}W load), requires {Q_cold_op/Across:.2f} m/s air velocity ({cfm_op:.1f} CFM).")

    if not plot:
        return Q_cold_op

    # ---------------------------------------------------------
    # Figure 3: Effectiveness and Steady-State Temp over CFM
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    # Plot effectiveness on left y-axis
    ax1.plot(cfm, e * 100, color='blue', linewidth=2, label='HX Effectiveness ($\\epsilon$)')
    ax1.set_xlabel('Airflow (CFM)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Effectiveness (%)', color='blue', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim([0, 100])

    # Plot steady-state temperature on right y-axis
    ax2.plot(cfm, T_steady, color='red', linewidth=2, linestyle='--', label='Steady-State Coolant Temp ($T_{hot}$)')
    ax2.set_ylabel('Coolant Inlet Temperature (°C)', color='red', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0,100])
    
    # Highlight required operating point
    ax2.axhline(T_hot_req, color='black', linestyle=':', label=f'Target Temp ({T_hot_req}°C)')
    ax1.axvline(cfm_op, color='black', linestyle='-.', label=f'Req. Airflow ({cfm_op:.0f} CFM)')

    # Combine legends from both axes

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(f'Radiator Performance vs. Airflow ({q_in/1000:.1f} kW Load)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    return Q_cold_op
