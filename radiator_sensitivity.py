import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ntu import ntu_method

# --- Unit Conversion Constants ---
LPM_TO_M3S = 1.0 / 60000.0          # Liters per minute -> m^3/s
CFM_TO_M3S = 0.000471947443         # Cubic feet per minute -> m^3/s

# --- Geometry Constants [m] ---
M_PER_TUBE = 0.0085  # 8.5 mm tube pitch
M_PER_FIN = 0.0008   # 0.8 mm fin/louver pitch

def fake_radiator(L, W, D):
    finN = L / M_PER_FIN
    tubeB = 0.0013       # shorter side of tube cross section (1.3 mm)
    finH = M_PER_TUBE - tubeB
    
    tubeA = D            # longer side of tube cross section
    tubeL = L            # length of tube
    tubeN = (W - 0.01) / M_PER_TUBE

    Awater = 2.0 * (tubeA + tubeB) * tubeL * tubeN
    Across = L * (W - tubeN * tubeB)
    Aair = 2.0 * tubeN * L * D + 2.0 * tubeN * finN * finH * D
    
    return Awater, Aair, Across, tubeA, tubeB, tubeN, D


def plot_sensitivity(x, y_label_vals, z, xlabel="Geometry Parameter [m]"):
    """
    Plots a 1x2 figure:
      - Left: 3D surface plot of Geometry (x), Q_cold (y), and q_out (z)
      - Right: 2D line plot of Geometry (x) vs q_out (z) at Min, Mid, and Max Q_cold
    """
    fig = plt.figure(figsize=(14, 6))
    
    # 2D meshgrid using 'ij' indexing to align shape (len(x), len(y))
    X, Y = np.meshgrid(x, y_label_vals, indexing='ij')
    
    # 1. 3D Surface Plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Q_cold Airflow [CFM]')
    ax1.set_zlabel('Heat Rejection q_out [W]')
    ax1.set_title(f'3D Surface: {xlabel} vs Airflow')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='q_out [W]')
    
    # 2. 2D Slice Plot (Low, Mid, and High airflow rates)
    ax2 = fig.add_subplot(1, 2, 2)
    indices_to_plot = [0, len(y_label_vals) // 2, -1]
    for idx in indices_to_plot:
        ax2.plot(x, z[:, idx], label=f'Q_cold = {y_label_vals[idx]:.0f} CFM', linewidth=2)
        
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Heat Rejection q_out [W]')
    ax2.set_title(f'2D Sensitivity: {xlabel}')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()


# --- Baseline Radiator Dimensions [m] ---
L_default = 0.24   # 240 mm length
W_default = 0.12   # 120 mm width
D_default = 0.046  # 46 mm thickness

# --- Flow Rates (Converted to m^3/s for ntu_method) ---
Q_hot_lpm = 10.0
Q_hot_m3s = Q_hot_lpm * LPM_TO_M3S

Q_cold_cfm = np.linspace(100.0, 500.0, 50)
Q_cold_m3s = Q_cold_cfm * CFM_TO_M3S

delta_T = 50.0 - 25.0  # Th_in - Tc_in = 25 K difference

# ==========================================
# 1. LENGTH SENSITIVITY STUDY
# ==========================================
L_v = np.linspace(0.1, 0.5, 50)
e_L = np.zeros((len(L_v), len(Q_cold_m3s)))
C_min_L = np.zeros((len(L_v), len(Q_cold_m3s)))

for i, L in enumerate(L_v):
    for j, Q_c in enumerate(Q_cold_m3s):
        rad_func = lambda: fake_radiator(L, W_default, D_default)
        res = ntu_method(rad_func, Q_hot_m3s, Q_c)
        e_L[i, j] = res[0]      # Effectiveness epsilon
        C_min_L[i, j] = res[6]  # C_min from index 6
        
q_out_L = C_min_L * delta_T * e_L
plot_sensitivity(L_v, Q_cold_cfm, q_out_L, xlabel="Radiator Length L [m]")


# ==========================================
# 2. WIDTH SENSITIVITY STUDY
# ==========================================
W_v = np.linspace(0.10, 0.5, 50)
e_W = np.zeros((len(W_v), len(Q_cold_m3s)))
C_min_W = np.zeros((len(W_v), len(Q_cold_m3s)))

for i, W in enumerate(W_v):
    for j, Q_c in enumerate(Q_cold_m3s):
        rad_func = lambda: fake_radiator(L_default, W, D_default)
        res = ntu_method(rad_func, Q_hot_m3s, Q_c)
        e_W[i, j] = res[0]
        C_min_W[i, j] = res[6]
        
q_out_W = C_min_W * delta_T * e_W
plot_sensitivity(W_v, Q_cold_cfm, q_out_W, xlabel="Radiator Width W [m]")


# ==========================================
# 3. DEPTH (THICKNESS) SENSITIVITY STUDY
# ==========================================
D_v = np.linspace(0.02, 0.1, 50)
e_D = np.zeros((len(D_v), len(Q_cold_m3s)))
C_min_D = np.zeros((len(D_v), len(Q_cold_m3s)))

for i, D in enumerate(D_v):
    for j, Q_c in enumerate(Q_cold_m3s):
        rad_func = lambda: fake_radiator(L_default, W_default, D)
        res = ntu_method(rad_func, Q_hot_m3s, Q_c)
        e_D[i, j] = res[0]
        C_min_D[i, j] = res[6]
        
q_out_D = C_min_D * delta_T * e_D
plot_sensitivity(D_v, Q_cold_cfm, q_out_D, xlabel="Radiator Depth D [m]")

plt.show()
