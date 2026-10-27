# ===
# Interpolates EMRAX 228 efficiency graph
# data from: https://emrax.com/wp-content/uploads/2025/03/EMRAX_228_datasheet_v1.6.pdf
# ===
# Developer's Note: rpm is scaled down by 0.01 because the axis magnitudes
# should be similar for interpolation to look nice.

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def build():
    """Return F(rpm, torque) -> efficiency. The interpolator is built once here."""
    # x is rpm, y is torque
    rpm150 = [400, 700, 1000, 1200, 4100, 5000]
    e150   = [.88, .92, .94,  .95,  .95,  .94]
    rpm100 = [300, 600, 950, 1400, 1600, 3500, 4000, 4500, 4750, 5000]
    e100   = [.86, .92, .94, .95,  .96,  .96,  .95,  .94,  .92,  .88]
    rpm75  = [300, 600, 950, 1700, 2200, 3200, 4200, 4500, 4750]
    e75    = [.86, .92, .94, .95,  .96,  .95,  .94,  .92,  .86]
    rpm50  = [300, 1000, 3800, 4300]
    e50    = [.86, .92,  .92,  .86]
    rpm25  = [1000, 2000, 3000, 4000, 5000]
    e25    = [.86,  .86,  .86,  .86,  .86]

    x = np.array(rpm25 + rpm50 + rpm75 + rpm100 + rpm150, dtype=float) * 0.01
    # NOTE: kept as in the MATLAB original: the 25-torque curve is assigned
    # torque = 1 (ones), not 25. Looks like a possible typo; change 1 -> 25 if so.
    y = np.array([1] * len(e25) + [50] * len(e50) + [75] * len(e75)
                 + [100] * len(e100) + [150] * len(e150), dtype=float)
    v = np.array(e25 + e50 + e75 + e100 + e150, dtype=float)

    pts = np.column_stack((x, y))
    lin = LinearNDInterpolator(pts, v)
    nearest = NearestNDInterpolator(pts, v)  # fallback outside the data's convex hull

    def F(rpm, torque):
        xi = np.asarray(rpm, dtype=float) / 100
        yi = np.asarray(torque, dtype=float)
        out = lin(xi, yi)
        return np.where(np.isnan(out), nearest(xi, yi), out)[()]

    return F


# call this to verify interpolated graph
def plot_efficiency():
    import matplotlib.pyplot as plt

    F = motor_efficiency()

    a = np.arange(1, 5001, 50)   # MATLAB 1:50:5000
    b = np.arange(1, 151, 5)     # MATLAB 1:5:150
    X, Y = np.meshgrid(a, b)
    Z = F(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel("rpm")
    ax.set_ylabel("torque")
    ax.set_zlabel("efficiency")
    plt.show()
