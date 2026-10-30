import copy
import time

import matplotlib.pyplot as plt
import numpy as np

import lap_sim 

# First creates the search space. 
# Given search space limits and axis fidelity, creates limit objects where each axis A spans 
# from lim_min.A to lim_max.A, with n_per_axis points (n_per_axis - 1 subdivisions).
def lap_space_scan (limits_min, limits_max, n_per_axis, car):

    if n_per_axis < 2:
        raise ValueError("n_per_axis must be at least 2 (one point for the min, one for the max)")
    
    # 1. Create search space
    lim_space = []

    # Accel Axis
    d = (limits_max.accel - limits_min.accel) / (n_per_axis-1)
    for i in range(n_per_axis):
        lim = copy.copy(limits_min) # clone
        lim.accel = limits_min.accel + d * i
        lim_space.append(lim)

    # Power Axis
    d = (limits_max.power - limits_min.power) / (n_per_axis-1)
    out = []
    for i in range(n_per_axis):
        for lim in lim_space:
            lim = copy.copy(lim) # clone, so every point is its own object
            lim.power = limits_min.power + d * i
            out.append(lim)
    lim_space = out

    # 2. Run search
    track = "2022 Michigan Endurance June - One Lap.csv"

    # Convenience function maps relevant inputs and outputs only relevant outputs
    def lap_wrapper (lim):
        out = lap_sim.lap_sim_track(track, car, lim, verbose=False)
        out = lap_sim.lap_sim_powertrain(car, out, verbose=False)
        return {
            "power": lim.power,
            "accel": lim.accel, #m/s2, convert back into Nm
            "energy_used": out["energy_used"] / 3600, # J -> Wh
            "lap_time": out.get("lap_time", out["t"][-1]) # s, last timestamp = lap time
        }

    # Run search.
    # drop if lap sim errors out
    points = []
    n_total = len(lim_space)
    n_dropped = 0
    print(f"[scan] START: {n_total} points ({n_per_axis} accel x {n_per_axis} power)")
    wall_start = time.perf_counter()

    for k, lim in enumerate(lim_space, start=1):
        try:
            point = lap_wrapper(lim)
        except Exception as e:
            n_dropped += 1
            print(f"[scan] dropped point {k}/{n_total} "
                  f"(accel={lim.accel:.2f} m/s2, power={lim.power / 1000:.1f} kW): {e}")
            continue
        points.append(point)

        if k % 10 == 0 or k == n_total:
            print(f"[scan] {k}/{n_total} done")

    wall = time.perf_counter() - wall_start
    print(f"[scan] END: {len(points)} points kept, {n_dropped} dropped, solved in {wall:.1f} s wall time")

    return points

def lap_space_visualize (points):
    # Plots:
    # - Plot lap_time over energy_used as a scatter plot
    #   Color shows the power limit and marker size shows the accel limit, so each dot can be traced
    #   back to the limits that produced it.
    if not points:
        print("[scan] no points to plot")
        return
    
    # Data points from other sources. Plotted as reference
    AS_points = [
        {
            "energy_used": 289,
            "lap_time": 61.8,
        },
        {
            "energy_used":  82,
            "lap_time": 81,
        },
        {
            "energy_used": 162,
            "lap_time": 67.6,
        },
        {
            "energy_used": 219,
            "lap_time": 62.5,
        },
    ]
    Mk11_comp = {
        "energy_used": 138.7,
        "lap_time": 90.4,
    }

    energy = np.array([p["energy_used"] for p in points])            # Wh
    lap_time = np.array([p["lap_time"] for p in points])             # s
    power_kw = np.array([p["power"] for p in points]) / 1000   # kW
    accel = np.array([p["accel"] for p in points])             # m/s2

    # marker size: 30 (lowest accel limit) to 150 (highest)
    a_min, a_max = accel.min(), accel.max()
    if a_max > a_min:
        sizes = 30 + 120 * (accel - a_min) / (a_max - a_min)
    else:
        sizes = np.full_like(accel, 80.0)

    fig, ax = plt.subplots(figsize=(9, 6))

    sc = ax.scatter(
        [p["energy_used"] for p in AS_points], 
        [p["lap_time"] for p in AS_points],
        c="r", marker='x', label="AS Mk11")
    sc = ax.scatter(
        [Mk11_comp["energy_used"]], 
        [Mk11_comp["lap_time"]], 
        c="c", marker='x', label="Mk11 comp")

    sc = ax.scatter(energy, lap_time, c=power_kw, s=sizes, cmap="viridis",
                    alpha=0.75, edgecolors="k", linewidths=0.4)
    fig.colorbar(sc, ax=ax, label="Pack power limit (kW)")

    # size legend (lowest / middle / highest accel limit)
    if a_max > a_min:
        for val, size in ((a_min, 30), (0.5 * (a_min + a_max), 90), (a_max, 150)):
            ax.scatter([], [], s=size, color="gray", edgecolors="k", linewidths=0.4,
                       label=f"accel limit {val:.1f} m/s²")
        ax.legend(loc="best", title="Marker size", labelspacing=1.2)

    ax.set_xlabel("Energy used (Wh)")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("Lap time vs. energy used")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    print("==Lap Optimizer==")

    # Define Car
    car = lap_sim.Car()
    car.mass = 225 # kg
    car.FDR = 3.71
    car.wheelR = 0.2  # m

    # Define seach space
    limit_min = lap_sim.LapLimits()
    limit_min.lat = 1.8 * 9.8
    limit_min.brake = -1.9 * 9.8
    limit_min.accel = 100 * car.FDR / (car.mass * car.wheelR)
    limit_min.power = 60e3   # W
    limit_min.motor = 5000   # rpm
    limit_min.regen_torque = -30 # Nm

    limit_max = copy.copy(limit_min)
    limit_max.accel = 200 * car.FDR / (car.mass * car.wheelR)
    limit_max.power = 80e3   # W

    # Run
    points = lap_space_scan(limit_min, limit_max, 10, car)
    lap_space_visualize(points)
