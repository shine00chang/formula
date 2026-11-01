import copy
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    if limits_max.power != limits_min.power:
        d = (limits_max.power - limits_min.power) / (n_per_axis-1)
        out = []
        for i in range(n_per_axis):
            for lim in lim_space:
                lim = copy.copy(lim) # clone, so every point is its own object
                lim.power = limits_min.power + d * i
                out.append(lim)
        lim_space = out

    # Brake Axis
    if limits_max.brake != limits_min.brake:
        d = (limits_max.brake - limits_min.brake) / (n_per_axis-1)
        out = []
        for i in range(n_per_axis):
            for lim in lim_space:
                lim = copy.copy(lim) # clone, so every point is its own object
                lim.brake = limits_min.brake + d * i
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
            "brake": lim.brake, #m/s2, convert back into Nm
            "energy_used": out["energy_used"] / 3600, # J -> Wh
            "lap_time": out.get("lap_time", out["t"][-1]) # s, last timestamp = lap time
        }

    # Run search.
    # drop if lap sim errors out
    points = []
    n_total = len(lim_space)
    n_dropped = 0
    print(f"[scan] START: {n_total} points")
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



def score_function (energy, lap_time):
    # using 2026 results
    t_min = 1312;
    t_max = t_min * 1.45; 
    e_max = 0.797;
    e_min = 0.289;
    E_min = 2.84; #kWh

    t_us = lap_time * 22
    score_time = np.where(
        t_us < t_min * 1.45, 
        250 * (np.fmax(t_max, t_us) / t_us - 1) / 0.45,
        0)
    score_laps = 25; # assume we finish XD
    E_us = energy * 22 / 1000; #kWh
    e_us = (np.fmin(t_us, t_min) / t_us) * (np.fmin(E_us, E_min) / E_us);
    score_eff = np.where(
        t_us < t_min * 1.45,
        100 * (e_us  - np.fmin(e_min, e_us) ) / (np.fmax(e_us, e_max) - np.fmin(e_min, e_us)),
        0)

    return score_eff + score_time + score_laps



def lap_space_visualize (points, score_function):
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
            "energy_used": 162,
            "lap_time": 67.6,
        },
        {
            "energy_used": 219,
            "lap_time": 62.5,
        },
    ]
    other_points = [
        # {
        #     "energy_used": 138.7,
        #     "lap_time": 90.4,
        #     "label": "Mk11"
        # },
        {
            "energy_used":  241,
            "lap_time": 59.65,
            "label": "OSU"
        },
        {
            "energy_used":  148,
            "lap_time": 65.19,
            "label": "UPitt"
        },
        {
            "energy_used":  212,
            "lap_time": 63.4,
            "label": "SLO"
        },
    ]

    energy = np.array([p["energy_used"] for p in points])            # Wh
    lap_time = np.array([p["lap_time"] for p in points])             # s
    power_kw = np.array([p["power"] for p in points]) / 1000   # kW
    brake_g = np.array([p["brake"] for p in points]) / 9.8     # G
    accel = np.array([p["accel"] for p in points])             # m/s2

    # marker size: 30 (lowest accel limit) to 150 (highest)
    a_min, a_max = accel.min(), accel.max()
    if a_max > a_min:
        sizes = 30 + 120 * (accel - a_min) / (a_max - a_min)
    else:
        sizes = np.full_like(accel, 80.0)

    # == Plotting ==
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Non-lap-sim points
    ax.scatter(
        [p["energy_used"] for p in AS_points], 
        [p["lap_time"] for p in AS_points],
        c="r", marker='x', label="AS Mk11")
    ax.scatter(
        [p["energy_used"] for p in other_points], 
        [p["lap_time"] for p in other_points],
        c="b", marker='x')
    for p in other_points:
        ax.annotate(p["label"], (p["energy_used"], p["lap_time"]))

    # Lap-sim points
    sc = ax.scatter(energy, lap_time, c=brake_g, s=sizes, cmap="plasma",
                    edgecolors="k", linewidths=0.4)

    #fig.colorbar(sc, ax=ax, label="Pack power limit (kW)")
    fig.colorbar(sc, ax=ax, label="Brake limit (G)")


    # == Score surface  ==
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = score_function(X,Y)

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    Z = Z[:-1, :-1]
    levels = ticker.MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())


    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.colormaps['inferno']
    # score_surface = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, zorder=0)

    # contours are *point* based plots, so convert our bound into point
    # centers
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    score_surface = ax.contourf(X[:-1, :-1] + dx/2.,
                      Y[:-1, :-1] + dy/2., 
                      Z,
                      levels=levels,
                      cmap=cmap,
                      zorder=0)

    fig.colorbar(score_surface, ax=ax, label="score")

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
    car.mass = 285 # kg
    car.FDR = 3.71
    car.wheelR = 0.2  # m
    car.coeff_rolling_resistance = 0.016 
    car.coeff_drag_area = 1.38

    # Define seach space
    limit_min = lap_sim.LapLimits()
    limit_min.lat = 1.8 * 9.8
    limit_min.lon = 1.8 * 9.8
    limit_min.brake = 0.7 * 9.8
    limit_min.accel = 100 * car.FDR / (car.mass * car.wheelR)
    limit_min.power = 80e3   # W
    limit_min.motor = 5000   # rpm
    limit_min.regen_torque = -30 # Nm

    limit_max = copy.copy(limit_min)
    limit_max.accel = 200 * car.FDR / (car.mass * car.wheelR)
    limit_max.power = 80e3   # W
    limit_max.brake = 1.7 * 9.8

    # Run
    points = lap_space_scan(limit_min, limit_max, 10, car)
    lap_space_visualize(points, score_function)
