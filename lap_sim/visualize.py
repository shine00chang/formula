import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from models import *

PSI0 = -np.pi/3

def visualize(data):
    # Plots to make
    # - 2-row subplot of power, rpm, and torque, with rpm and torque on the same graph
    # - Pack power over time graph
    # - A 2D drawing of the track, with the line being a gradient where the color indicates the speed of the car.
    # - The same 2D drawing but with the gradient being the logitudinal acceleration of the car
    t = data["t"]

    # == 2-row subplot: pack power on top, rpm and torque sharing one graph below ==
    fig, (ax_power, ax_rpm) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_power.plot(t, data["pack_power"] / 1000, color="tab:green")
    ax_power.axhline(0, color="k", linewidth=0.5)
    ax_power.set_ylabel("Pack power (kW)")
    ax_power.set_title("Power, motor speed and torque over lap")
    ax_power.grid(True, alpha=0.3)

    ax_torque = ax_rpm.twinx()
    (rpm_line,) = ax_rpm.plot(t, data["rpm"], color="tab:blue", label="Motor speed (rpm)")
    (torque_line,) = ax_torque.plot(t, data["motor_torque"], color="tab:orange", label="Motor torque (Nm)")
    ax_torque.axhline(0, color="k", linewidth=0.5)
    ax_rpm.set_xlabel("Time (s)")
    ax_rpm.set_ylabel("Motor speed (rpm)", color="tab:blue")
    ax_torque.set_ylabel("Motor torque (Nm)", color="tab:orange")
    ax_rpm.tick_params(axis="y", colors="tab:blue")
    ax_torque.tick_params(axis="y", colors="tab:orange")
    ax_rpm.legend(handles=[rpm_line, torque_line], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax_rpm.grid(True, alpha=0.3)
    fig.tight_layout()

    # == Pack power over time ==
    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1,figsize=(10, 4))
    ax1.plot(t, data["pack_power"] / 1000, color="b", linewidth=1)
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pack power (kW)")
    ax1.set_title(f"Pack power over lap (net energy {data['energy_used'] / 3.6e6:.3f} kWh)")
    ax1.grid(True, alpha=0.3)

    ax3 = ax2.twinx()
    ax3.plot(t, data["voltage"], label="voltage", color="g", linestyle='--',linewidth=1)
    ax3.set_ylabel("Voltage (V)")

    ax2.plot(t, data["current"], label="current", color="b", linewidth=1)
    ax2.axhline(0, color="k", linewidth=0.5)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Current (A)")
    ax2.set_title(f"Pack power over lap (net energy {data['energy_used'] / 3.6e6:.3f} kWh)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()

def visualize_on_track(data):
    # == Track Plots ==
    # plots track bounds on the given axis
    def plot_track_bounds(ax):
        sections = data["sections"]

        perp = lambda psi: np.array([np.cos(psi + np.pi/2), np.sin(psi + np.pi/2)])  # 90 deg left of the heading
        tang = lambda psi: np.array([np.cos(psi), np.sin(psi)])

        D = TRACK_WIDTH / 2
        ds = 0.25
        P = np.zeros(2)   # centerline position
        psi = PSI0         # heading
        path1 = (P + D * perp(psi))[None, :]   # left edge, starts at the first point
        path2 = (P - D * perp(psi))[None, :]   # right edge

        for sec in sections:
            L = sec["length_track"]
            R = sec["radius_track"]
            n = max(2, int(math.ceil(L / ds)) + 1)
            t = np.linspace(0.0, L, n)[1:]     # skip t = 0, that point is already the end of the path

            if sec["type"] == "straight":
                path1 = np.concatenate([path1, path1[-1] + np.outer(t, tang(psi))])
                path2 = np.concatenate([path2, path2[-1] + np.outer(t, tang(psi))])
                P = P + L * tang(psi)
            else:
                s = 1 if sec["type"] == "left" else -1      # left = counter-clockwise
                theta = L / R
                O = P + s * R * perp(psi)                   # center of the turn
                phi = psi - s * np.pi / 2 + s * t / R     # polar angle about O
                circ = np.column_stack(tang(phi))

                # the left edge is the inside edge of a left turn and the outside edge of a right turn
                path1 = np.concatenate([path1, O + max(R - s * D, 0.0) * circ])
                path2 = np.concatenate([path2, O + max(R + s * D, 0.0) * circ])
                P = O + R * circ[-1]                        # end of the centerline arc
                psi = psi + s * theta

        for path in (path1, path2):
            segments = np.stack([path[:-1], path[1:]], axis=1)   # (n - 1, 2, 2): one segment per pair of points
            ax.add_collection(LineCollection(segments, linestyle='-', colors='blue', linewidth=1))
        ax.autoscale_view()


    # Build the track centerline from the sections (straights go straight, turns are arcs
    # of the given radius; left = counter-clockwise, right = clockwise), using the track
    # geometry ("radius_track" / "length_track"; falls back to "radius" / "length" if absent).
    # Sim distance runs along each section's "length" (the racing line's length in a turn),
    # so every sample is placed by how far through its section it is.
    # Also returns the racing line entry / exit points of the turns (R1 and R2 from the center
    # of the turn, set by each section's "entry" and "exit"), for the red dots.
    def make_track_time_segments(sections, distances, ds=0.25):

        perp = lambda psi: np.array([np.cos(psi + np.pi/2), np.sin(psi + np.pi/2)])  # 90 deg left of the heading
        tang = lambda psi: np.array([np.cos(psi), np.sin(psi)])
        rot = lambda psi: np.array([[np.sin(psi), np.cos(psi)], [-np.cos(psi), np.sin(psi)]])  # (dx right, dy forward) -> world

        path = np.empty((0,2))
        path_dist = np.empty((0))
        dots = []
        P = np.zeros(2)
        D = 0
        psi = PSI0
        for sec in sections:
            L0 = sec["length_track"]
            R0 = sec["radius_track"]
            L = sec.get("length", L0)
            R = sec.get("radius", R0)

            # if straight, simple: from the racing line's start to its end
            if sec["type"] == "straight":
                start = P + rot(psi) @ sec["Da"]
                end = P + L0 * tang(psi) + rot(psi) @ sec["Db"]
                n = max(2, int(math.ceil(L / ds)) + 1)
                t = np.linspace(0.0, L, n)
                pts = start + np.outer(np.linspace(0.0, 1.0, n), end - start)
                path = np.concatenate([path, pts])
                path_dist = np.concatenate([path_dist, D+t])
                P = P + L0 * tang(psi) 
                D += sec["length"]
                continue

            # if turn
            theta = L / R
            theta0 = L0 / R0

            n = max(2, int(math.ceil(L / ds)) + 1)
            t = np.linspace(0.0, L, n)

            s = 1 if sec["type"] == "left" else -1      # left = counter-clockwise

            # geometric end of the section
            O = P + s * R0 * perp(psi)                  # center of the turn
            P_end = O + R0 * tang(psi - s * np.pi / 2 + s * theta0)
            psi_end = psi + s * theta0

            # racing line start and end, shifted from the geometric ones
            start = P + rot(psi) @ sec["Da"]
            end = P_end + rot(psi_end) @ sec["Db"]

            # the chord of an arc points half its sweep past the heading at its start
            chord = end - start
            h = np.arctan2(chord[1], chord[0]) - s * theta / 2
            O = start + s * R * perp(h)                 # center of the racing arc
            phi = h - s * np.pi / 2 + s * t / R         # polar angle about O
            circ = np.column_stack(tang(phi))
            path = np.concatenate([path, O + R * circ])
            path_dist = np.concatenate([path_dist, D+t])

            # corner entry & exit point markers
            if sec["Da"].any():
                dots.append(start)
            if sec["Db"].any():
                dots.append(end)

            P = P_end
            psi = psi_end
            D += sec["length"]


        # for every time segment i, find the largest k where path_dist[k] < distance[i].
        # then interpolate between path[k][k+1] through path_dist[k:k+1] and distance[i]
        # for x[i] y[i].



        k = np.clip(np.searchsorted(path_dist, distances, side="right"), 0, len(path_dist) - 1)
        x = np.interp(distances, path_dist, path[:, 0])
        y = np.interp(distances, path_dist, path[:, 1])
        dots = np.array(dots)
        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        return segments, dots 

    segments, dots = make_track_time_segments(data["sections"], data["distance"])

    # plots some time series onto racing line 
    def plot_along_track(values, label, title, cmap, norm):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor('black')
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_track_bounds(ax)

        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=6)
        lc.set_array(0.5 * (values[:-1] + values[1:]))  # color each segment by its mean value
        ax.add_collection(lc)
        ax.plot(segments[0][0][0], segments[0][0][1], "ko", markersize=6, label="Start")

        # plot dots
        if len(dots):
            ax.plot(dots[:,0], dots[:,1], "x", color="red", markersize=2, label="Racing line entry / exit", zorder=3)

        fig.colorbar(lc, ax=ax, label=label)
        fig.tight_layout()

    # speed
    plot_along_track(data["v"], "Speed (m/s)", "Track colored by speed", "viridis", plt.Normalize(0, data["v"].max()))

    # longitudinal acceleration (green = accelerating, red = braking, centered on zero)
    a_lim = max(float(np.abs(data["a"]).max()), 1e-6)
    plot_along_track(data["a"], "Longitudinal acceleration (m/s²)", "Track colored by longitudinal acceleration",
               "RdYlGn", plt.Normalize(-a_lim, a_lim))


