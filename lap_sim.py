# == Shine's Lap Sim ==
# First written 2026, Mk12
#
# Problem Statement:
# - Optimal endurance time is a balance between energy consumption and speed. How can we discover the
#   optimal compromise?
# - The mapping of Wh per lap to lap time is highly non-trivial.
#   The "truest" way is to have the driver drive the car at different paces.
#   The second truest way is to have the driver drive the sim at different paces.
# - We want a way to predict the mapping of f(E_lap) -> t_lap.
# 
# Theoretical Explaination:
# - Given power limits and car limits, simulate a lap and report the time and energy consumed.
#   "power limits" are limits that affect the energy consumed (ex. max torque).
#   "performance limits" are other car limits that do not affect energy consumed (ex. max lateral accel)
# - Generally, this is a function f(X_limits, X_performance) -> t_lap.
#   Where X is the car vector, refered to as just "car"
#   And X_limits is the power limit vector, refered to as just "limit"

import csv
import math
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# == Global sim settings ==
DT = 0.01                 # s, time step
LOG_INTERVAL_S = 5.0      # s of *simulated* time between console logs
MAX_SIM_TIME = 1200.0     # s, safety stop if the sim stalls (e.g. lap starts in a corner)
CORNER_ENTRY_TOL = 1.02   # warn if corner entry speed is >2% over the corner's max speed
TRACK_WIDTH = 2           # m, to determine racing line
MIN_STRAIGHT = 10.0       # m, a turn only gets a racing line if the straights on both sides are at least this long

PSI0 = -np.pi/3

class Car:
    mass: float  # kg

    FDR: float
    wheelR: float  # m

class LapLimits:
    lat: float    # m/s2
    brake: float  # m/s2, negative

    accel: float  # m/s2
    power: float  # W, pack power limit
    motor: float  # rpm

    regen_torque: float # Nm

    def pack_limited_accel(self, rpm, car):
        # return maximum acceleration under the pack power limit, at that rpm
        omega = rpm * 2 * np.pi / 60
        if omega < 1e-9:
            return math.inf  # no power limit at standstill; tire limit (accel) takes over

        # pack limit -> mechanical power available at the motor
        mech_power_limit = self.power * 0.95 
        motor_torque = mech_power_limit / omega
        wheel_force = motor_torque * car.FDR / car.wheelR
        return wheel_force / car.mass


def lap_sim_track(track_file, car, limits, *, verbose=True):

    def log(*args, **kwargs):
        if verbose:
            return print(*args, **kwargs)

    # == Car State object ==
    # Everything that changes during the simulation lives here. No state is kept anywhere else.
    @dataclass
    class State:
        # lap sim relevant states
        v: float = 0.0                 # m/s
        a: float = 0.0                 # m/s2 logitudinal acceleration. lateral acceleration is irrelevant for PT maths
        section_traveled: float = 0.0  # m, distance into the current section

        # progress through the lap
        section_idx: int = 0           # index of the current track section
        distance: float = 0.0          # m, total distance traveled

        # simulation results
        energy_used: float = 0.0       # Joules (filled in by lap_sim_powertrain)

    states = []
    state = State()

    # == helper functions ==
    rpm_to_v = lambda rpm: rpm * 2 * np.pi / 60 * car.wheelR / car.FDR
    v_to_rpm = lambda v:   60 * v * car.FDR / (2 * np.pi * car.wheelR)

    def load_track():
        type_aliases = {
            "straight": "straight", "s": "straight",
            "left": "left", "l": "left",
            "right": "right", "r": "right",
        }
        filepath = f"tracks/{track_file}"
        sections = []
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for line_no, row in enumerate(reader, start=2):
                if not row or all(c.strip() == "" for c in row):
                    continue
                if len(row) < 3:
                    raise ValueError(f"{filepath} line {line_no}: expected 3 columns (length, radius, type)")

                length = float(row[0])
                radius = float(row[1]) if row[1].strip() else 0.0
                sec_type = type_aliases.get(row[2].strip().lower())
                if sec_type is None:
                    raise ValueError(f"{filepath} line {line_no}: unknown section type '{row[2]}'")
                if sec_type != "straight" and radius <= 0:
                    raise ValueError(f"{filepath} line {line_no}: turn needs radius > 0")

                sections.append({"length_track": length, "radius_track": radius, "type": sec_type})
        return sections


    # Racing arc for a turn of radius R0 and length S0 (turn angle theta = S0 / R0), entering and
    # exiting on the outside edge of the track, tangent to the straights on both sides.
    # s = distance before the turn's entry that the car starts turning (and, by symmetry, the
    # distance after the exit that it stops turning). The arc has the same turn angle theta.
    # Closed form, with R_out = R0 + TRACK_WIDTH / 2 (outside edge):
    #   apex radius from the turn's center  = R_out - s * tan(theta / 4)
    #   arc radius                          = R_out + s / tan(theta / 2)
    # 2a. s = smax. Valid while the apex stays on the track, i.e. R_out - s * tan(theta / 4) >= R0 - TRACK_WIDTH / 2
    # 2b. otherwise the apex sits on the inside edge: s = TRACK_WIDTH / tan(theta / 4)
    # Returns Ro, So, s.
    def racing_line(R0, S0, smax):
        theta = S0 / R0

        if theta >= np.pi:  # edges are parallel (or diverge): no tangent run-up, keep the track geometry
            return R0, S0, 0.0

        R_out = R0 + TRACK_WIDTH / 2
        s = min(smax, TRACK_WIDTH / np.tan(theta / 4))
        Ro = R_out + s / np.tan(theta / 2)
        So = Ro * theta
        return Ro, So, s


    # Add a "v_max" parameter to every section, and find the racing line of every turn
    def precompute(sections):
        n = len(sections)

        # Pass 1: every section starts from the raw track geometry.
        # (Done for all sections before any turn is processed, so a turn shortening its neighbor
        # straights is never overwritten by a later reset, and neighbor lengths are always the raw ones.)
        for sec in sections:
            sec["radius"] = sec["radius_track"]
            sec["length"] = sec["length_track"]
            sec["turn_start"] = 0.0
            sec["v_max"] = math.inf  # straights: no speed limit from the track itself

        # Pass 2: turns
        def usable(neighbor):
            return (neighbor is not None
                    and neighbor["type"] == "straight"
                    and neighbor["length_track"] >= MIN_STRAIGHT)

        for i, sec in enumerate(sections):
            # straight: nothing to do
            if sec["type"] == "straight":
                continue

            # turns: finding the racing line!
            R0 = sec["radius_track"]
            S0 = sec["length_track"]
            prev_sec = sections[i - 1] if i > 0 else None
            next_sec = sections[i + 1] if i < n - 1 else None

            if usable(prev_sec) and usable(next_sec):
                # 1. the run-up is limited by half of each neighboring straight (the other half
                #    belongs to the turn on that straight's other side), the same on both sides
                smax = min(prev_sec["length_track"], next_sec["length_track"]) / 2

                # 2. racing arc, tangent to the straights, on the outside edge at entry and exit
                Ro, So, s = racing_line(R0, S0, smax)
            else:
                # no straight on one side, or too short: keep the track geometry
                Ro, So, s = R0, S0, 0.0

            # 3. write the new radius and length
            sec["v_max"] = np.sqrt(limits.lat * Ro)
            sec["radius"] = Ro
            sec["length"] = So
            sec["turn_entry_distance"] = s

            # 4. the car is already turning s before the entry and until s after the exit,
            #    so the neighboring straights are s shorter
            if s > 0:
                prev_sec["entry_length"] = s
                next_sec["exit_length"] = s
                prev_sec["length"] -= s
                next_sec["length"] -= s


    # Returns highest speed allowed at this position such that every upcoming turn can still be
    # entered at or below its own max speed, braking at limits.brake (constant decel).
    #  - the current section's own limit applies directly (inf on a straight)
    #  - an upcoming section j starting d meters ahead caps speed at sqrt(v_j^2 + 2*|brake|*d)
    # Sections are scanned only until they are too far away to matter.
    def speed_cap(section_idx, traveled):
        
        brake = abs(limits.brake)
        cap = sections[section_idx]["v_max"]
        d = max(0.0, sections[section_idx]["length"] - traveled)  # distance to the end of this section
        for j in range(section_idx + 1, n_sections):
            if 2 * brake * d >= v_motor_max ** 2:
                break  # even from standstill-level limits, nothing further ahead can bind
            cap = min(cap, np.sqrt(sections[j]["v_max"] ** 2 + 2 * brake * d))
            d += sections[j]["length"]
        return cap

    log("=" * 60)
    log("[track] START: time-stepping lap solve")

    # == Get track ==
    log(f"[track] using track:\t'{track_file}'...")
    sections = load_track()
    precompute(sections)

    n_sections = len(sections)
    total_length = sum(s["length"] for s in sections)
    v_motor_max = rpm_to_v(limits.motor)
    log(f"[track] {n_sections} sections, {total_length:.1f} m total, "
          f"motor-limited top speed {v_motor_max:.1f} m/s ({v_motor_max * 3.6:.0f} km/h)")

    # == Time stepping ==
    log_every = int(round(LOG_INTERVAL_S / DT))
    wall_start = time.perf_counter()

    while state.section_idx < n_sections:
        if len(states) * DT > MAX_SIM_TIME:  # simulated time = steps taken * DT
            raise RuntimeError(
                f"Sim exceeded {MAX_SIM_TIME:.0f} s, stopping. "
                "Likely stalled (speed is not increasing)."
            )

        section = sections[state.section_idx]
        v_old = state.v

        # Each step:
        # 1. Max acceleration available
        #    - straight: up to the power, tire (limits.accel), or rpm limit
        #    - turn: friction ellipse, (a_lat / lat)^2 + (a_long / accel)^2 = 1, with a_lat = v^2 / r.
        #      The grip used for cornering is not available for accelerating, so max a_long
        #      falls to zero as the car reaches the turn's max speed. Still power limited.
        # 2. Speed cap from the track ahead (see speed_cap). The car follows the cap down when
        #    it has to brake, and cannot slow faster than limits.brake.
        a_tire = limits.accel
        if section["type"] != "straight" and section["radius"] != 0:
            a_lat = v_old ** 2 / section["radius"]
            a_tire = limits.accel * np.sqrt(max(0.0, 1 - (a_lat / limits.lat) ** 2))
        a_max = min(a_tire, limits.pack_limited_accel(v_to_rpm(v_old), car))

        # cap evaluated where the car will be at the end of this step
        v_cap = speed_cap(state.section_idx, state.section_traveled + v_old * DT)

        v_new = v_old + a_max * DT
        v_new = min(v_new, v_motor_max)      # rpm limit
        v_new = min(v_new, v_cap)            # upcoming turn limits
        v_new = max(v_new, v_old + limits.brake * DT)  # braking limit
        v_new = max(v_new, 0.0)

        # update state: effective acceleration after clamping (keeps torque consistent with speed)
        step_dist = 0.5 * (v_old + v_new) * DT  # trapezoidal position increment
        state.a = (v_new - v_old) / DT
        state.v = v_new
        state.section_traveled += step_dist
        state.distance += step_dist
        states.append(replace(state))  # copy, otherwise every entry aliases the same object
        t = len(states) * DT           # simulated time at the end of this step

        # console log every LOG_INTERVAL_S of simulated time
        if len(states) % log_every == 0:
            log(f"[track] t={t:7.1f} s | section {state.section_idx + 1}/{n_sections} "
                  f"({section['type']:8s}) | {state.distance:8.1f}/{total_length:.1f} m | "
                  f"v={state.v:5.1f} m/s ({state.v * 3.6:5.1f} km/h) | "
                  f"rpm={v_to_rpm(state.v):6.0f} | a={state.a:+6.2f} m/s2")

        # check if section is done (carry leftover distance into the next one)
        while (state.section_idx < n_sections
               and state.section_traveled >= sections[state.section_idx]["length"]):
            state.section_traveled -= sections[state.section_idx]["length"]
            state.section_idx += 1
            if state.section_idx < n_sections and sections[state.section_idx]["type"] != "straight":
                v_limit = sections[state.section_idx]["v_max"]
                if state.v > v_limit * CORNER_ENTRY_TOL:
                    log(f"[track] WARNING: entered section {state.section_idx + 1} at "
                          f"{state.v:.1f} m/s, above its limit of {v_limit:.1f} m/s")

    wall = time.perf_counter() - wall_start
    log(f"[track] END: lap time {len(states) * DT:.2f} s, {len(states)} steps, "
          f"{state.distance:.1f} m, solved in {wall:.2f} s wall time")

    # Convert states into powertrain numbers and return
    # 1. Motor Torque: Calculated from states.a. Use mass, FDR, wheelR
    #     - Capped at -30Nm for regen.
    # 2. Motor speed: Calculated from states.v. Use FDR
    v = np.array([s.v for s in states])
    a = np.array([s.a for s in states])
    out = {
        "t": (np.arange(len(states)) + 1) * DT,
        "v": v,
        "a": a,
        "distance": np.array([s.distance for s in states]),  # m, for placing samples on the track
        "sections": sections,                                # track geometry, for the track plots
        "torque": np.maximum(limits.regen_torque, a * car.mass * car.wheelR / car.FDR),
        "rpm": 60 * v * car.FDR / (car.wheelR * 2 * np.pi),
        "lap_time": (len(states) + 1) * DT
    }
    return out


def lap_sim_powertrain(car, data, *, verbose=True):

    def log(*args, **kwargs):
        if verbose:
            return print(*args, **kwargs)

    # Now with all the driving states calculated, let's find some PT-relevant stats
    log("=" * 60)
    log("[powertrain] START")

    # 1. Wheel power: torque * motor speed (rad/s)
    # 2. Pack power: wheel power / efficiency when driving, * efficiency when regenerating
    #    - do not take the absolute value. keep negative torque as negative power
    # 3. Energy consumption: int( pack power * dt )
    #    - factor in regen. if torque < 0, then integrate a negative step
    dt = float(np.mean(np.diff(data["t"])))
    wheel_power = data["torque"] * data["rpm"] * 2 * np.pi / 60

    # == pack math ==
    efficiency = 0.95 
    pack_power = np.where(wheel_power >= 0, wheel_power / efficiency, wheel_power * efficiency)
    energy_used = float(np.sum(pack_power) * dt)  # J

    # == log results ==
    drive_energy = float(np.sum(pack_power[pack_power > 0]) * dt)
    regen_energy = float(-np.sum(pack_power[pack_power < 0]) * dt)
    log(f"[powertrain] net energy used:   {energy_used / 1e6:.3f} MJ  "
          f"({energy_used / 3600:.0f} Wh, {energy_used / 3.6e6:.3f} kWh)")
    log(f"[powertrain] drive / regen:     {drive_energy / 3600:.0f} Wh / {regen_energy / 3600:.0f} Wh")
    log(f"[powertrain] peak pack power:   {pack_power.max() / 1000:.1f} kW")
    log("[powertrain] END")

    # == output ==
    out = {
        **data,
        "wheel_power": wheel_power,
        "pack_power": pack_power,
        "energy_used": energy_used,
    }
    return out


def lap_sim_visualize(data):
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
    (torque_line,) = ax_torque.plot(t, data["torque"], color="tab:orange", label="Motor torque (Nm)")
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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, data["pack_power"] / 1000)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pack power (kW)")
    ax.set_title(f"Pack power over lap (net energy {data['energy_used'] / 3.6e6:.3f} kWh)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

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

            # if straight, simple
            if sec["type"] == "straight":
                Sa = sec.get("exit_length", 0)
                Sb = sec.get("entry_length", 0)
                n = max(2, int(math.ceil(L / ds)) + 1)
                t = np.linspace(0.0, L, n)
                pts = P + np.outer((Sa + t), tang(psi))
                path = np.concatenate([path, pts])
                path_dist = np.concatenate([path_dist, D+t])
                P = P + L0 * tang(psi) 
                D += sec["length"]
                continue

            # if turn
            S = sec.get("turn_entry_distance", 0)
            W = TRACK_WIDTH / 2
            theta = L / R
            theta0 = L0 / R0

            n = max(2, int(math.ceil(L / ds)) + 1)
            t = np.linspace(0.0, L, n)

            s = 1 if sec["type"] == "left" else -1      # left = counter-clockwise

            P1 = P - S * tang(psi) - (s * W * perp(psi) if S != 0 else 0)  # move to edge, move back to starting point
            O = P1 + s * R * perp(psi)                  # center of the turn
            phi = psi - s * np.pi / 2 + s * t / R     # polar angle about O
            circ = np.column_stack(tang(phi))
            path = np.concatenate([path, O + R * circ])
            path_dist = np.concatenate([path_dist, D+t])

            # corner entry point markers
            if S != 0:
                dot = P - S * tang(psi)
                dots.append(dot)

            # calculate new P, psi along the centerline 
            O = P + s * R0 * perp(psi)                  # center of the turn
            P = O + R0 * tang(psi - s * np.pi / 2 + s * theta0)
            psi = psi + s * theta0
            D += sec["length"]

            # corner exit point marker
            if S != 0:
                dot = P + S * tang(psi)
                dots.append(dot)


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
        ax.legend(loc="best")
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

    plt.show()


if __name__ == "__main__":
    print("==Lap Sim==")

    car = Car()
    car.mass = 285 # kg
    car.FDR = 3.91
    car.wheelR = 0.2  # m

    limit = LapLimits()
    limit.lat = 1.8 * 9.8
    limit.brake = -1.9 * 9.8
    limit.accel = 200 * car.FDR / (car.mass * car.wheelR)
    limit.power = 80e3   # W
    limit.motor = 5000   # rpm
    limit.regen_torque = -30 # Nm

    print(f"[main] Accel limit: {limit.accel:.2f} m/s2")

    lap_data = lap_sim_track("2022 Michigan Endurance June - One Lap.csv", car, limit)
    lap_data = lap_sim_powertrain(car, lap_data)
    lap_sim_visualize(lap_data)
