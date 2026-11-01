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
CHICANE_MAX_ANGLE = np.pi / 4  # rad, two back to back turns only count as a chicane if both turn this angle or less

PSI0 = -np.pi/3

class Car:
    mass: float  # kg

    FDR: float
    wheelR: float  # m

class LapLimits:
    lat: float    # m/s2
    lon: float    # m/s2

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


    # Racing line through a chicane: two arcs of the same radius r that turn opposite ways, with a
    # straight of length l between them (l = 0 when the turns are back to back). Like a single turn,
    # the path is tangent to the straights before and after the chicane, and starts s before the first
    # turn and ends s after the second turn, on the outside edge of each turn. The path stays on the
    # track, and s is the value that gives the largest radius. The arcs do not keep the track's turn
    # angles th1 and th2: they sweep ph1 and ph2, with ph1 - ph2 = th1 - th2.
    # Frame used here: the first turn is a left turn starting at the origin, x forward and y left
    # (a chicane that starts with a right turn is the mirror image of this).
    # Returns None if no racing line has a larger radius than both of the track's turns.
    def chicane_fit(R1, th1, Lm, R2, th2, smax):
        W = TRACK_WIDTH / 2
        dth = th1 - th2                                   # heading change over the whole chicane
        fwd = lambda a: np.array([np.cos(a), np.sin(a)])
        left = lambda a: np.array([-np.sin(a), np.cos(a)])

        # track centerline: B1 = end of turn 1, B2 = start of turn 2, C = end of turn 2
        O1 = np.array([0.0, R1])
        B1 = O1 + R1 * fwd(th1 - np.pi / 2)
        B2 = B1 + Lm * fwd(th1)
        O2 = B2 - R2 * left(th1)
        C = O2 + R2 * fwd(th1 + np.pi / 2 - th2)

        s_cap = min(smax, 30.0)
        n_pts = lambda length: max(2, int(np.ceil(length / 0.1)) + 1)

        def arc(center, R, a0, sweep, sign):
            a = a0 + sign * np.linspace(0.0, sweep, n_pts(R * sweep))
            return center + R * np.column_stack([np.cos(a), np.sin(a)])

        centerline = np.concatenate([[[-(s_cap + 1.0), 0.0]],
                                     arc(O1, R1, -np.pi / 2, th1, +1),
                                     arc(O2, R2, th1 + np.pi / 2, th2, -1),
                                     [C + (s_cap + 1.0) * fwd(dth)]])
        seg_a, seg_b = centerline[:-1], centerline[1:]
        seg_d = seg_b - seg_a
        seg_len2 = np.maximum((seg_d ** 2).sum(1), 1e-12)

        # True if every point of the path is within half the track width of the centerline
        def on_track(path):
            rel = path[:, None, :] - seg_a[None, :, :]
            u = np.clip((rel * seg_d[None, :, :]).sum(2) / seg_len2[None, :], 0.0, 1.0)
            closest = seg_a[None, :, :] + u[:, :, None] * seg_d[None, :, :]
            dist = np.sqrt(((path[:, None, :] - closest) ** 2).sum(2)).min(1)
            return dist.max() <= W + 2e-3

        # arc 1, straight l, arc 2: from E, with arc 1 sweeping ph1 and arc 2 sweeping ph2
        def racing_path(E, r, ph1, ph2, l):
            t1 = np.linspace(0.0, ph1, n_pts(r * ph1))
            p1 = E + r * np.column_stack([np.sin(t1), 1 - np.cos(t1)])
            J1 = p1[-1]
            J2 = J1 + l * fwd(ph1)
            t2 = np.linspace(0.0, ph2, n_pts(r * ph2))
            p2 = J2 + r * np.column_stack([np.sin(ph1) - np.sin(ph1 - t2), np.cos(ph1 - t2) - np.cos(ph1)])
            pm = J1 + np.outer(np.linspace(0.0, 1.0, n_pts(l)), J2 - J1)
            return np.concatenate([p1, pm, p2]), J1, J2

        best = None
        for s in np.arange(0.0, s_cap + 1e-9, 0.25):
            E = np.array([-s, -W])                         # entry, on the outside edge of turn 1
            X = C + s * fwd(dth) + W * left(dth)           # exit, on the outside edge of turn 2
            V = X - E

            # candidate (r, ph1, l): the arcs and the straight have to cover the displacement V
            if Lm == 0:
                # turn - turn: closed form, V = r * (2 sin(ph1) - sin(dth), 1 + cos(dth) - 2 cos(ph1))
                a = V @ V
                b = V[0] * np.sin(dth) - V[1] * (1 + np.cos(dth))
                c = 2 * np.cos(dth) - 2
                u = (-b + np.sqrt(b * b - a * c)) / a      # u = 1 / r
                if u <= 0:
                    continue
                candidates = [(1 / u, np.arctan2(V[0] * u + np.sin(dth), 1 + np.cos(dth) - V[1] * u), 0.0)]
            else:
                # turn - straight - turn: one free parameter, so scan ph1 and try the largest radii first
                ph = np.linspace(max(dth, 0.0) + 0.005, 1.5, 300)
                A1 = 2 * np.sin(ph) - np.sin(dth)
                A2 = 1 + np.cos(dth) - 2 * np.cos(ph)
                det = 2 - np.cos(ph - dth) - np.cos(ph)
                r_all = (V[0] * np.sin(ph) - V[1] * np.cos(ph)) / det
                l_all = (A1 * V[1] - A2 * V[0]) / det
                order = [k for k in np.argsort(-r_all) if r_all[k] > 0 and l_all[k] >= 0][:40]
                candidates = [(r_all[k], ph[k], l_all[k]) for k in order]

            for r, ph1, l in candidates:
                ph2 = ph1 - dth
                if ph1 <= 0 or ph2 <= 0 or r <= max(R1, R2):
                    continue
                path, J1, J2 = racing_path(E, r, ph1, ph2, l)
                if np.hypot(*(path[-1] - X)) > 1e-3 or not on_track(path):
                    continue
                if best is None or r > best["r"]:
                    best = dict(s=s, r=r, ph1=ph1, ph2=ph2, l=l, E=E, X=X, J1=J1, J2=J2,
                                B1=B1, B2=B2, C=C, th1=th1, dth=dth)
                break
        return best

    # Writes a chicane racing line into the sections: radius, length, v_max, and Da / Db.
    # Da and Db are [dx, dy] in the section's frame (y forward along the heading, x to the right):
    # where the racing line starts, from the geometric start, and ends, from the geometric end.
    # m = +1 if the first turn is a left turn, -1 if it is a right turn.
    def chicane_apply(sec1, sec2, fit, m):
        def shift(p, origin, heading):
            d = p - origin
            return np.round(np.array([-m * (d @ np.array([-np.sin(heading), np.cos(heading)])),
                                      d @ np.array([np.cos(heading), np.sin(heading)])]), 9)  # rounded so 0 is exactly 0

        sec1["radius"], sec1["length"] = fit["r"], fit["r"] * fit["ph1"]
        sec2["radius"], sec2["length"] = fit["r"], fit["r"] * fit["ph2"]
        sec1["v_max"] = sec2["v_max"] = np.sqrt(limits.lat * fit["r"])
        sec1["Da"] = shift(fit["E"], np.zeros(2), 0.0)
        sec1["Db"] = shift(fit["J1"], fit["B1"], fit["th1"])
        sec2["Da"] = shift(fit["J2"], fit["B2"], fit["th1"])
        sec2["Db"] = shift(fit["X"], fit["C"], fit["dth"])
        return shift

    # Chicane of two turns back to back. The caller shortens the straights before and after by
    # s = -sec1["Da"][1]. smax is the most run-up (s) the neighboring straights can give.
    # Returns the sections unchanged if they are not a chicane or if there is no better racing line.
    def chicane_section(sec1, sec2, smax=math.inf):
        if sec1["type"] == "straight" or sec2["type"] == "straight" or sec1["type"] == sec2["type"]:
            return sec1, sec2
        fit = chicane_fit(sec1["radius_track"], sec1["length_track"] / sec1["radius_track"], 0.0,
                          sec2["radius_track"], sec2["length_track"] / sec2["radius_track"], smax)
        if fit is not None:
            chicane_apply(sec1, sec2, fit, 1 if sec1["type"] == "left" else -1)
        return sec1, sec2

    # Chicane of two turns with a straight between them (not used yet). The straight gets a new
    # length and its own Da / Db. The caller shortens the straights before sec1 and after sec2 by
    # s = -sec1["Da"][1], as for the turn - turn chicane.
    def chicane_section_with_straight(sec1, mid, sec2, smax=math.inf):
        if sec1["type"] == "straight" or sec2["type"] == "straight" or sec1["type"] == sec2["type"] \
                or mid["type"] != "straight":
            return sec1, mid, sec2
        fit = chicane_fit(sec1["radius_track"], sec1["length_track"] / sec1["radius_track"], mid["length_track"],
                          sec2["radius_track"], sec2["length_track"] / sec2["radius_track"], smax)
        if fit is not None:
            shift = chicane_apply(sec1, sec2, fit, 1 if sec1["type"] == "left" else -1)
            mid["length"] = fit["l"]
            mid["Da"] = shift(fit["J1"], fit["B1"], fit["th1"])
            mid["Db"] = shift(fit["J2"], fit["B2"], fit["th1"])
        return sec1, mid, sec2


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
            sec["Da"] = np.zeros(2)  # [dx, dy] of the racing line start from the geometric start (y forward, x right)
            sec["Db"] = np.zeros(2)  # same for the end

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
            if s > 0:
                # the racing line starts s before the turn and ends s after it, on the outside edge
                out = 1 if sec["type"] == "left" else -1  # outside edge: right (+x) of a left turn
                sec["Da"] = np.array([out * TRACK_WIDTH / 2, -s])
                sec["Db"] = np.array([out * TRACK_WIDTH / 2, s])

            # 4. the car is already turning s before the entry and until s after the exit,
            #    so the neighboring straights are s shorter
            if s > 0:
                prev_sec["Db"] = np.array([0.0, -s])
                next_sec["Da"] = np.array([0.0, s])
                prev_sec["length"] -= s
                next_sec["length"] -= s

        # Pass 3: chicanes. Straight, turn, turn, straight, with the turns bending opposite ways and neither
        # turning more than CHICANE_MAX_ANGLE: the two turns get one racing line together.
        for i in range(n - 4):
            a, b, c, d, e = sections[i:i + 5]
            if usable(a) and usable(d) and b["type"] != "turn" and c["type"] != "straight" and b["type"] != c["type"]:

                if max(b["length_track"] / b["radius_track"], c["length_track"] / c["radius_track"]) > CHICANE_MAX_ANGLE:
                    continue

                # run-up limited by half of each straight, as for a single turn
                smax = min(a["length_track"], d["length_track"]) / 2
                b, c = chicane_section(b, c, smax)

                # the car is already turning s before the first turn and until s after the second one
                s = -b["Da"][1]
                if s > 0:
                    a["Db"] = np.array([0.0, -s])
                    d["Da"] = np.array([0.0, s])
                    a["length"] -= s
                    d["length"] -= s


    # explanation TODO:

    def sufficient_braking_distance(j0, d0, v0):
        v = v0
        E = 0

        for j in range(j0, n_sections):
            S = sections[j]
            d = S["length"] if j != j0 else d0
            
            if S["type"] != "straight" and S["radius"] != 0:
                a_lat = v ** 2 / S["radius"]
                a_lon = limits.lon * np.sqrt(max(0.0, 1 - (a_lat / limits.lat) ** 2))
            else:
                a_lon = limits.lon
            a = min(a_lon, limits.brake)
            E += a * d

            vj = S["v_max"]
            if 0.5 * (v0 * v0 - vj * vj) > E:
                #print(f"i said no! cuz of section {j0}, {j}")
                return False
               
            if 0.5*v0*v0 < E:
                #print(f"damn alright, after section {j0}, {j}")
                return True

            v = np.sqrt(v*v - 2*a*d)

        return True


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
    state.v = 13

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
        a_lon = limits.lon
        if section["type"] != "straight" and section["radius"] != 0:
            a_lat = v_old ** 2 / section["radius"]
            a_lon = limits.lon * np.sqrt(max(0.0, 1 - (a_lat / limits.lat) ** 2))
    
        # try accel
        a_accel = min(a_lon, limits.accel, limits.pack_limited_accel(v_to_rpm(v_old), car))
        v_accel = min(v_motor_max, v_old + a_accel * DT) 
        if sufficient_braking_distance(state.section_idx, section["length"] - state.section_traveled - v_accel * DT, v_accel):
            v_accel = min(v_motor_max, v_old + a_accel * DT) 
            v_new = v_accel
        elif sufficient_braking_distance(state.section_idx, section["length"] - state.section_traveled - v_old * DT, v_old):
            v_new = v_old
        else:
            v_new = v_old - min(a_lon, limits.brake) * DT

        if v_new < 0:
            raise RuntimeError(f"time stepped velocity turned negative on section {state.section_idx}")

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
                          f"{state.v:.1f} m/s, above its limit of {v_limit:.1f} m/s. correcting")
                    state.v = v_limit

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
    limit.lon = 1.8 * 9.8
    limit.brake = 1.7 * 9.8
    limit.accel = 200 * car.FDR / (car.mass * car.wheelR)
    limit.power = 80e3   # W
    limit.motor = 5000   # rpm
    limit.regen_torque = -30 # Nm

    # guesstimated top formula car
    # car = Car()
    # car.mass = 250 # kg
    # car.FDR = 3.91
    # car.wheelR = 0.2  # m
    #
    # limit = LapLimits()
    # limit.lat = 2 * 9.8
    # limit.brake = -2.2 * 9.8
    # limit.accel = 2.5 * 9.8
    # limit.power = 80e3   # W
    # limit.motor = 6000   # rpm
    # limit.regen_torque = -30 # Nm


    print(f"[main] Accel limit: {limit.accel:.2f} m/s2")

    lap_data = lap_sim_track("2022 Michigan Endurance June - One Lap.csv", car, limit)
    lap_data = lap_sim_powertrain(car, lap_data)
    lap_sim_visualize(lap_data)
