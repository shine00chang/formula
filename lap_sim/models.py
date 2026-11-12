import numpy as np

from typing import Callable
from dataclasses import dataclass, replace

# == Global sim settings ==
DT = 0.01                 # s, time step
LOG_INTERVAL_S = 5.0      # s of *simulated* time between console logs
MAX_SIM_TIME = 1200.0     # s, safety stop if the sim stalls (e.g. lap starts in a corner)
CORNER_ENTRY_TOL = 1.02   # warn if corner entry speed is >2% over the corner's max speed
TRACK_WIDTH = 2           # m, to determine racing line
MIN_STRAIGHT = 10.0       # m, a turn only gets a racing line if the straights on both sides are at least this long
CHICANE_MAX_ANGLE = np.pi / 4  # rad, two back to back turns only count as a chicane if both turn this angle or less


class Car:
    mass: float  # kg

    FDR: float
    wheelR: float  # m
    coeff_rolling_resistance: float
    coeff_drag_area: float # Cd * m2
    drivetrain_eff: float

    pack_voltage: Callable[[], float]
    pack_IR: Callable[[], float]
    charge_current_max: float # A


class LapLimits:
    lat: float    # m/s2
    lon: float    # m/s2

    brake: float  # m/s2, negative
    accel: float  # m/s2
    power: float  # W, pack power limit
    motor: float  # rpm
    regen_torque_mae: Callable[[float], float] # Nm

    def pack_limited_accel(self, rpm, car):
        # return maximum acceleration under the pack power limit, at that rpm
        omega = rpm * 2 * np.pi / 60
        if omega < 1e-9:
            return math.inf  # no power limit at standstill; tire limit (accel) takes over

        # 0.95 represent the estimated motor efficiency at this top end.
        motor_torque = self.power * 0.95  / omega
        wheel_force = motor_torque * car.FDR / car.wheelR * car.drivetrain_eff
        return wheel_force / car.mass


mk12 = Car()
mk12.mass = 285 # kg
mk12.FDR = 3.71
mk12.wheelR = 0.2  # m
mk12.coeff_rolling_resistance = 0.016 
mk12.coeff_drag_area = 1.38 # m2
mk12.drivetrain_eff = 0.928
# TODO: Technically, go off of V/SoC graph, however, thats not gonna help unless you
# run the lap sim for all 22 laps. Instead, just grab an eyeball voltage
mk12.pack_voltage = lambda : 4.2 * 114
mk12.pack_IR = lambda : 0.017 * 114 / 6 + 0.05
mk12.charge_current_max = 9 *6
