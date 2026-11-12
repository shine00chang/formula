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

import matplotlib.pyplot as plt

from track import sim_track 
from powertrain import sim_powertrain
from visualize import visualize, visualize_on_track
from models import *

if __name__ == "__main__":
    print("==Lap Sim==")

    limit = LapLimits()
    limit.lat = 1.8 * 9.8
    limit.lon = 1.8 * 9.8
    limit.brake = 0.4 * 9.8
    limit.accel = 200 * mk12.drivetrain_eff * mk12.FDR / (mk12.mass * mk12.wheelR)
    limit.power = 70e3   # W
    limit.motor = 6000   # rpm
    #limit.regen_torque_max = lambda rpm: -car.charge_current_max * car.pack_voltage() / (2 * np.pi / 60 * rpm) 
    limit.regen_torque_max = lambda rpm: np.ones(np.shape(rpm)) * -40

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

    lap_data = sim_track("2022 Michigan Endurance June - One Lap.csv", mk12, limit)
    lap_data = sim_powertrain(mk12, limit, lap_data)
    visualize(lap_data)
    visualize_on_track(lap_data)

    plt.show()
