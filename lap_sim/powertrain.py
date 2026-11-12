import csv
import time
import numpy as np

from motor_efficiency import F as motor_efficiency
from inverter_loss import inverter_loss

from models import *

def sim_powertrain(car, limit, data, *, verbose=True):

    def log(*args, **kwargs):
        if verbose:
            return print(*args, **kwargs)

    # Now with all the driving states calculated, let's find some PT-relevant stats
    log("=" * 60)
    log("[powertrain] START")

    dt = float(np.mean(np.diff(data["t"])))

    # == Motor Power ==
    # 1. motor torque: DT eff depends on if we are regen'ing
    # 2. max regen torque: I = P / V = k*tau*w / V. 
    # 3. motor power: clamp motor torque and k*tau*w

    wheel_torque = data["torque"]
    motor_torque = wheel_torque / car.FDR * np.where(wheel_torque > 0, 1/car.drivetrain_eff, car.drivetrain_eff)
    motor_torque = np.maximum(limit.regen_torque_max(data["rpm"]), motor_torque)
    motor_power = motor_torque * data["rpm"] * 2 * np.pi / 60

    # == Find V & I ==
    # TODO: inverter loss is a manufacturer-given function of V and I. This makes it difficult to solve in
    # reverse. I've simplified it to assume 500W losses average, which is around the average for most run profiles
    inverter_power = 500
    motor_e = motor_efficiency(data["rpm"], wheel_torque / car.FDR)
    pack_power = np.where(motor_power >= 0, motor_power / motor_e + inverter_power, motor_power * motor_e - inverter_power)

    energy_used = float(np.sum(pack_power) * dt)  # J


    # solve for current and voltage
    voltage = np.zeros((len(pack_power)))
    current = np.zeros((len(pack_power)))
    for i, pi in enumerate(pack_power):
        packv = car.pack_voltage();
        packIR = car.pack_IR();

        # solve for current accounting for voltage sag
        if pi > 0:
            r = np.roots([-packIR, packv, -pi])
            current[i] = min(r)
        else:
            r = np.roots([ packIR, packv, -pi])
            current[i] = max(r)

        # calculate voltage
        voltage[i] = pi / current[i] if current[i] != 0 else packv


        # print(np.roots([-packIR, packv, -pi]))
        # print(current[i], voltage[i], pi)
        # assert False

    # Very important number
    current_rms = np.sqrt(np.mean(current**2))

    # == heat loads ==
    pack_heat = current_rms * current_rms * car.pack_IR()
    motor_heat = np.where(motor_power >= 0, motor_power * (1-motor_e) / motor_e, -motor_power * (1-motor_e))
    motor_heat = np.mean(motor_heat)
    inverter_heat = inverter_loss(voltage, np.abs(current))
    inverter_heat = np.mean(inverter_heat)

    # == log results ==
    drive_energy = float(np.sum(pack_power[pack_power > 0]) * dt)
    regen_energy = float(-np.sum(pack_power[pack_power < 0]) * dt)
    log(f"[powertrain] net energy used:   {energy_used / 1e6:.3f} MJ  "
          f"({energy_used / 3600:.0f} Wh, {energy_used / 3.6e6:.3f} kWh)")
    log(f"[powertrain] drive / regen:     {drive_energy / 3600:.0f} Wh / {regen_energy / 3600:.0f} Wh")
    log(f"[powertrain] peak pack power:   {pack_power.max() / 1000:.1f} kW")
    log(f"[powertrain] current RMS:       {current_rms:.2f} A")
    log(f"[powertrain] heat, pack:        {pack_heat:.1f} W")
    log(f"[powertrain] heat, motor:       {motor_heat:.1f} W")
    log(f"[powertrain] heat, inverter:    {inverter_heat:.1f} W")
    log("[powertrain] END")

    # == output ==
    out = {
        **data,
        "motor_torque": motor_torque,
        "pack_power": pack_power,
        "voltage": voltage,
        "current": current,
        "energy_used": energy_used,
        "current_rms": current_rms,
        "pack_heat": pack_heat,
        "motor_heat": motor_heat,
        "inverter_heat": inverter_heat,
    }
    return out
