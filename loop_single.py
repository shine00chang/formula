import numpy as np
import matplotlib.pyplot as plt

# Assuming these are imported from your local environment
from loop_components import emrax228, cm200dx
from loop_helper import solve_system, draw_system
from radiator_helper import sim_rad_in_loop

def using(radiator, pump, fan, plot=True):
    def system(lpm):
        return cm200dx(lpm) + emrax228(lpm) + radiator.flow(lpm)

    def pump_f(lpm):
        return pump.flow(lpm) * 2

    def pump_p(lpm):
        return pump.power(lpm) * 2

    x = np.arange(0, 30.1, 0.1)
    system_op_lpm = solve_system(x, pump_f, system)

    pump_op_power = pump_p(system_op_lpm)

    Q_cold_op = sim_rad_in_loop(radiator, 1500 * 1.25, system_op_lpm * (1.0 / 60000.0), 50.0, 20.0, plot)
    fan_cfm = Q_cold_op / 0.0004719474
    fan_op_power = fan.power(fan_cfm)
    system_op_power = pump_op_power + fan_op_power

    if not plot:
        # return system_op_lpm, pump_op_power, fan_op_power
        return system_op_lpm, pump_op_power, fan_cfm

    print("===")
    print(f"At hydraulic operating point ({system_op_lpm:.2f} LPM)")
    print(f"pump system consumes {pump_op_power:.2f} W, fan consumes {fan_op_power:.2f} W. Total {system_op_power:.2f} W")
    print("===")

    # ---------------------------------------------------------
    # Figure 1: Component Impedance Curves
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(x, emrax228(x), label='Motor EMRAX 228')
    plt.plot(x, cm200dx(x), label='Inverter CM200DX')
    plt.plot(x, radiator.flow(x), label='select radiator')
    plt.plot(x, pump.flow(x), 'g-', label='select pump')
    plt.plot(x, [pump.power(val) for val in x], linestyle=':', label='select pump power')

    plt.xlabel('Flow Rate (LPM)')
    plt.ylabel('Pressure Drop (bar) / Power (W)')
    plt.title('Component Hydraulic & Power Curves')
    plt.ylim([0, 5])
    plt.grid(True)
    plt.legend()

    # ---------------------------------------------------------
    # Figure 2: System Operating Point
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_system(ax, x, pump_f, system, system_op_lpm)
    ax.axvline(6, color='b', linestyle='--', label='Motor Min Spec')
    ax.axvline(12, color='g', linestyle='--', label='Inverter Min Spec')
    ax.set_title('Hydraulic System Operating Point')
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import radiators
    import pumps
    import fans

    using(radiators.blackice_280, pumps.boschePCE1, fans.orion1238, True)

