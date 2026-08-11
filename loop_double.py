import numpy as np
import matplotlib.pyplot as plt

from loop_components import emrax228, cm200dx
from loop_helper import solve_system, draw_system
from radiator_helper import sim_rad_in_loop

def using(radiator, pump, fan, plot=True):
    def system1(lpm):
        return cm200dx(lpm) + radiator.flow(lpm);

    def pump1(lpm):
        return pump.flow(lpm);

    def pump_power1(lpm):
        return pump.power(lpm);

    def system2(lpm):
        return emrax228(lpm) + radiator.flow(lpm);

    def pump2(lpm):
        return pump.flow(lpm);

    def pump_power2(lpm):
        return pump.power(lpm);

    # Generate x-axis data (0 to 30 with 0.1 increments)
    x = np.arange(0, 30, 0.1)
    system1_op_lpm = solve_system(x, pump1, system1);
    system2_op_lpm = solve_system(x, pump2, system2);

    Q1_cold_op = sim_rad_in_loop(radiator, 500 * 1.25, system1_op_lpm * (1.0 / 60000.0), 50.0, 20.0, plot)
    fan1_op_power = fan.power(Q1_cold_op /  0.0004719474)
    pump1_op_power = pump_power1(system1_op_lpm) 
    system1_op_power = pump1_op_power + fan1_op_power

    Q2_cold_op = sim_rad_in_loop(radiator, 1000 * 1.25, system2_op_lpm * (1.0 / 60000.0), 50.0, 20.0, plot)
    fan2_op_power = fan.power(Q2_cold_op / 0.0004719474)
    pump2_op_power = pump_power2(system2_op_lpm) 
    system2_op_power = pump2_op_power + fan2_op_power

    if not plot:
        return (system1_op_lpm, system2_op_lpm), pump1_op_power + pump2_op_power, fan1_op_power + fan2_op_power

    print("=== System 1 ===")
    print(f"At hydraulic operating point ({system1_op_lpm:.2f} LPM)")
    print(f"pump system consumes {pump1_op_power:.2f} W, fan consumes {fan1_op_power:.2f} W. Total {system1_op_power:.2f} W")
    print("=== System 2 ===")
    print(f"At hydraulic operating point ({system2_op_lpm:.2f} LPM)")
    print(f"pump system consumes {pump2_op_power:.2f} W, fan consumes {fan2_op_power:.2f} W. Total {system2_op_power:.2f} W")
    print(f"=== Total power consumed: {system1_op_power + system2_op_power:.2f} W ===")

    # ---------------------------------------------------------
    # Figure 1: Component Curves
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(x, emrax228(x), label='Motor EMRAX 228')
    plt.plot(x, cm200dx(x), label='Inverter CM200DX')
    plt.plot(x, radiator.flow(x), label='select radiator')
    plt.plot(x, pump.flow(x), 'g-', label='select pump')
    plt.plot(x, list(map(pump.power,x)), label='select pump power')

    plt.xlabel('LPM')
    plt.ylabel('Pressure (bar)')
    plt.title('Component Curves')
    plt.ylim([0, 5])
    plt.grid(True)
    plt.legend()

    # ---------------------------------------------------------
    # Figure 2: System1 Curves
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    draw_system(ax1, x, pump1, system1, system1_op_lpm)
    ax1.axvline(12, color='g', linestyle='--', label='inverter spec')
    ax1.set_title("System 1: Inverter")
    ax1.legend()
     
    # ---------------------------------------------------------
    # Figure 3: System2 Curves
    # ---------------------------------------------------------
    draw_system(ax2, x, pump2, system2, system2_op_lpm)
    ax2.axvline(6, color='b', linestyle='--', label='motor spec')
    ax2.set_title("System 2: Motor")
    ax2.legend()
     
    plt.tight_layout()
    plt.show();

if __name__ == "__main__":
    import radiators
    import pumps
    import fans

    using(radiators.blackice_280, pumps.boschePCE1, fans.orion1238, True)

