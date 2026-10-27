import math

def in_to_m(i: float) -> float:
    return i * 0.0254

def m_to_in(i: float) -> float:
    return i / 0.0254

def lpm_to_m3s(lpm: float) -> float:
    return lpm / 60.0 * 0.001

def constricted_v(lpm: float, d1: float, d2: float) -> float:
    return lpm * (d1 * d1) / (d2 * d2)

def p_major(lpm: float, l: float, d: float, print_out: float = 0.0) -> float:
    if print_out > 0:
        print(f'Pipe dimensions: {m_to_in(d):.2f}"ID for {m_to_in(l):.2f}" @ {lpm:.2f} LPM')
        
    rho = 1000.0
    
    a = math.pi * d * d / 4.0
    v = lpm_to_m3s(lpm) / a  # flow rate from lpm to m^3/s
    
    f = 0.02  # attained from moody table
    
    p = f * l * rho * v * v / (2.0 * d)
    
    if print_out > 0:
        print(f"Expected pressure loss:\n{p:.0f} Pa\n{p/1e5:.3f} bar\n")
        
    return p / 1e5

def p_minor(lpm: float, k: float, d: float, print_out: float = 0.0) -> float:
    if print_out > 0:
        print(f'Minor coefficient: {k:.2f} for {m_to_in(d):.2f}"ID @ {lpm:.2f} LPM')
        
    rho = 1000.0
    
    a = math.pi * d * d / 4.0
    v = lpm_to_m3s(lpm) / a
    
    p = k * rho * v * v / 2.0
    
    if print_out > 0:
        print(f"Expected pressure loss:\n{p:.0f} Pa\n{p/1e5:.3f} bar\n")
        
    return p / 1e5

def solve_system(x, pump, system):

    system_op_lpm = 0.0

    # Calculate system values across the range
    for lpm in x:
        if system(lpm) < pump(lpm):
            system_op_lpm = lpm

    return system_op_lpm


def draw_system(ax, x, pump, system, system_op_lpm):

    ax.plot(x, system(x), 'r', label='System')
    ax.plot(x, pump(x), 'g', label='Pumps')
    ax.axvline(system_op_lpm, color='k', linestyle='--', label='operating point')

    ax.set_xlabel('LPM')
    ax.set_ylabel('Pressure (bar)')
    ax.set_title('System Curves')
    ax.set_ylim([0, 5])
    ax.set_xlim([0, 30])
    ax.grid(True)
    ax.legend()
