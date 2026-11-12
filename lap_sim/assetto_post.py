# Import Assetto Corsa Motec data, and apply powertrain math on it to produce the same logs
# Used for general post-processing of data, but also checking correlation between machine and Tyler (not a machine)

import csv
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List

from models import *
from visualize import visualize
from powertrain import sim_powertrain 


def parse_motec_csv(filepath: str) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Parses a MoTeC CSV export into column names, units, and a 2D NumPy matrix.
    """
    headers: List[str] = []
    units: List[str] = []
    data_rows: List[List[float]] = []

    with open(filepath, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        state = "METADATA"

        for row in reader:
            # Skip completely empty rows
            if not row or all(cell.strip() == "" for cell in row):
                continue

            if state == "METADATA":
                # Locate the channel header row (starts with "Time")
                if row[0].strip() == "Time":
                    headers = [cell.strip() for cell in row]
                    state = "UNITS"

            elif state == "UNITS":
                # The immediate next non-empty row contains the channel units
                units = [cell.strip() for cell in row]
                state = "DATA"

            elif state == "DATA":
                # Parse numeric telemetry rows
                data_rows.append([float(cell) for cell in row])

    data_matrix = np.array(data_rows, dtype=np.float64)
    return headers, units, data_matrix


# Grab data
filename = "datasets/mk11_150Nm_30NmRegen_STANDARDPACE.csv";
headers, units, matrix = parse_motec_csv(filename)

print("Columns:", headers)
print("Units:  ", units)
print("Matrix shape:", matrix.shape)

cols = {name: matrix[:, i] for i, name in enumerate(headers)}
time = cols["Time"]
rpm = cols["Engine RPM"]
motor_torque = cols["Engine Torque"]
motor_power = cols["Motor Power"]
throttle = cols["Driver Throttle Pos"]

limit = LapLimits()
limit.lat = 1.8 * 9.8
limit.lon = 1.8 * 9.8
limit.brake = 0.7 * 9.8
limit.accel = 150 * mk12.drivetrain_eff * mk12.FDR / (mk12.mass * mk12.wheelR)
limit.power = 70e3   # W
limit.motor = 6000   # rpm
limit.regen_torque_max = lambda rpm: np.ones(np.shape(rpm)) * -30

out = {
    "t": time,
    "v": None,
    "a": None,
    "distance": None,
    "sections": None,
    "torque": motor_torque * mk12.FDR,
    "rpm": rpm,
    "lap_time": time[-1]
}

out = sim_powertrain(mk12, limit, out)
visualize(out)

plt.show()
