"""
===========================================================
utils.py
-----------------------------------------------------------
Utility Functions

Contains helper functions used by

• Simulation
• Plotting
• Metrics
• Stability
• Main

===========================================================
"""

import os
import numpy as np
import csv

from parameters import *

# ==========================================================
# Directory Creation
# ==========================================================

def create_directories():
    """
    Create output folders if they do not exist.
    """

    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)


# ==========================================================
# Angle Conversion
# ==========================================================

def deg2rad(angle):
    return np.deg2rad(angle)


def rad2deg(angle):
    return np.rad2deg(angle)


# ==========================================================
# Clamp
# ==========================================================

def clamp(value, minimum, maximum):

    return max(
        minimum,
        min(value, maximum)
    )


# ==========================================================
# Normalize
# ==========================================================

def normalize(data):

    data = np.asarray(data)

    minimum = np.min(data)

    maximum = np.max(data)

    if maximum == minimum:
        return np.zeros_like(data)

    return (
        data - minimum
    ) / (
        maximum - minimum
    )


# ==========================================================
# Moving Average
# ==========================================================

def moving_average(data, window=5):

    kernel = np.ones(window) / window

    return np.convolve(
        data,
        kernel,
        mode="same"
    )


# ==========================================================
# Numerical Derivative
# ==========================================================

def derivative(signal, dt):

    return np.gradient(
        signal,
        dt
    )


# ==========================================================
# Numerical Integral
# ==========================================================

def integral(signal, dt):

    return np.cumsum(
        signal
    ) * dt


# ==========================================================
# Save Dictionary
# ==========================================================

def save_dictionary(dictionary,
                    filename):

    with open(CSV_DIR + filename, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Metric", "Value"])
        writer.writerows(dictionary.items())


# ==========================================================
# Print Dictionary
# ==========================================================

def print_dictionary(dictionary):

    print()

    print("=" * 55)

    for key, value in dictionary.items():

        if isinstance(value, float):

            print(
                f"{key:<35}"
                f"{value:>12.4f}"
            )

        else:

            print(
                f"{key:<35}"
                f"{value}"
            )

    print("=" * 55)


# ==========================================================
# Progress Bar
# ==========================================================

def progress(current,
             total):

    percent = (
        current / total
    ) * 100

    print(
        f"\rProgress : "
        f"{percent:6.2f} %",
        end=""
    )


# ==========================================================
# Simulation Header
# ==========================================================

def print_header():

    print()

    print("=" * 60)

    print(" Intelligent Smart Wheelchair Simulation ")

    print("=" * 60)


# ==========================================================
# Simulation Footer
# ==========================================================

def print_footer():

    print()

    print("=" * 60)

    print(" Simulation Completed ")

    print("=" * 60)


# ==========================================================
# Controller Name
# ==========================================================

def controller_title(name):

    mapping = {

        "pid":
            "PID Controller",

        "nonlinear_pid":
            "Nonlinear PID Controller",

        "fuzzy":
            "Fuzzy Logic Controller",

        "smc":
            "Sliding Mode Controller"

    }

    return mapping.get(
        name.lower(),
        name
    )


# ==========================================================
# Check Stability
# ==========================================================

def safety_color(state):

    if state in ("Stable", "Static screens passed"):

        return "green"

    elif state in ("Slip", "Traction screen failed"):

        return "orange"

    else:

        return "red"


# ==========================================================
# Maximum Absolute
# ==========================================================

def max_abs(signal):

    return np.max(
        np.abs(signal)
    )


# ==========================================================
# Percentage Error
# ==========================================================

def percentage_error(actual,
                     theoretical):

    if theoretical == 0:

        return 0

    return (
        abs(actual - theoretical)
        /
        theoretical
        * 100
    )


# ==========================================================
# Time Formatting
# ==========================================================

def format_time(seconds):

    minutes = int(seconds // 60)

    seconds = seconds % 60

    return f"{minutes} min {seconds:.2f} sec"


# ==========================================================
# Banner
# ==========================================================

def banner():

    print("""
=============================================================
        SMART WHEELCHAIR DYNAMICS SIMULATOR
-------------------------------------------------------------
Dynamic Modeling
Wheelchair Stability
Controller Comparison
Energy Analysis
Publication Quality Graphs
=============================================================
""")
