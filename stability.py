"""
===========================================================
stability.py
-----------------------------------------------------------
Smart Wheelchair Stability Analysis

Provides static screening utilities for center of gravity, directional
adhesion, slope-induced slip, support-edge tip-over, and normalized screen
margin. It does not implement a complete stability-pyramid analysis or a
dynamic safety certification.

Reference:
Simulation Analysis on Motion Stability of Smart Wheelchair
(Journal of Physics: Conference Series, 2021)

===========================================================
"""

import numpy as np

from parameters import *


def _validate_static_inputs(slope_deg, cross_slope_deg, phi=None):
    if (not np.isfinite(slope_deg) or not np.isfinite(cross_slope_deg)
            or abs(slope_deg) >= 90 or abs(cross_slope_deg) >= 90):
        raise ValueError("slope angles must be finite and have magnitude below 90 deg")
    if phi is not None and (not np.isfinite(phi) or phi < 0):
        raise ValueError("friction coefficient phi must be finite and nonnegative")

# ==========================================================
# Center of Gravity
# ==========================================================

def compute_cog(masses, coordinates):
    """
    Eq. (1)

    masses      : list or ndarray
    coordinates : Nx3 matrix

    Returns
    -------
    x,y,z of combined center of gravity
    """

    masses = np.asarray(masses)
    coordinates = np.asarray(coordinates)

    if masses.ndim != 1 or coordinates.shape != (masses.size, 3):
        raise ValueError("coordinates must have shape (number_of_masses, 3)")
    if (masses.size == 0 or not np.all(np.isfinite(masses))
            or not np.all(np.isfinite(coordinates)) or np.any(masses < 0)):
        raise ValueError("masses and coordinates must be finite, with nonnegative masses")

    total_mass = np.sum(masses)
    if total_mass <= 0:
        raise ValueError("total mass must be positive")

    x = np.sum(masses * coordinates[:, 0]) / total_mass
    y = np.sum(masses * coordinates[:, 1]) / total_mass
    z = np.sum(masses * coordinates[:, 2]) / total_mass

    return np.array([x, y, z])


# ==========================================================
# Adhesion Coefficient
# ==========================================================

def adhesion_coefficient(phi_longitudinal,
                         phi_lateral,
                         azimuth_deg):
    """
    Eq. (2)
    """

    theta = np.deg2rad(azimuth_deg)

    return np.sqrt(
        (phi_longitudinal * np.cos(theta))**2 +
        (phi_lateral * np.sin(theta))**2
    )


# ==========================================================
# Maximum Slip Angle
# ==========================================================

def maximum_slip_angle(phi):
    """
    Eq. (4)
    """

    if not np.isfinite(phi) or phi < 0:
        raise ValueError("friction coefficient phi must be finite and nonnegative")
    return np.rad2deg(np.arctan(phi))


# ==========================================================
# Slip Detection
# ==========================================================

def check_slip(slope_deg,
               phi,
               cross_slope_deg=0.0):
    """
    Returns

    True -> Slip

    False -> No Slip
    """

    _validate_static_inputs(slope_deg, cross_slope_deg, phi)
    resultant_slope = np.rad2deg(np.arctan(np.hypot(
        np.tan(np.deg2rad(slope_deg)),
        np.tan(np.deg2rad(cross_slope_deg)))))
    critical = maximum_slip_angle(phi)
    return resultant_slope > critical


# ==========================================================
# Stability Margin
# ==========================================================

def stability_margin(
        distance,
        cog_height):
    """
    Eq. (5)

    eta = atan(distance/H)
    """

    return np.rad2deg(
        np.arctan(
            distance /
            cog_height
        )
    )


# ==========================================================
# Tip-over Margin
# ==========================================================

def tip_margin():

    front = wheel_base / 2 - cog_x
    rear = wheel_base / 2 + cog_x

    left = track_width / 2 - cog_y
    right = track_width / 2 + cog_y

    eta_front = stability_margin(
        front,
        cog_height
    )

    eta_rear = stability_margin(
        rear,
        cog_height
    )

    eta_left = stability_margin(
        left,
        cog_height
    )

    eta_right = stability_margin(
        right,
        cog_height
    )

    return min(
        eta_front,
        eta_rear,
        eta_left,
        eta_right
    )


# ==========================================================
# Tip-over Detection
# ==========================================================

def check_tipover(
        slope_deg,
        cross_slope_deg=0.0):
    """
    Returns True if tipping occurs.
    """

    _validate_static_inputs(slope_deg, cross_slope_deg)
    # Gravity moves the support resultant downhill from the CoG.
    shift_x = -cog_height * np.tan(np.deg2rad(slope_deg))
    shift_y = -cog_height * np.tan(np.deg2rad(cross_slope_deg))
    x_limit = wheel_base/2 - cog_x if shift_x >= 0 else wheel_base/2 + cog_x
    y_limit = track_width/2 - cog_y if shift_y >= 0 else track_width/2 + cog_y
    return abs(shift_x) > x_limit or abs(shift_y) > y_limit


# ==========================================================
# Stability Index
# ==========================================================

def stability_index(
        slope_deg,
        phi,
        cross_slope_deg=0.0):
    """
    Static traction/tip-over screening proxy. This is not a dynamic safety metric.

    1 = no modeled static demand

    0 = at a screen boundary

    Negative = outside a static screen
    """

    _validate_static_inputs(slope_deg, cross_slope_deg, phi)
    slip_ratio = np.hypot(
        np.tan(np.deg2rad(slope_deg)),
        np.tan(np.deg2rad(cross_slope_deg))) / max(float(phi), 1e-12)
    shift_x = -cog_height * np.tan(np.deg2rad(slope_deg))
    shift_y = -cog_height * np.tan(np.deg2rad(cross_slope_deg))
    x_limit = wheel_base/2 - cog_x if shift_x >= 0 else wheel_base/2 + cog_x
    y_limit = track_width/2 - cog_y if shift_y >= 0 else track_width/2 + cog_y
    tip_ratio = max(abs(shift_x)/x_limit, abs(shift_y)/y_limit)
    return float(min(1.0-slip_ratio, 1.0-tip_ratio))


# ==========================================================
# Stability State
# ==========================================================

def stability_state(
        slope_deg,
        phi,
        cross_slope_deg=0.0):
    """
    Classification

    Stable

    Slip

    Tip-over
    """

    slip = check_slip(
        slope_deg,
        phi,
        cross_slope_deg
    )

    tip = check_tipover(
        slope_deg,
        cross_slope_deg
    )

    if tip:
        return "Tip-over screen failed"
    elif slip:
        return "Traction screen failed"
    else:
        return "Static screens passed"


# ==========================================================
# Safe Operating Region
# ==========================================================

def safe_region(cross_slope_deg=0.0):

    result = []

    for angle in terrain_angles:

        state = stability_state(
            angle,
            mu_static,
            cross_slope_deg
        )

        result.append(
            (angle, state)
        )

    return result


# ==========================================================
# Report
# ==========================================================

def stability_report(cross_slope_deg=0.0):

    print("\n========== Stability Report ==========")

    print(
        f"Maximum Slip Angle : "
        f"{maximum_slip_angle(mu_static):.2f} deg"
    )

    print(
        f"Tip Margin         : "
        f"{tip_margin():.2f} deg"
    )

    print(f"Cross-slope        : {cross_slope_deg:.2f} deg")
    print("Scope              : static traction/tip-over screens only; not dynamic validation")

    print("\nTerrain Analysis")

    for angle, state in safe_region(cross_slope_deg):

        print(
            f"{angle:2d} deg --> {state}"
        )

    print("======================================")
