"""Spatial terrain and support-polygon screening for the wheelchair model.

This is a quasi-static load-transfer calculation using longitudinal/lateral
acceleration. It is not a six-DOF body-pitch/roll simulation.
"""

import numpy as np

from parameters import *


def assess_spatial_stability(grade_deg, cross_slope_deg, acceleration_x,
                             acceleration_y, wheel_torque):
    """Estimate CoP, four corner normal loads and combined traction demand.

    Axes are aligned with the wheelchair on the local ground plane: +x uphill
    and +y toward the positive cross-slope. Negative corner loads indicate
    that the quasi-static resultant lies outside the support rectangle.
    """
    ax, ay, torque = np.broadcast_arrays(
        np.asarray(acceleration_x, dtype=float),
        np.asarray(acceleration_y, dtype=float),
        np.asarray(wheel_torque, dtype=float),
    )
    # Use the exact local terrain basis from the dynamics model, including
    # combined grade and cross-slope, so load/traction screening shares axes.
    from dynamics import terrain
    normal, tangent_x, tangent_y = terrain(grade_deg, cross_slope_deg)
    gravity = np.array([0.0, 0.0, -g])
    gravity_x = float(np.dot(gravity, tangent_x))
    gravity_y = float(np.dot(gravity, tangent_y))
    gravity_normal = -float(np.dot(gravity, normal))
    gravity_normal = max(gravity_normal, 1e-9)

    cop_x = cog_x + cog_height * (gravity_x - ax) / gravity_normal
    cop_y = cog_y + cog_height * (gravity_y - ay) / gravity_normal
    normal_total = mass_total * gravity_normal

    # Bilinear load split over an assumed rectangular four-wheel support.
    front_fraction = 0.5 + cop_x / wheel_base
    rear_fraction = 0.5 - cop_x / wheel_base
    right_fraction = 0.5 + cop_y / track_width
    left_fraction = 0.5 - cop_y / track_width
    loads = {
        "normal_front_right": normal_total * front_fraction * right_fraction,
        "normal_front_left": normal_total * front_fraction * left_fraction,
        "normal_rear_right": normal_total * rear_fraction * right_fraction,
        "normal_rear_left": normal_total * rear_fraction * left_fraction,
    }

    margin_x = wheel_base / 2 - np.abs(cop_x)
    margin_y = track_width / 2 - np.abs(cop_y)
    margin_min = np.minimum(margin_x, margin_y)

    # Aggregate friction demand. Lateral demand balances gravity and imposed
    # lateral acceleration; longitudinal drive demand comes from wheel torque.
    force_x = torque / wheel_radius
    force_y = mass_total * (ay - gravity_y)
    friction_capacity = mu_static * normal_total
    friction_use = np.hypot(force_x, force_y) / max(friction_capacity, 1e-9)

    return {
        **loads,
        "cop_x": cop_x,
        "cop_y": cop_y,
        "support_margin_longitudinal": margin_x,
        "support_margin_lateral": margin_y,
        "support_margin_min": margin_min,
        "support_polygon_screen_safe": margin_min >= 0,
        "friction_utilization_3d": friction_use,
        "combined_traction_screen_safe": friction_use <= 1.0,
    }
