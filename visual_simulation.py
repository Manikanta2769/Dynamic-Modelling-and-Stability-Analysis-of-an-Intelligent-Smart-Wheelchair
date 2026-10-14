"""Compatibility entry point for the maintained PyBullet co-simulation."""
from run_co_simulation import run_wheelchair_stability_analyzer


def run_pybullet_simulation(urdf_path=None, target_speed=1.2, duration=15.0):
    """Run the shared PyBullet implementation (legacy function name)."""
    return run_wheelchair_stability_analyzer(
        controller_type="Linear PID", urdf_path=urdf_path,
        user_speed=target_speed, sim_time=duration)


if __name__ == "__main__":
    run_pybullet_simulation()
