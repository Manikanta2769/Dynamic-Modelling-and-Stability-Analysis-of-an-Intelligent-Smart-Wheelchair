"""Entry point for the smart-wheelchair 3-D dynamics and control project.

Examples (run from this folder)::

    python main.py --sim --controller nonlinear_pid --slope 5
    python main.py --sim --controller nonlinear_pid --slope 5 --ml-adaptive
    python compare_ml_adaptive.py --scenario sudden --horizon 6 --step 0.01

The default run is the numerical 3-D model. PyBullet visualization is opt-in
with ``--visual`` or ``--all`` and currently uses its separate legacy control
implementation (it is not ML-controlled).
"""

from run_all import main as _run


def main():
    _run()


if __name__ == "__main__":
    main()
