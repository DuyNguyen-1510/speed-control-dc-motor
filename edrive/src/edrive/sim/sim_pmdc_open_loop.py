# src/runtime/sim_open_loop.py
from __future__ import annotations
import os, time, argparse, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import control as ct

# Add project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from edrive.plants.motor.pmdc_state_space import pmdc_ss, step_u 

if __name__ == "__main__":
    # Example parameters (SI):
    r_a, l_a  = 0.6, 0.35e-3        # Ohm, H
    k_e, k_t = 0.0191, 0.0187        # V·s/rad, N·m/A
    j_rotor, b  = 1.25e-4, 0.0000095     # kg·m^2, N·m·s/rad

    sys = pmdc_ss(r_a, l_a, k_e, k_t, j_rotor, b)

    # Time & input: 24 V step
    T = np.linspace(0, 2.5, 5000)

    # --- input profiles ---
    # U = 24.0 * np.ones_like(T)
    # --- Case 1: STEP 0→24 V at t=0 ---
    U_step = step_u(T, V0=0.0, V1=12.0, t_step=1)

    # --- Case 2: SAG: 24 V, tụt 24→18 V từ 0.15s→0.25s ---
    # U_sag = sag_u(T, V_nom=24.0, V_sag=18.0, t_start=0.15, t_end=0.25)

    X0 = [0, 0]
    response = ct.forced_response(sys, T, U_step, X0)
    # cplt = response.plot()
    # plt.show()

    # Plot the outputs of the system on the same graph, in different colors
    t = response.time
    x = response.states

    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t, x[1], 'r-', label="omega")
    ax[0].set_ylabel("omega, rad/s")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, x[0], 'b-', label="current")
    ax[1].set_ylabel("Сurrent, A")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(t, U_step, 'm-', label="input voltage")
    ax[2].set_ylabel("Input Voltage, V")
    ax[2].set_xlabel("Time, s")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()