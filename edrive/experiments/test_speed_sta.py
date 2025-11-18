# experiments/exp_test_sta.py
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import control as ct

# add src/ to path
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from edrive.plants.motor.pmdc_state_space import pmdc_ss
from edrive.controllers.speed_sta import SuperTwistingController

def main():
    # --- Motor params (Energies' Table 2 demo-like) ---
    r_a = 8.32
    l_a = 0.0813
    k_e = 0.549
    k_t = 0.549
    j   = 0.0099
    b   = 8.3e-4
    Vdc = 120.0

    # --- Build plant with load as 2nd input: [v_a, T_L] ---
    sys_c = pmdc_ss(r_a, l_a, k_e, k_t, j, b, include_load=True)

    # --- Discretize for simple loop simulation ---
    dt = 1e-3  # 1 ms
    sys_d = ct.c2d(sys_c, dt, method='zoh')
    Ad, Bd, Cd, Dd = sys_d.A, sys_d.B, sys_d.C, sys_d.D

    # --- Super-Twisting controller (quick demo gains) ---
    sta = SuperTwistingController(
        lam=100.0,       # tune up/down if needed
        alpha=1000.0,   # tune up/down if needed
        dt=dt,
        u_min=0.0,
        u_max=Vdc,
        sgn_eps=0     # small deadband to soften chattering
    )

    # --- Time & references ---
    T_end = 2.0
    T = np.arange(0.0, T_end, dt)
    N = len(T)

    # speed ref: 0 -> 60 rad/s at 0.2 s;  -> 100 rad/s at 1.2 s
    omega_ref = np.zeros(N)
    omega_ref[T >= 0.2] = 60.0
    omega_ref[T >= 1.2] = 100.0

    # load torque profile: 0 -> 0.5 N·m at 1.0 s
    T_L = np.zeros(N)
    T_L[T >= 1.0] = 0.5

    # --- Simulation state/logs ---
    x = np.zeros(2)  # [i, omega]
    i_log = np.zeros(N)
    w_log = np.zeros(N)
    u_log = np.zeros(N)
    xs_log = np.zeros(N)

    # --- Closed-loop loop ---
    for k in range(N-1):
        i_k, w_k = x[0], x[1]
        # sliding variable (simple test): x_s = omega_ref - omega
        x_s = omega_ref[k] - w_k
        u = sta.step(x_s)  # armature voltage (saturated in STA)

        # 2-input vector: [v_a, T_L]
        u_vec = np.array([u, T_L[k]])
        # x_{k+1} = Ad x_k + Bd u_k
        x = Ad @ x + Bd @ u_vec

        # log
        i_log[k] = i_k
        w_log[k] = w_k
        u_log[k] = u
        xs_log[k] = x_s

    # --- Plots ---
    fig1 = plt.figure(figsize=(9, 7))
    ax1 = fig1.add_subplot(3, 1, 1)
    ax1.plot(T, omega_ref, label='ω_ref [rad/s]')
    ax1.plot(T, w_log, label='ω [rad/s]')
    ax1.set_ylabel('Speed')
    ax1.legend(); ax1.grid(True)

    # ax2 = fig1.add_subplot(3, 1, 2)
    # ax2.plot(T, i_log, label='i [A]')
    # ax2.set_ylabel('Current')
    # ax2.legend(); ax2.grid(True)

    # ax3 = fig1.add_subplot(3, 1, 3)
    # ax3.plot(T, u_log, label='v_a [V]')
    # ax3.plot(T, T_L * 50, '--', label='T_L (×50)')  # scaled for view
    # ax3.set_ylabel('Voltage'); ax3.set_xlabel('Time [s]')
    # ax3.legend(); ax3.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

