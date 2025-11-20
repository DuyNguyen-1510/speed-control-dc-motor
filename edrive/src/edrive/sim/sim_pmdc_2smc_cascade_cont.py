"""
Closed-loop simulation: PMDC speed control with cascade **2‑SMC (continuous law)**
- Plant: your state-space module `pmdc_state_space.py`
- Controllers: `controllers/smc2.py` (TwoSMC) + `controllers/cascade.py` (Cascade2SMC)

Architecture (paper notation, ported to PMDC):
  Outer (speed): 2‑SMC outputs i_ref
  Inner (current): 2‑SMC outputs V, mapped to fixed‑frequency PWM via duty=V/Vdc

Run:  python -m sims.sim_pmdc_2smc_cont
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Add project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import your plant and controllers
from edrive.plants.motor.pmdc_state_space import pmdc_ss
from edrive.controllers.cascade import MotorParams, CascadeConfig, Cascade2SMC


def run(total_time: float = 8,
        f_pwm: float = 1e3,
        w_step_time: float = 2,
        w_target: float = 120.0,
        TL_step_time: float = 4.0,
        TL_val: float = 0.2,
        seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)

    # ---------- Motor physical parameters (consistent across plant & controller)
    motor = MotorParams(
        R=0.25, L=0.00224, Kt=0.035, Kb=0.035,
        J=0.0023, B=0.0005, Vdc=45.0, I_max=22.5,
    )

    # ---------- Sampling
    Ts_i = 1.0 / f_pwm             # inner (current)
    Ts_w = 10 * Ts_i               # outer (speed) ≈ 10× slower

    # ---------- 2‑SMC gains (safe starting point)
    cfg = CascadeConfig(
        Ts_i=Ts_i, Ts_w=Ts_w,
        
        k1_w=40,  k2_w=50, alpha_w=1,

        k1_i=10, k2_i=3000, alpha_i=0.5,
        
        tau_ir=5*Ts_i,
    )

    # ---------- Controllers (outer→inner)
    cc = Cascade2SMC(motor, cfg)

    # ---------- Plant from your state-space module (continuous A,B)
    sys = pmdc_ss(motor.R, motor.L, motor.Kb, motor.Kt, motor.J, motor.B)
    A = np.array(sys.A)
    B = np.array(sys.B)   # columns: [V,  -T_L/J]

    # ---------- Time base & logs
    N = int(total_time / Ts_i)
    t = np.arange(N) * Ts_i
    x = np.zeros((2, 1))   # [i, ω]^T

    omega_ref = np.zeros(N)
    TL = np.zeros(N)
    i_log = np.zeros(N)
    w_log = np.zeros(N)
    iref_log = np.zeros(N)
    Vcmd_log = np.zeros(N)
    duty_log = np.zeros(N)

    # ---------- Simulation loop (Euler integration for plant)
    for k in range(N):
        tk = t[k]
        # references
        omega_ref[k] = w_target if tk >= w_step_time else 0.0
        TL[k] = TL_val if tk >= TL_step_time else 0.0

        # controller step (outer runs every Ts_w internally)
        V_cmd, duty, info = cc.step(i_meas=x[0, 0], omega_meas=x[1, 0], omega_ref=omega_ref[k])

        # plant update
        V_cmd = np.clip(V_cmd, -motor.Vdc, motor.Vdc)
        u = np.array([[V_cmd], [TL[k]]])
        xdot = A @ x + B @ u
        x += Ts_i * xdot

        # logs
        i_log[k] = x[0, 0]
        w_log[k] = x[1, 0]
        iref_log[k] = info['i_ref_f']
        Vcmd_log[k] = V_cmd
        duty_log[k] = V_cmd / motor.Vdc

    # ---------- Plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(t, w_log, label='ω_текущая')
    axs[0].plot(t, omega_ref, '--', label='ω_задание')
    axs[0].set_ylabel('Угловая скорость, рад/с'); axs[0].legend(); axs[0].grid(True)

    axs[1].plot(t, i_log, label='i_текущий')
    axs[1].plot(t, iref_log, '--', label='i_задание (после фильтрации)')
    axs[1].set_ylabel('Ток, А'); axs[1].legend(); axs[1].grid(True)

    axs[2].plot(t, Vcmd_log, label='u_управляющий сигнал')
    axs[2].set_ylabel('Напряжение, В'); axs[2].legend(); axs[2].grid(True)

    axs[3].plot(t, TL, label='T_L')
    axs[3].set_ylabel('Момент нагрузки, Н·м'); axs[3].set_xlabel('Время, с'); axs[3].legend(); axs[3].grid(True)

    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    run()
