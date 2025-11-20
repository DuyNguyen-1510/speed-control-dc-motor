"""
Averaged co-simulation: boost converter with ST controller feeding the PMDC cascade 2-SMC loop.

This script ties together the existing averaged boost model/observer/ST controller and the
PMDC cascade controller. The boost output voltage (V_bus) is fed directly into the cascade
as its DC bus limit, so any bus sag immediately saturates the inner SMC loop. The motor load
is reflected to the boost stage through a time-varying equivalent resistance computed from
V_bus and the motor current estimate (duty * i_a). This matches the step-1 plan of using an
averaged model without an explicit H-bridge model or DC-link capacitor dynamics.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root for absolute imports when launched as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from edrive.plants.boost.averaged import BoostParams, BoostParasitics, BoostAveragedPlant
from edrive.observers.boost_obs import BoostObserverParams, BoostObserver
from edrive.controllers.bus_st import STCParams, STController
from edrive.controllers.cascade import MotorParams, CascadeConfig, Cascade2SMC
from edrive.plants.motor.pmdc_state_space import pmdc_ss


def input_voltage_drop_profile(t0: float, t1: float, v_nom: float, v_drop: float) -> Callable[[float], float]:
    """Return Vin(t) profile with a rectangular drop for disturbance tests."""
    def profile(t: float) -> float:
        if t0 <= t <= t1:
            return v_drop
        return v_nom

    return profile


def make_dynamic_load(initial_R: float) -> Tuple[Dict[str, float], Callable[[float], float]]:
    """Create a callable R(t) = state['R'] so we can update load equivalence each step."""
    state = {'R': max(initial_R, 1e-3)}

    def load_fun(_: float) -> float:
        return state['R']

    return state, load_fun


def effective_load_from_bus(V_bus: float, I_bus: float, R_min: float, R_max: float, eps: float, R_open: float) -> float:
    """Convert (V_bus, I_bus) into an equivalent resistance seen by the boost output."""
    if abs(I_bus) <= eps or V_bus <= 0.0:
        return R_open
    R_eff = abs(V_bus) / max(abs(I_bus), eps)
    return float(np.clip(R_eff, R_min, R_max))


def run(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="Boost + PMDC cascade averaged co-simulation")

    # Timing / duration
    parser.add_argument('--duration', type=float, default=5, help='Simulation time [s]')
    parser.add_argument('--Ts', type=float, default=1e-4, help='Simulation step (shared by boost & motor) [s]')
    parser.add_argument('--tau_ir', type=float, default=5e-4, help='Current reference shaper time constant [s]')

    # Motion profile / load torque
    parser.add_argument('--omega_ref', type=float, default=120.0, help='Target speed [rad/s]')
    parser.add_argument('--w_step_time', type=float, default=1, help='Time to apply speed command [s]')
    parser.add_argument('--TL_step_time', type=float, default=3, help='Time to apply load torque [s]')
    parser.add_argument('--TL_val', type=float, default=0.2, help='Load torque magnitude [N*m]')

    # Boost input disturbance
    parser.add_argument('--Vin_nom', type=float, default=24.0, help='Nominal input voltage [V]')
    parser.add_argument('--Vin_dropped', type=float, default=10.0, help='Dropped input voltage [V]')
    parser.add_argument('--t_drop_start', type=float, default=0.02, help='Start time for Vin drop [s]')
    parser.add_argument('--t_drop_end', type=float, default=0.04, help='End time for Vin drop [s]')
    parser.add_argument('--Vref_bus', type=float, default=48.0, help='Boost output voltage target [V]')

    # Boost plant / controller params
    parser.add_argument('--L', type=float, default=200e-6, help='Boost inductance [H]')
    parser.add_argument('--C', type=float, default=330e-6, help='Boost capacitance [F]')
    parser.add_argument('--RL', type=float, default=0.35, help='Inductor ESR [Ohm]')
    parser.add_argument('--c1', type=float, default=1.0, help='ST surface weight c1')
    parser.add_argument('--c2', type=float, default=3.2, help='ST surface weight c2')
    parser.add_argument('--alpha', type=float, default=1000.0, help='ST gain alpha')
    parser.add_argument('--beta', type=float, default=10000.0, help='ST gain beta')

    parser.add_argument('--k1', type=float, default=-3.2768e3, help='Observer gain k1')
    parser.add_argument('--k2', type=float, default=9.9394e3, help='Observer gain k2')
    parser.add_argument('--k3', type=float, default=2.1249e3, help='Observer gain k3')

    parser.add_argument('--d_min', type=float, default=0.05, help='Duty min for boost stage')
    parser.add_argument('--d_max', type=float, default=0.95, help='Duty max for boost stage')
    parser.add_argument('--init_duty', type=float, default=0.4, help='Initial duty guess for boost stage')

    # Motor + cascade parameters
    parser.add_argument('--R_a', type=float, default=0.25, help='Motor armature resistance [Ohm]')
    parser.add_argument('--L_a', type=float, default=0.00224, help='Motor inductance [H]')
    parser.add_argument('--Kt', type=float, default=0.035, help='Torque constant [N*m/A]')
    parser.add_argument('--Kb', type=float, default=0.035, help='Back-emf constant [V*s/rad]')
    parser.add_argument('--J', type=float, default=0.0023, help='Rotor inertia [kg*m^2]')
    parser.add_argument('--B', type=float, default=0.0005, help='Viscous friction [N*m*s/rad]')
    parser.add_argument('--I_max', type=float, default=22.5, help='Peak phase current for saturation [A]')
    parser.add_argument('--outer_div', type=float, default=10.0, help='Speed loop period multiplier wrt inner loop')
    parser.add_argument('--k1_w', type=float, default=40.0, help='Outer SMC k1')
    parser.add_argument('--k2_w', type=float, default=50.0, help='Outer SMC k2')
    parser.add_argument('--alpha_w', type=float, default=1.0, help='Outer boundary layer width')
    parser.add_argument('--k1_i', type=float, default=10.0, help='Inner SMC k1')
    parser.add_argument('--k2_i', type=float, default=3000.0, help='Inner SMC k2')
    parser.add_argument('--alpha_i', type=float, default=0.5, help='Inner boundary layer width')






    # Load reflection tuning
    parser.add_argument('--R_load_init', type=float, default=100.0, help='Initial equivalent load [Ohm]')
    parser.add_argument('--R_min', type=float, default=0.5, help='Minimum equivalent load [Ohm]')
    parser.add_argument('--R_max', type=float, default=5e3, help='Maximum equivalent load [Ohm]')
    parser.add_argument('--R_open', type=float, default=1e6, help='Resistance used when motor feeds back or idle [Ohm]')
    parser.add_argument('--I_load_eps', type=float, default=1e-3, help='Current threshold before declaring open-circuit [A]')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    Ts = args.Ts
    duration = args.duration
    n_steps = int(duration / Ts)

    vin_profile = input_voltage_drop_profile(args.t_drop_start, args.t_drop_end, args.Vin_nom, args.Vin_dropped)
    load_state, load_fun = make_dynamic_load(args.R_load_init)

    # Boost components
    plant_cfg = BoostParams(
        L=args.L,
        C=args.C,
        Vin=vin_profile,
        R=load_fun,
        d_min=args.d_min,
        d_max=args.d_max,
        non_ideal=True,
        par=BoostParasitics(rL=args.RL)
    )
    plant = BoostAveragedPlant(plant_cfg)
    observer = BoostObserver(BoostObserverParams(L=args.L, C=args.C, R_L=args.RL, V_ref=args.Vref_bus,
                                                 k1=args.k1, k2=args.k2, k3=args.k3,
                                                 d_min=args.d_min, d_max=args.d_max))
    st_params = STCParams(L=args.L, C=args.C, R_L=args.RL, V_ref=args.Vref_bus,
                          c1=args.c1, c2=args.c2, k1=args.k1, k2=args.k2,
                          alpha=args.alpha, beta=args.beta, d_min=args.d_min, d_max=args.d_max)
    controller = STController(st_params)

    # Motor + cascade
    motor = MotorParams(R=args.R_a, L=args.L_a, Kt=args.Kt, Kb=args.Kb,
                        J=args.J, B=args.B, Vdc=args.Vref_bus, I_max=args.I_max)
    Ts_i = Ts
    Ts_w = args.outer_div * Ts_i
    cascade_cfg = CascadeConfig(Ts_i=Ts_i, Ts_w=Ts_w,
                                k1_w=args.k1_w, k2_w=args.k2_w, alpha_w=args.alpha_w,
                                k1_i=args.k1_i, k2_i=args.k2_i, alpha_i=args.alpha_i,
                                tau_ir=args.tau_ir)
    cascade = Cascade2SMC(motor, cascade_cfg)

    # PMDC plant matrices (continuous) + Euler integration for this averaged test
    sys_c = pmdc_ss(motor.R, motor.L, motor.Kb, motor.Kt, motor.J, motor.B)
    A = np.array(sys_c.A, dtype=float)
    B = np.array(sys_c.B, dtype=float)
    x_motor = np.zeros((2, 1), dtype=float)

    # Logging arrays
    t_log = np.zeros(n_steps)
    Vbus_log = np.zeros(n_steps)
    vin_log = np.zeros(n_steps)
    iL_log = np.zeros(n_steps)
    duty_boost_log = np.zeros(n_steps)
    duty_motor_log = np.zeros(n_steps)
    omega_log = np.zeros(n_steps)
    omega_ref_log = np.zeros(n_steps)
    i_arm_log = np.zeros(n_steps)
    Vcmd_log = np.zeros(n_steps)
    TL_log = np.zeros(n_steps)
    R_eff_log = np.zeros(n_steps)

    prev_duty = float(np.clip(args.init_duty, args.d_min, args.d_max))
    controller.reset()
    cascade.reset()
    observer.reset()
    plant.reset((0.0, 0.0))

    I_bus_motor = 0.0
    print("Running boost + PMDC cascade co-simulation...")

    for k in range(n_steps):
        t = k * Ts
        vin = vin_profile(t)
        vo = float(plant.x[1])
        obs_out = observer.step(Ts, vin, vo, prev_duty)
        ctrl_out = controller.step(Ts, vin, obs_out)
        duty_boost = ctrl_out['d']
        boost_state, _ = plant.propagate(t, t + Ts, duty_boost)
        iL = float(boost_state[0])
        V_bus = float(boost_state[1])

        omega_ref = args.omega_ref if t >= args.w_step_time else 0.0
        TL = args.TL_val if t >= args.TL_step_time else 0.0
        V_cmd, duty_motor, info = cascade.step(i_meas=float(x_motor[0, 0]),
                                               omega_meas=float(x_motor[1, 0]),
                                               omega_ref=omega_ref,
                                               Vdc_override=V_bus)
        u = np.array([[V_cmd], [TL]])
        xdot = A @ x_motor + B @ u
        x_motor += Ts * xdot
        i_arm = float(x_motor[0, 0])
        omega = float(x_motor[1, 0])

        # Equivalent load for the next boost step
        I_bus_motor = duty_motor * i_arm
        R_eff = effective_load_from_bus(V_bus, I_bus_motor, args.R_min, args.R_max,
                                        args.I_load_eps, args.R_open)
        
        # !!! cần tìm cách để controller boost converter ổn định trước thay đổi của
        # tải 
        load_state['R'] = R_eff
        print(R_eff)

        # khi tải cố định 100 Om thì controller boost converter hoạt động bình tốt
        # load_state['R'] = 100

        # Logs
        t_log[k] = t
        Vbus_log[k] = V_bus
        vin_log[k] = vin
        iL_log[k] = iL
        duty_boost_log[k] = duty_boost
        duty_motor_log[k] = duty_motor
        omega_log[k] = omega
        omega_ref_log[k] = omega_ref
        i_arm_log[k] = i_arm
        Vcmd_log[k] = V_cmd
        TL_log[k] = TL
        R_eff_log[k] = R_eff

        prev_duty = duty_boost
        if k % max(1, n_steps // 10) == 0:
            print(f"t={t:.4f}s Vbus={V_bus:.2f}V Vin={vin:.2f}V d_boost={duty_boost:.3f} ω={omega:.1f} rad/s")

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(t_log, Vbus_log, label='V_bus')
    axs[0].hlines(args.Vref_bus, t_log[0], t_log[-1], colors='k', linestyles='--', label='V_ref')
    axs[0].plot(t_log, vin_log, label='V_in')
    axs[0].set_ylabel('Voltage [V]')
    axs[0].legend(); axs[0].grid(True)

    axs[1].plot(t_log, omega_log, label='ω actual')
    axs[1].plot(t_log, omega_ref_log, '--', label='ω ref')
    axs[1].set_ylabel('Speed [rad/s]')
    axs[1].legend(); axs[1].grid(True)

    axs[2].plot(t_log, TL_log, label='T_L')
    axs[2].set_ylabel('Torque [N*m]')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend(); axs[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()

