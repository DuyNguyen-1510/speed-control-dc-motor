"""
ST Controller Closed-Loop Simulation for Boost Converter
Tests output voltage regulation under input voltage drop scenario.
  
Example:
--V_ref 48 --Vin_dropped 10 --t_drop_start 0.02
    
This script:
1. Creates a boost converter plant with input voltage drop at specified time
2. Uses boost observer to estimate states and disturbances  
3. Implements ST controller for output voltage regulation
4. Plots results showing control performance during input disturbance
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from edrive.plants.boost.averaged import BoostParams, BoostParasitics, BoostAveragedPlant
from edrive.observers.boost_obs import BoostObserverParams, BoostObserver
from edrive.controllers.bus_st import STCParams, STController

def input_voltage_drop_profile(t_drop_start: float, t_drop_end: float, 
                              v_nominal: float, v_dropped: float) -> Callable[[float], float]:
    """Create input voltage drop profile for testing."""
    def vin_profile(t: float) -> float:
        if t_drop_start <= t <= t_drop_end:
            return v_dropped
        return v_nominal
    return vin_profile

def steady_state_ic(Vin: float, R: float, D: float, V_ref: float) -> tuple[float, float]:
    """Calculate steady-state initial conditions for boost converter."""
    # Ideal boost: Vo = Vin / (1-D), Io = Vo/R, IL = Io / (1-D)
    vo_ss = V_ref  # target output voltage
    iL_ss = vo_ss / (R * (1.0 - D)) if (1.0 - D) > 1e-6 else 1.0
    return iL_ss, vo_ss

def main():
    parser = argparse.ArgumentParser(description="ST Controller Closed-Loop Boost Simulation")
    
    # Plant parameters
    parser.add_argument("--L", type=float, default=200e-6, help="Inductance [H]")
    parser.add_argument("--C", type=float, default=330e-6, help="Capacitance [F]")
    parser.add_argument("--R", type=float, default=100, help="Load resistance [Ohm]")
    parser.add_argument("--RL", type=float, default=0.35, help="Inductor ESR [Ohm]")
    
    # Input voltage & references
    parser.add_argument("--Vin_nom", type=float, default=24.0, help="Nominal input voltage [V]")
    parser.add_argument("--Vin_dropped", type=float, default=10.0, help="Dropped input voltage [V]")
    parser.add_argument("--V_ref", type=float, default=48.0, help="Target output voltage [V]")
    
    # Test scenario timing
    parser.add_argument("--t_drop_start", type=float, default=0.02, help="Start time for voltage drop [s]")
    parser.add_argument("--t_drop_end", type=float, default=0.04, help="End time for voltage drop [s]")
    
    # Simulation parameters
    parser.add_argument("--Ts", type=float, default=1e-4, help="Sampling time [s]")
    parser.add_argument("--duration", type=float, default=0.06, help="Simulation duration [s]")
    
    # ST Controller parameters
    parser.add_argument("--c1", type=float, default=1, help="Sliding surface weight c1")
    parser.add_argument("--c2", type=float, default=3.2, help="Sliding surface weight c2")
    parser.add_argument("--alpha", type=float, default=1000.0, help="ST gain alpha")
    parser.add_argument("--beta", type=float, default=10000.0, help="ST gain beta")
    
    # Observer parameters
    parser.add_argument("--k1", type=float, default=-3.2768e+3, help="Observer gain k1")
    parser.add_argument("--k2", type=float, default=9.9394e+3, help="Observer gain k2")
    parser.add_argument("--k3", type=float, default=2.1249e+3, help="Observer gain k3")
    
    # Initialization
    parser.add_argument("--init_duty", type=float, default=0.5, help="Initial duty for steady-state calculation")
    parser.add_argument("--use_steady_state_ic", action="store_true", help="Use steady-state initial conditions")
    
    args = parser.parse_args()
    
    # Create input voltage profile with drop
    vin_profile = input_voltage_drop_profile(
        args.t_drop_start, args.t_drop_end, 
        args.Vin_nom, args.Vin_dropped
    )
    
    # Setup plant
    parasitics = BoostParasitics(rL=args.RL, rDS=0.0, VF=0.0, RF=0.0)
    plant_cfg = BoostParams(
        L=args.L, C=args.C, Vin=vin_profile, R=args.R,
        d_min=0.02, d_max=0.98, non_ideal=True, par=parasitics
    )
    plant = BoostAveragedPlant(plant_cfg)
    
    # Setup observer
    obs_params = BoostObserverParams(
        L=args.L, C=args.C, R_L=args.RL, V_ref=args.V_ref,
        k1=args.k1, k2=args.k2, k3=args.k3
    )
    observer = BoostObserver(obs_params)
    
    # Setup ST controller
    st_params = STCParams(
        L=args.L, C=args.C, R_L=args.RL, V_ref=args.V_ref,
        c1=args.c1, c2=args.c2, k1=args.k1, k2=args.k2,
        alpha=args.alpha, beta=args.beta, d_min=0.02, d_max=0.98
    )
    controller = STController(st_params)
    
    # Initial conditions
    if args.use_steady_state_ic:
        iL_0, vo_0 = steady_state_ic(args.Vin_nom, args.R, args.init_duty, args.V_ref)
        plant.reset((iL_0, vo_0))
        observer.reset(z1_hat=iL_0, z2_hat=vo_0 - args.V_ref, p=0.0)
    else:
        plant.reset((0.0, 0.0))
        observer.reset(0.0, 0.0, 0.0)
    
    controller.reset()
    
    # Simulation loop
    Ts = args.Ts
    duration = args.duration
    n_steps = int(duration / Ts) + 1
    
    # Storage arrays
    t_array = np.zeros(n_steps)
    vin_array = np.zeros(n_steps)
    vo_array = np.zeros(n_steps)
    iL_array = np.zeros(n_steps)
    duty_array = np.zeros(n_steps)
    vo_ref_array = np.zeros(n_steps)
    s_array = np.zeros(n_steps)
    w_hat_array = np.zeros(n_steps)
    
    t = 0.0
    
    print("Starting ST Controller simulation...")
    print(f"Reference voltage: {args.V_ref} V")
    print(f"Input voltage drop: {args.Vin_nom}V -> {args.Vin_dropped}V at {args.t_drop_start}-{args.t_drop_end}s")
    
    # Initialize with previous duty for first iteration
    prev_duty = 0.4  # Initial duty guess
    
    for k in range(n_steps):
        # Get current measurements (from previous step)
        iL_meas = plant.x[0]
        vo_meas = plant.x[1]
        vin_meas = vin_profile(t)
        
        # Observer step (uses previous duty cycle)
        obs_outputs = observer.step(Ts, vin_meas, vo_meas, prev_duty)
        iL_hat = obs_outputs["iL_hat"]
        vo_hat = obs_outputs["vo_hat"]
        w_hat = obs_outputs["w_hat"]
        e2 = obs_outputs["e2"]
        
        # Controller step
        ctrl_inputs = {
            "iL_hat": iL_hat,
            "vo_hat": vo_hat, 
            "w_hat": w_hat,
            "e2": e2
        }
        ctrl_outputs = controller.step(Ts, vin_meas, ctrl_inputs)
        duty = ctrl_outputs["d"]
        
        # Update plant with computed duty
        plant_states, _ = plant.propagate(t, t + Ts, duty)
        iL_meas = plant_states[0]
        vo_meas = plant_states[1]
        
        # Store previous duty for next iteration
        prev_duty = duty
        
        # Store results
        t_array[k] = t
        vin_array[k] = vin_meas
        vo_array[k] = vo_meas
        iL_array[k] = iL_meas
        duty_array[k] = duty
        vo_ref_array[k] = args.V_ref
        s_array[k] = ctrl_outputs["s"]
        w_hat_array[k] = w_hat
        
        t += Ts
        
        if k % 1000 == 0:
            print(f"t={t:.3f}s, Vo={vo_meas:.2f}V, Vin={vin_meas:.2f}V, d={duty:.3f}")
    
    # Analysis and plotting
    print("\nSimulation Results:")
    
    # Find steady-state before drop
    idx_before = np.where((t_array >= 0.05) & (t_array < args.t_drop_start))[0]
    vo_before = np.mean(vo_array[idx_before]) if len(idx_before) > 0 else vo_array[0]
    
    # Find steady-state during drop
    idx_during = np.where((t_array >= args.t_drop_start + 0.05) & (t_array < args.t_drop_end))[0]
    vo_during = np.mean(vo_array[idx_during]) if len(idx_during) > 0 else vo_array[-1]
    
    # Find steady-state after drop
    idx_after = np.where(t_array >= args.t_drop_end + 0.05)[0]
    vo_after = np.mean(vo_array[idx_after]) if len(idx_after) > 0 else vo_array[-1]
    
    print(f"Output voltage before drop: {vo_before:.2f} V (error: {vo_before - args.V_ref:.2f} V)")
    print(f"Output voltage during drop: {vo_during:.2f} V (error: {vo_during - args.V_ref:.2f} V)")
    print(f"Output voltage after drop: {vo_after:.2f} V (error: {vo_after - args.V_ref:.2f} V)")
    
    # Plotting
    mpl.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 10,                 # cỡ chữ cố định (pt)
    })

    fig, axes = plt.subplots(2, 1, figsize=(5.01, 4.2), dpi=300, sharex=True)
    
    # Input and output voltages
    axes[0].plot(t_array, vin_array, 'b-', label='Vi - Входное напряжение', linewidth=0.7)
    axes[0].plot(t_array, vo_array, 'r-', label='Vo - Выходное напряжение', linewidth=0.7)
    axes[0].plot(t_array, vo_ref_array, 'k--', label='Vo_ref - Заданное напряжение', linewidth=0.7)
    axes[0].axvspan(args.t_drop_start, args.t_drop_end, alpha=0.2, color='gray', label='Период проседания напряжения')
    axes[0].set_ylabel('Напряжения [В]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # axes[0].set_title('ST Регулятор: Проверка падения входного напряжения')
    
    # Inductor current
    # axes[1].plot(t_array, iL_array, 'g-', linewidth=1.5)
    # axes[1].axvspan(args.t_drop_start, args.t_drop_end, alpha=0.2, color='gray')
    # axes[1].set_ylabel('Ток индуктивности, iL [A]')
    # axes[1].grid(True, alpha=0.3)
    
    # Duty cycle
    axes[1].plot(t_array, duty_array, 'm-', label="d - Скважность", linewidth=0.7)
    axes[1].axvspan(args.t_drop_start, args.t_drop_end, alpha=0.2, color='gray')
    axes[1].set_ylabel('Скважность, d')
    axes[1].set_xlabel('Время, t [с]')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    # Sliding surface and disturbance estimate
    # axes[3].plot(t_array, s_array, 'c-', label='Скользящая поверхность, s', linewidth=1.5)
    # axes[3].plot(t_array, w_hat_array, 'orange', label='Возмущение, w_hat', linewidth=1.5)
    # axes[3].axvspan(args.t_drop_start, args.t_drop_end, alpha=0.2, color='gray')
    # axes[3].set_ylabel('Оценка')
    # axes[3].set_xlabel('Время, t [s]')
    # axes[3].legend()
    # axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'time': t_array,
        'vin': vin_array,
        'vo': vo_array,
        'iL': iL_array,
        'duty': duty_array,
        'vo_ref': vo_ref_array,
        'sliding_surface': s_array,
        'disturbance_hat': w_hat_array,
        'metrics': {
            'vo_before': vo_before,
            'vo_during': vo_during, 
            'vo_after': vo_after,
            'error_before': vo_before - args.V_ref,
            'error_during': vo_during - args.V_ref,
            'error_after': vo_after - args.V_ref
        }
    }

if __name__ == "__main__":
    results = main()
