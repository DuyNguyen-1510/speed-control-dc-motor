"""
Cascade 2‑SMC controller for PMDC speed control (continuous‑time law, paper notation).

Outer loop (speed):  x = ω,  v = i_ref,   ẋ = −A_ω x + B_ω v + d_ω,
    i_ref = (A_ω ω_ref + ω̇_ref + k1_ω |e_ω|^{1/2} sat(e_ω/α_ω) − w_ω) / B_ω,
    ẇ_ω = −k2_ω sat(e_ω/α_ω),   e_ω = ω_ref − ω.

Inner loop (current): x = i,  v = V,      ẋ = −A_i x + B_i v + d_i,
    V = (A_i i_ref + i̇_ref + k1_i |e_i|^{1/2} sat(e_i/α_i) − w_i) / B_i,
    ẇ_i = −k2_i sat(e_i/α_i),   e_i = i_ref − i.

This module composes two `TwoSMC` channels (see controllers/smc2.py) and adds:
- Reference‑rate generation: ω̇_ref via backward difference, i̇_ref via 1st‑order filter (τ_ir).
- Fixed‑frequency PWM mapping: duty = sat(V/Vdc), with |V| ≤ Vdc and |i_ref| ≤ I_max.
- Two sampling periods: Ts_i (inner) and Ts_ω (outer). Outer updates every N = round(Ts_ω/Ts_i) ticks.

Keep signs & symbols exactly as the paper: e = ref − actual, S ≡ e, sat(e/α) as smooth sign.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import os, sys

# Add project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from edrive.controllers import TwoSMC, TwoSMCParams, pmdc_speed_AB, pmdc_current_AB

# -----------------------------
# Utilities
# -----------------------------

def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# -----------------------------
# Motor & controller configuration
# -----------------------------
@dataclass
class MotorParams:
    R: float
    L: float
    Kt: float
    Kb: float
    J: float
    B: float
    Vdc: float
    I_max: float


@dataclass
class CascadeConfig:
    # Sampling
    Ts_i: float                 # inner loop period (seconds)
    Ts_w: float                 # outer loop period (seconds)
    # 2‑SMC gains (outer: speed)
    k1_w: float
    k2_w: float
    alpha_w: float
    # 2‑SMC gains (inner: current)
    k1_i: float
    k2_i: float
    alpha_i: float
    # Command smoothing for i_ref (ref shaper): τ_ir (seconds)
    tau_ir: float = 5e-4

    def validate(self) -> None:
        assert self.Ts_i > 0 and self.Ts_w > 0
        assert self.k1_w > 0 and self.k2_w > 0 and self.k1_i > 0 and self.k2_i > 0
        assert self.alpha_w >= 0 and self.alpha_i >= 0
        # outer must be slower or equal to inner
        assert self.Ts_w >= self.Ts_i


# -----------------------------
# Cascade controller
# -----------------------------
class Cascade2SMC:
    """Cascade 2‑SMC (outer speed → i_ref, inner current → V) with fixed‑frequency PWM mapping.

    Usage:
        cc = Cascade2SMC(motor, cfg)
        V_cmd, duty, info = cc.step(i_meas, omega_meas, omega_ref)
    Call `reset()` to clear integrators.
    """
    def __init__(self, motor: MotorParams, cfg: CascadeConfig):
        cfg.validate()
        self.m = motor
        self.cfg = cfg

        # Map physical params to (A,B) for each loop
        A_w, B_w = pmdc_speed_AB(motor.J, motor.B, motor.Kt)
        A_i, B_i = pmdc_current_AB(motor.R, motor.L)

        # Build 2‑SMC channels with their own sampling
        self.smc_w = TwoSMC(TwoSMCParams(A=A_w, B=B_w, k1=cfg.k1_w, k2=cfg.k2_w,
                                         alpha=cfg.alpha_w, Ts=cfg.Ts_w))
        self.smc_i = TwoSMC(TwoSMCParams(A=A_i, B=B_i, k1=cfg.k1_i, k2=cfg.k2_i,
                                         alpha=cfg.alpha_i, Ts=cfg.Ts_i))

        # Scheduling
        self._N = max(1, int(round(cfg.Ts_w / cfg.Ts_i)))  # outer updates every N inner ticks
        self._ctr = 0

        # Internal refs & rates
        self.i_ref_cmd = 0.0       # raw from outer SMC
        self.i_ref_f = 0.0         # filtered/smoothed current ref (goes to inner loop)
        self.i_ref_dot = 0.0
        self._omega_ref_prev = 0.0

    def reset(self) -> None:
        self.smc_w.reset(); self.smc_i.reset()
        self._ctr = 0
        self.i_ref_cmd = 0.0
        self.i_ref_f = 0.0
        self.i_ref_dot = 0.0
        self._omega_ref_prev = 0.0

    def _outer_update(self, omega: float, omega_ref: float) -> Dict[str, float]:
        # Backward‑difference for ω̇_ref (paper expects ẋ_ref term)
        wdot_ref = (omega_ref - self._omega_ref_prev) / self.cfg.Ts_w
        self._omega_ref_prev = omega_ref
        # 2‑SMC (speed): returns i_ref_cmd
        i_ref_cmd, dbg_w = self.smc_w.step(x=omega, x_ref=omega_ref, x_ref_dot=wdot_ref)
        # Saturate to current capability
        i_ref_cmd = clip(i_ref_cmd, -self.m.I_max, self.m.I_max)
        self.i_ref_cmd = i_ref_cmd
        return {
            'omega_ref_dot': wdot_ref,
            'i_ref_cmd': i_ref_cmd,
            **{f'w_{k}': v for k, v in dbg_w.items()}
        }

    def _shape_i_ref(self) -> None:
        # First‑order ref shaper: i̇_ref = (i_ref_cmd − i_ref_f)/τ_ir;  i_ref_f += Ts_i * i̇_ref
        tau = max(self.cfg.tau_ir, 1e-9)
        self.i_ref_dot = (self.i_ref_cmd - self.i_ref_f) / tau
        self.i_ref_f += self.cfg.Ts_i * self.i_ref_dot
        # Enforce current limits
        self.i_ref_f = clip(self.i_ref_f, -self.m.I_max, self.m.I_max)

    def _inner_update(self, i_meas: float, Vdc_eff: float) -> Tuple[float, Dict[str, float]]:
        # 2-SMC (current): returns V command (pre-saturation)
        V_cmd, dbg_i = self.smc_i.step(x=i_meas, x_ref=self.i_ref_f, x_ref_dot=self.i_ref_dot)
        # PWM/saturation (support runtime bus-voltage overrides)
        bus_mag = max(abs(Vdc_eff), 1e-9)
        V_sat = clip(V_cmd, -bus_mag, bus_mag)
        duty = V_sat / Vdc_eff if abs(Vdc_eff) > 1e-9 else 0.0
        return V_sat, {**dbg_i, 'V_raw': V_cmd, 'V_sat': V_sat, 'duty': duty, 'Vdc_eff': Vdc_eff}


    def step(self, i_meas: float, omega_meas: float, omega_ref: float,
             Vdc_override: float | None = None) -> Tuple[float, float, Dict[str, float]]:
        """Advance one inner-loop tick (Ts_i). Outer loop runs every N ticks.
        Returns (V_cmd, duty, info).
        """
        info: Dict[str, float] = {}
        if self._ctr == 0:
            info.update(self._outer_update(omega=omega_meas, omega_ref=omega_ref))
        self._ctr = (self._ctr + 1) % self._N

        # Shape i_ref every inner tick and run inner loop
        self._shape_i_ref()
        Vdc_eff = self.m.Vdc if Vdc_override is None else float(Vdc_override)
        V_cmd, dbg_i = self._inner_update(i_meas, Vdc_eff)
        info.update({
            'i_ref_f': self.i_ref_f,
            'i_ref_dot': self.i_ref_dot,
            **{f'i_{k}': v for k, v in dbg_i.items()}
        })
        return V_cmd, dbg_i['duty'], info
