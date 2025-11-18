"""
Two-SMC (second-order sliding mode) controller for 1st-order SISO plants —
kept **exactly** in the notation of the PMSM paper (ported to PMDC).

Model (continuous-time):  ẋ = -A x + B v + d,  error e = x_ref - x, sliding S = e.
2‑SMC law (outer/inner loops use the same form):
    u = -k1 |e|^{1/2} sat(e/α) + w,     ẇ = -k2 sat(e/α)
and with the transformation  u = -B v + A x_ref + ẋ_ref  ⇒
    v = (A x_ref + ẋ_ref + k1 |e|^{1/2} sat(e/α) - w) / B.
Here `sat(z)` = clip(z, -1, 1). Setting α→0 recovers sgn(·).

This module provides a small, dependency‑free class `TwoSMC` implementing the above.
It integrates w with forward Euler using the controller sampling time Ts.

Convenience helpers `pmdc_speed_AB` and `pmdc_current_AB` map PMDC parameters
(J,B,Kt) and (R,L) to (A,B) for the two loops.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math

# -----------------------------
# Utilities
# -----------------------------

def sat_unit(z: float) -> float:
    """Saturation to [-1, 1]."""
    if z > 1.0:
        return 1.0
    if z < -1.0:
        return -1.0
    return z


def sat_over_alpha(e: float, alpha: float) -> float:
    """Smooth sign: sat(e/alpha). If alpha<=0, return exact sign (0 at e=0)."""
    if alpha is None or alpha <= 0.0:
        # exact sign with zero mapping
        if e > 0.0:
            return 1.0
        if e < 0.0:
            return -1.0
        return 0.0
    return sat_unit(e / alpha)


def sqrt_abs(x: float, eps: float = 0.0) -> float:
    """√|x| with optional epsilon to avoid denormals."""
    return math.sqrt(abs(x) + eps)


# -----------------------------
# Generic 2‑SMC controller
# -----------------------------
@dataclass
class TwoSMCParams:
    """Parameters for a 2‑SMC channel.

    A, B: plant coefficients in ẋ = -A x + B v + d (A>0, B>0 for standard PMDC forms)
    k1, k2: 2‑SMC gains (k1>0, k2>0). k2 shapes the integral ẇ, k1 shapes |e|^{1/2} term.
    alpha: boundary layer width for sat(e/alpha). Use small positive to reduce chattering.
    Ts: controller integration period for w (seconds).
    """
    A: float
    B: float
    k1: float
    k2: float
    alpha: float
    Ts: float

    def validate(self) -> None:
        assert self.B != 0.0, "B must be nonzero"
        assert self.k1 > 0.0 and self.k2 > 0.0, "k1,k2 must be positive"
        assert self.Ts > 0.0, "Ts must be > 0"
        # Typical PMDC loops have A>0, B>0; we do not hard‑enforce to keep it generic.


class TwoSMC:
    """Second‑order SMC for a 1st‑order SISO channel.

    Implements (in the paper's notation):
        v = (A x_ref + x_ref_dot + k1 |e|^{1/2} sat(e/alpha) - w) / B
        w[k+1] = w[k] + Ts * ( -k2 * sat(e/alpha) )
    with e = x_ref - x.

    Usage:
        smc = TwoSMC(TwoSMCParams(A,B,k1,k2,alpha,Ts))
        v, dbg = smc.step(x=x_meas, x_ref=ref, x_ref_dot=ref_dot)
    """
    def __init__(self, params: TwoSMCParams):
        params.validate()
        self.p = params
        self.w: float = 0.0

    def reset(self, w0: float = 0.0) -> None:
        self.w = float(w0)

    def step(self, x: float, x_ref: float, x_ref_dot: float) -> Tuple[float, Dict[str, float]]:
        p = self.p
        e = x_ref - x                # sliding variable S = e (paper's convention)
        s = sat_over_alpha(e, p.alpha)
        root_abs_e = sqrt_abs(e)

        # Control law
        v = (p.A * x_ref + x_ref_dot + p.k1 * root_abs_e * s - self.w) / p.B

        # Integrator of the discontinuous term (reduces sensitivity / finite‑time)
        self.w += p.Ts * (-p.k2 * s)

        dbg = {
            'e': e,
            'sat_e_over_alpha': s,
            'root_abs_e': root_abs_e,
            'v': v,
            'w': self.w,
        }
        return v, dbg


# -----------------------------
# PMDC convenience: map physical params to (A,B)
# -----------------------------

def pmdc_speed_AB(J: float, Bm: float, Kt: float) -> Tuple[float, float]:
    """For J ω̇ = -Bm ω + Kt i - T_L  ⇒  ω̇ = -(Bm/J) ω + (Kt/J) i + d.
    Returns (Aω, Bω)."""
    return (Bm / J, Kt / J)


def pmdc_current_AB(R: float, L: float) -> Tuple[float, float]:
    """For L i̇ = -R i - Kb ω + V  ⇒  i̇ = -(R/L) i + (1/L) V + d.
    Returns (Ai, Bi)."""
    return (R / L, 1.0 / L)
