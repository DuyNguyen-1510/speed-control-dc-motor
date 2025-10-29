# controllers/st_controller.py
"""
Super-Twisting Controller (STC) for boost converter (observer-based)

Formulas follow the paper:
  s = c1*(iL_hat - z1_star) + c2*z2_hat,  with z2_hat = (V_o_hat - V_ref)
  d = 1 - [ L * (A - v) ] / [ c1 * (z2_hat + V_ref) ],
  where
    A = c1*(-RL/L*iL_hat + Vi/L + k1*e2) + c2*( iL_hat/C + w_hat + k2*e2 )
    v = -alpha*sqrt(|s|)*sign(s) - beta*eta,     eta_dot = sign(s)

Inputs expected each step:
  - dt  : time step [s]
  - Vi  : source voltage [V]
  - obs : dict from BoostObserver.step(), keys: iL_hat, vo_hat, w_hat, e2

Outputs:
  - dict {d, s, v, eta, clipped}
Notes:
  - d is hard-clipped to [d_min, d_max]
  - 'sign' is smoothed via a saturation layer of width 'phi' (optional)
"""

from dataclasses import dataclass
from typing import Dict
import math

@dataclass
class STCParams:
    # Electrical parameters (match plant & observer)
    L: float
    C: float
    R_L: float
    V_ref: float

    # Sliding surface weights
    c1: float
    c2: float

    # Observer injection gains (from the observer you use)
    k1: float
    k2: float

    # ST gains
    alpha: float
    beta: float

    # Duty limits and numerics
    d_min: float = 0.0
    d_max: float = 1.0
    phi: float = 0.0          # boundary layer for sign (0 -> hard sign)
    denom_eps: float = 1e-9   # avoid division by zero in the denominator

class STController:
    def __init__(self, params: STCParams):
        self.pms = params
        self.eta = 0.0         # integral of sign(s)
        self.z1_star = 0.0     # iL steady ref (optional, can stay 0)

    # ---- configuration ----
    def reset(self, eta: float = 0.0) -> None:
        self.eta = eta

    def set_z1_star(self, z1_star: float) -> None:
        """Optionally set desired inductor current at equilibrium."""
        self.z1_star = z1_star

    # ---- internal helpers ----
    def _sat_sign(self, s: float) -> float:
        """Saturated sign: returns in [-1,1]. If phi=0 -> hard sign."""
        phi = self.pms.phi
        if phi <= 0.0:
            return 1.0 if s > 0.0 else (-1.0 if s < 0.0 else 0.0)
        r = s / phi
        if r > 1.0:  return 1.0
        if r < -1.0: return -1.0
        return r

    def _clip_d(self, d: float) -> float:
        return max(self.pms.d_min, min(self.pms.d_max, d))

    # ---- main API ----
    def step(self, dt: float, Vi: float, obs: Dict[str, float]) -> Dict[str, float]:
        """
        Compute duty 'd' using Super-Twisting law based on observer estimates.
        obs must provide: iL_hat, vo_hat, w_hat, e2
        """
        L, C, RL, Vref = self.pms.L, self.pms.C, self.pms.R_L, self.pms.V_ref
        c1, c2 = self.pms.c1, self.pms.c2
        k1, k2 = self.pms.k1, self.pms.k2
        alpha, beta = self.pms.alpha, self.pms.beta

        iL_hat = float(obs["iL_hat"])
        vo_hat = float(obs["vo_hat"])
        w_hat  = float(obs["w_hat"])
        e2     = float(obs.get("e2", 0.0))  # prefer from observer

        z2_hat = vo_hat - Vref
        s = c1 * (iL_hat - self.z1_star) + c2 * z2_hat

        # Super-Twisting corrector
        sgn = self._sat_sign(s)
        v = -alpha * math.sqrt(max(abs(s), 0.0)) * sgn - beta * self.eta
        self.eta += sgn * dt

        # A term (uses observer injection terms per paper)
        A = (
            c1 * (-RL / L * iL_hat + Vi / L + k1 * e2)
            + c2 * (iL_hat / C + w_hat + k2 * e2)
        )

        # Denominator c1*(z2_hat + Vref) = c1*vo_hat
        denom = c1 * (z2_hat + Vref)
        if abs(denom) < self.pms.denom_eps:
            denom = self.pms.denom_eps if denom >= 0.0 else -self.pms.denom_eps

        # Duty from exact algebra: d = 1 - [ L * (A - v) ] / denom
        d = 1.0 - (L * (A - v)) / denom
        d_clipped = self._clip_d(d)

        return {
            "d": d_clipped,
            "s": s,
            "v": v,
            "eta": self.eta,
            "clipped": (d_clipped != d),
        }


