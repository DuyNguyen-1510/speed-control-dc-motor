# observers/boost_obs.py
"""
Boost Observer (state + disturbance) — per the paper

States of the observer:
    z1_hat ~ i_L          (inductor current estimate)
    z2_hat ~ V_o - V_ref  (output voltage deviation estimate)
    p      — auxiliary state for disturbance observer
Disturbance estimate:
    w_hat = p + k3 * z2_hat

Equations:
    e2 = (v_o - V_ref) - z2_hat
    dz1_hat/dt = -(R_L/L) * z1_hat - ((1 - d)/L) * (z2_hat + V_ref) + (V_i/L) + k1 * e2
    dz2_hat/dt = (z1_hat/C) + w_hat + k2 * e2
    dp/dt      = -k3 * ( z1_hat/C + w_hat )

Notes:
- This module does NOT implement the controller; it only provides the observer block.
- Keep d in [0, 1]. V_ref is the target output voltage.
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class BoostObserverParams:
    L: float          # Inductance [H]
    C: float          # Capacitance [F]
    R_L: float        # Inductor series resistance [Ohm]
    V_ref: float      # Desired output voltage V*
    k1: float         # Observer gains (per paper)
    k2: float
    k3: float
    d_min: float = 0.0
    d_max: float = 1.0

class BoostObserver:
    def __init__(self, params: BoostObserverParams):
        self.p = 0.0
        self.z1_hat = 0.0
        self.z2_hat = 0.0
        self.pms = params

    # ----- lifecycle -----
    def reset(self, z1_hat: float = 0.0, z2_hat: float = 0.0, p: float = 0.0) -> None:
        """Set initial observer states."""
        self.z1_hat = z1_hat
        self.z2_hat = z2_hat
        self.p = p

    # ----- internal helpers -----
    def _clip_d(self, d: float) -> float:
        return max(self.pms.d_min, min(self.pms.d_max, d))

    def _deriv(self, z1: float, z2: float, p: float, vi: float, vo: float, d: float):
        """Right-hand side (continuous-time) of the observer dynamics."""
        L, C, RL, Vref = self.pms.L, self.pms.C, self.pms.R_L, self.pms.V_ref
        k1, k2, k3 = self.pms.k1, self.pms.k2, self.pms.k3
        d = self._clip_d(d)

        z2_meas = vo - Vref          # true deviation
        e2 = z2_meas - z2            # output estimation error
        w_hat = p + k3 * z2

        dz1 = -(RL / L) * z1 - ((1.0 - d) / L) * (z2 + Vref) + (vi / L) + k1 * e2
        dz2 = (z1 / C) + w_hat + k2 * e2
        dp  = -k3 * (z1 / C + w_hat)
        return dz1, dz2, dp

    # ----- public API -----
    def step(self, dt: float, vi: float, vo: float, d: float) -> Dict[str, float]:
        """
        Advance the observer by one time step using RK4.
        Inputs:
            dt : simulation step [s]
            vi : input source voltage V_i [V]
            vo : measured output voltage V_o [V]
            d  : duty ratio in [0,1]
        Returns dict with iL_hat, vo_hat, w_hat, e2.
        """
        z1_0, z2_0, p0 = self.z1_hat, self.z2_hat, self.p

        # RK4 stages
        k1 = self._deriv(z1_0,                 z2_0,                 p0,                 vi, vo, d)
        k2 = self._deriv(z1_0 + 0.5*dt*k1[0],  z2_0 + 0.5*dt*k1[1],  p0  + 0.5*dt*k1[2], vi, vo, d)
        k3 = self._deriv(z1_0 + 0.5*dt*k2[0],  z2_0 + 0.5*dt*k2[1],  p0  + 0.5*dt*k2[2], vi, vo, d)
        k4 = self._deriv(z1_0 + dt*k3[0],      z2_0 + dt*k3[1],      p0  + dt*k3[2],     vi, vo, d)

        self.z1_hat = z1_0 + (dt/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        self.z2_hat = z2_0 + (dt/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        self.p      = p0   + (dt/6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])

        # Outputs (consistent with the paper)
        w_hat = self.p + self.pms.k3 * self.z2_hat
        vo_hat = self.pms.V_ref + self.z2_hat
        e2 = (vo - self.pms.V_ref) - self.z2_hat

        return {"iL_hat": self.z1_hat, "vo_hat": vo_hat, "w_hat": w_hat, "e2": e2}

    def outputs(self) -> Dict[str, float]:
        """Return current estimates without stepping."""
        w_hat = self.p + self.pms.k3 * self.z2_hat
        vo_hat = self.pms.V_ref + self.z2_hat
        return {"iL_hat": self.z1_hat, "vo_hat": vo_hat, "w_hat": w_hat}
