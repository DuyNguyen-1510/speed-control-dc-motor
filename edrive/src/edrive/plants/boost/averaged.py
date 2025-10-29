"""Placeholder: averaged models for power converters (e.g., boost)."""
# src/plant/power_avg.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Union, Dict, Tuple
import numpy as np
from scipy.integrate import solve_ivp

ScalarOrFun = Union[float, Callable[[float], float]]

@dataclass
class BoostParasitics:
    rL: float = 0.0      # inductor ESR [ohm]
    rDS: float = 0.0     # MOSFET on-res [ohm]
    VF: float = 0.0      # diode forward drop [V]
    RF: float = 0.0      # diode series res [ohm]
    # NOTE: capacitor ESR (rC) không đưa vào ở bản tối thiểu này
    # vì mô hình chuẩn sẽ cần state v_C, xem TODO ở dưới.

@dataclass
class BoostParams:
    L: float
    C: float
    Vin: ScalarOrFun        # hằng hoặc hàm t -> Vin(t)
    R: ScalarOrFun          # hằng hoặc hàm t -> R(t)
    d_min: float = 0.02
    d_max: float = 0.98
    non_ideal: bool = False
    par: BoostParasitics = field(default_factory=BoostParasitics)
    rtol: float = 1e-8
    atol: float = 1e-9

def _to_fun(v: ScalarOrFun) -> Callable[[float], float]:
    if callable(v):
        return v
    val = float(v)
    return lambda t: val

class BoostAveragedPlant:
    """
    Averaged CCM model for DC-DC Boost (continuous-time plant, ZOH on duty).
    States: x = [iL, vo]
        iL: inductor current [A]
        vo: output voltage across load R [V]
    Inputs (held ZOH over [t0,t1]): duty d in [d_min, d_max]
    Vin(t), R(t): constant or callable profiles.
    """
    def __init__(self, cfg: BoostParams):
        self.cfg = cfg
        self._Vin = _to_fun(cfg.Vin)
        self._R = _to_fun(cfg.R)
        self.x = np.zeros(2, dtype=float)  # [iL, vo]
        self._held_d = float(np.clip(0.5, cfg.d_min, cfg.d_max))

    def reset(self, x0: Tuple[float, float] = (0.0, 0.0)) -> None:
        self.x = np.array(x0, dtype=float)

    # ---- Core ODE (ideal + optional non-ideal lumped) ----
    def _f(self, t: float, x: np.ndarray) -> np.ndarray:
        L, C = self.cfg.L, self.cfg.C
        iL, vo = x
        d = self._held_d          # ZOH: duty giữ hằng trong bước tích phân
        d = float(np.clip(d, self.cfg.d_min, self.cfg.d_max))
        Vin = self._Vin(t)
        R = max(1e-6, self._R(t)) # tránh chia 0

        if not self.cfg.non_ideal:
            # --- Ideal averaged CCM ---
            diL = (Vin - (1.0 - d) * vo) / L
            dvo = ((1.0 - d) * iL - vo / R) / C
            return np.array([diL, dvo])

        # --- Non-ideal (lumped, bản tối thiểu) ---
        # cần kiểm chứng qua các bài báo, sách (chưa đủ tin tưởng)
        p: BoostParasitics = self.cfg.par
        # Sụt áp & tổn hao gộp:
        #   - Khi ON (tỷ lệ d): rDS * iL
        #   - ESR cuộn cảm rL * iL
        #   - Khi OFF (tỷ lệ 1-d): diode drop VF + RF*iD, với iD ≈ (1-d)*iL (averaged)
        iD = (1.0 - d) * iL
        v_drop = p.rL * iL + d * (p.rDS * iL) + (1.0 - d) * (p.VF + p.RF * iD)

        diL = (Vin - v_drop - (1.0 - d) * vo) / L
        dvo = ((1.0 - d) * iL - vo / R) / C  # chưa mô hình ESR tụ (rC)
        return np.array([diL, dvo])

    # ---- Integrate ODE on [t0, t1] with ZOH duty ----
    def propagate(self, t0: float, t1: float, duty: float) -> Tuple[np.ndarray, Dict[str, float]]:
        self._held_d = float(np.clip(duty, self.cfg.d_min, self.cfg.d_max))
        sol = solve_ivp(self._f, (t0, t1), self.x,
                        method="RK45", rtol=self.cfg.rtol, atol=self.cfg.atol,
                        dense_output=False)
        self.x = sol.y[:, -1]
        return self.x.copy(), {"iL": self.x[0], "vo": self.x[1], "d": self._held_d}

# ------------ Helpers (profiles & quick checks) ------------
def constant(v: float) -> Callable[[float], float]:
    return _to_fun(v)

def step_at(t_step: float, v0: float, v1: float) -> Callable[[float], float]:
    def f(t: float) -> float:
        return v0 if t < t_step else v1
    return f

def rhp_zero_freq(R: float, D: float, L: float) -> float:
    """RHP zero frequency [Hz] for ideal boost: wz = R(1-D)^2/L."""
    wz = R * (1.0 - D) ** 2 / max(1e-12, L)
    return wz / (2.0 * np.pi)

