# # src/observer/sm_observer.py
# from dataclasses import dataclass
# from typing import Protocol, Tuple

# class MotorParamsLike(Protocol):
#     r_a: float
#     l_a: float
#     k_e: float
#     k_t: float
#     j_rotor: float
#     b: float

# @dataclass
# class SMObserver:
#     """
#     Sliding-Mode Observer (sensorless) for PMDC motor using current-only measurement.

#     Plant sign conventions (consistent with dc_motor.py):
#         L di/dt = -R i - k_e * omega + v_a
#         J dω/dt =  k_t i - b ω      - T_L

#     Observer (continuous-time idea, integrated in discrete time with dt):
#         e_i      = i_meas - i_hat
#         v_inj    = U_M * sgn(e_i)            # [V] sliding injection (soft sign optional)
#         d i_hat  = (-R*i_hat - k_e*omega_hat + v_a + v_inj)/L
#         d ω_hat  = ( k_t*i_used - b*omega_hat - T_L_hat)/J + l1*e_i
#         d T̂_L    = l2*e_i

#     Notes:
#       - i_used can be i_meas (default) to reduce lag, or i_hat if you prefer full observer consistency.
#       - v_inj magnitude U_M should exceed bounds of uncertainties to guarantee sliding.
#       - l1, l2 are chosen (pole placement) so estimation error for [ω_hat, T̂_L] decays fast.

#     Parameters
#     ----------
#     params : MotorParamsLike
#         Motor parameters: r_a, l_a, k_e, k_t, j_rotor, b
#     dt : float
#         Sampling time [s]
#     U_M : float
#         Magnitude of sliding injection [V]
#     l1, l2 : float
#         Observer gains (mechanical channel)
#     sgn_eps : float
#         Soft-sign width; 0 => hard sign
#     use_meas_current_in_mech : bool
#         If True, uses i_meas in ω̂ dynamics; else uses i_hat
#     TL_min, TL_max : float
#         Optional bounds to avoid TL_hat windup
#     """
#     params: MotorParamsLike
#     dt: float
#     U_M: float
#     l1: float
#     l2: float
#     sgn_eps: float = 0.0
#     use_meas_current_in_mech: bool = True
#     TL_min: float = -1e9
#     TL_max: float =  1e9

#     # Internal observer states
#     i_hat: float = 0.0
#     omega_hat: float = 0.0
#     TL_hat: float = 0.0

#     def reset(self, i0: float = 0.0, omega0: float = 0.0, TL0: float = 0.0):
#         self.i_hat = i0
#         self.omega_hat = omega0
#         self.TL_hat = TL0

#     def _softsign(self, x: float) -> float:
#         if self.sgn_eps <= 0.0:
#             return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)
#         # Linear in a small neighborhood to reduce chattering
#         if x > self.sgn_eps:
#             return 1.0
#         if x < -self.sgn_eps:
#             return -1.0
#         return x / self.sgn_eps

#     def step(self, i_meas: float, v_a: float) -> Tuple[float, float, float, float]:
#         """
#         One discrete-time update.

#         Parameters
#         ----------
#         i_meas : float
#             Measured armature current [A]
#         v_a : float
#             Applied armature voltage [V]

#         Returns
#         -------
#         i_hat, omega_hat, TL_hat, v_inj : tuple of floats
#             Current, speed and load-torque estimates, and the injection voltage.
#         """
#         R = self.params.r_a
#         L = self.params.l_a
#         ke = self.params.k_e
#         kt = self.params.k_t
#         J = self.params.j_rotor
#         B = self.params.b

#         if L <= 0.0 or J <= 0.0:
#             raise ValueError("l_a (L) and j_rotor (J) must be > 0")
#         dt = self.dt

#         # Innovation on current
#         e_i = i_meas - self.i_hat
#         s = self._softsign(e_i)
#         v_inj = self.U_M * s  # [V]

#         # Electrical channel (Euler)
#         di_hat = (-R * self.i_hat - ke * self.omega_hat + v_a + v_inj) / L
#         i_hat_next = self.i_hat + dt * di_hat

#         # Mechanical channel innovation uses current (choose measured by default)
#         i_used = i_meas if self.use_meas_current_in_mech else i_hat_next

#         domega_hat = (kt * i_used - B * self.omega_hat - self.TL_hat) / J + self.l1 * e_i
#         omega_hat_next = self.omega_hat + dt * domega_hat

#         dTL_hat = self.l2 * e_i
#         TL_hat_next = self.TL_hat + dt * dTL_hat
#         # Anti-windup / physical bounds
#         if TL_hat_next > self.TL_max:
#             TL_hat_next = self.TL_max
#         elif TL_hat_next < self.TL_min:
#             TL_hat_next = self.TL_min

#         # Commit
#         self.i_hat = i_hat_next
#         self.omega_hat = omega_hat_next
#         self.TL_hat = TL_hat_next

#         return self.i_hat, self.omega_hat, self.TL_hat, v_inj

