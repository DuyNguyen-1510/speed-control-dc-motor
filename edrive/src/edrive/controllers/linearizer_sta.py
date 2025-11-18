# # src/control/linearizer.py
# from dataclasses import dataclass
# from typing import Protocol

# class MotorParamsLike(Protocol):
#     r_a: float
#     l_a: float
#     k_e: float
#     k_t: float
#     j_rotor: float
#     b: float

# @dataclass
# class LinearizerSTA:
#     """
#     Exact-feedback linearization for PMDC speed loop (scalar sliding variable).

#     Definitions (consistent with dc_motor.py sign conventions):
#       Mechanical: J dω/dt = k_t i - b ω - T_L
#       z1 = ω_ref - ω
#       z2 = (b/J)*ω - (k_t/J)*i + (1/J)*T̂_L
#       x_s = C*z1 + z2,  with C > 0 (tunable)

#     Inputs:
#       omega_ref : float  [rad/s]   desired speed
#       i         : float  [A]       armature current (measured)
#       omega     : float  [rad/s]   speed (measured or estimated)
#       TL_hat    : float  [N·m]     estimated load torque

#     Output:
#       x_s : float  sliding variable for STA
#     """
#     params: MotorParamsLike
#     C: float = 50.0  # choose/tune > 0

#     def xs(self, omega_ref: float, i: float, omega: float, TL_hat: float) -> float:
#         J = self.params.j_rotor
#         B = self.params.b
#         kt = self.params.k_t
#         if J <= 0.0:
#             raise ValueError("j_rotor (J) must be > 0")

#         z1 = omega_ref - omega
#         z2 = (B / J) * omega - (kt / J) * i + (1.0 / J) * TL_hat
#         return self.C * z1 + z2

# # Optional: functional helper if you prefer functions over class
# def compute_xs(params: MotorParamsLike, C: float,
#                omega_ref: float, i: float, omega: float, TL_hat: float) -> float:
#     J = params.j_rotor
#     B = params.b
#     kt = params.k_t
#     if J <= 0.0:
#         raise ValueError("j_rotor (J) must be > 0")

#     z1 = omega_ref - omega
#     z2 = (B / J) * omega - (kt / J) * i + (1.0 / J) * TL_hat
#     return C * z1 + z2
