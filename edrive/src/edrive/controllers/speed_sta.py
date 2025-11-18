# # src/control/sta.py
# from dataclasses import dataclass

# @dataclass
# class SuperTwistingController:
#     """
#     Super-Twisting Algorithm (STA) for scalar sliding variable x_s.

#     u = lambda * |x_s|^{1/2} * sgn(x_s) + u1
#     u1_dot = alpha * sgn(x_s)

#     Parameters
#     ----------
#     lam : float
#         λ > 0 (gain on sqrt term)
#     alpha : float
#         α > 0 (gain on integral term)
#     dt : float
#         sampling time [s]
#     u_min, u_max : float
#         saturation limits for control output u
#     sgn_eps : float
#         small band for soft sign near 0 to reduce chattering (0 = hard sign)
#     """
#     lam: float
#     alpha: float
#     dt: float
#     u_min: float
#     u_max: float
#     sgn_eps: float = 0.0

#     # internal state
#     u1: float = 0.0

#     def reset(self, u1: float = 0.0):
#         self.u1 = u1

#     def _softsign(self, x: float) -> float:
#         if self.sgn_eps <= 0.0:
#             return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)
#         # smooth sign in a small neighborhood of zero
#         if x > self.sgn_eps:
#             return 1.0
#         if x < -self.sgn_eps:
#             return -1.0
#         return x / self.sgn_eps  # linear in [-eps, eps]

#     def step(self, x_s: float) -> float:
#         """One sample update; returns saturated control u."""
#         sgn_x = self._softsign(x_s)
#         u_sta = self.lam * (abs(x_s) ** 0.5) * sgn_x + self.u1

#         # discrete integration of u1_dot = alpha * sgn(x_s)
#         self.u1 += self.alpha * sgn_x * self.dt

#         # saturation
#         if u_sta > self.u_max:
#             u = self.u_max
#         elif u_sta < self.u_min:
#             u = self.u_min
#         else:
#             u = u_sta
#         return u

