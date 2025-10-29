# pmdc_state_space.py
from os import name
import numpy as np
import matplotlib.pyplot as plt
import control as ct

def pmdc_ss(r_a: float, l_a: float, k_e: float, k_t: float, j_rotor: float, b: float) -> ct.StateSpace:
    """
    PMDC motor (linear viscous friction) in state-space form.
    States: x = [i, omega]^T
    Input:  u (armature voltage)
    Outputs: y = [i, omega, torque]^T with torque = k_t * i
    """
    A = np.array([[-r_a / l_a,      -k_e / l_a],
                  [ k_t / j_rotor,  -b  / j_rotor]], 
    dtype=float)

    B = np.array([[1.0 / l_a],
                   [0.0      ]],
    dtype=float)

    C = np.array([[1.0, 0.0],   # i
                  [0.0, 1.0],   # omega
                  [k_t, 0.0]],  # tourque
    dtype=float)

    D = np.zeros((3, 1))

    return ct.ss(A, B, C, D)   # continuous-time system

def step_u(T, V0, V1, t_step):
    return np.where(T >= t_step, V1, V0)

def sag_u(T, V_nom, V_sag, t_start, t_end):
    return np.where((T >= t_start) & (T <= t_end), V_sag, V_nom)