# pmdc_state_space.py
from os import name
import numpy as np
import matplotlib.pyplot as plt
import control as ct

def pmdc_ss(r_a: float, l_a: float, k_e: float, k_t: float,
            j_rotor: float, b: float) -> ct.StateSpace:
    """
    PMDC motor (linear viscous friction) in state-space form.

    States:  x = [i, omega]^T
    Inputs:  u = [v_a, T_L]^T  (armature voltage [V], load torque [N·m])

    Outputs: y = [i, omega]^T

    Continuous-time model:
      L di/dt = -R i - k_e * omega + v_a
      J dω/dt =  k_t i - b * ω      - T_L
    """
    A = np.array([[-r_a / l_a,      -k_e / l_a],
                  [ k_t / j_rotor,  -b  / j_rotor]], # state space matrix
    dtype=float)

    # Two-input system: [v_a, T_L]^T
    # v_a enters electrical eq.; T_L enters mechanical eq. with negative sign
    B = np.array([[1.0 / l_a,   0.0],
                    [0.0,        -1.0 / j_rotor]], dtype=float)

    C = np.array([[1.0, 0.0],   # i
                  [0.0, 1.0]],   # omega
    dtype=float)

    D = np.zeros((2, 2))

    sys_c = ct.ss(A, B, C, D,               # hệ liên tục theo thời gian dt=0
                    inputs=['v_a', 'T_L'],
                    outputs=['i', 'omega'],
                    states=['i', 'omega'])

    return sys_c

# đê tích hợp với vi xử lý thì việc thiết kế trong miền rời rạc là bắt buộc
def discretize(sys_c, Ts):
    sys_d = ct.sample_system(sys_c, Ts, method="zoh")
    return sys_d




# test chương trình
if __name__ == "__main__":
    sys_c = pmdc_ss(r_a = 1.0, l_a = 1e-3 , k_e = 0.1, k_t = 0.1,
            j_rotor = 1e-3, b = 1e-4)
    print("Continuous-time poles:", ct.poles(sys_c))
    print("Continuous-time zeros:", ct.zeros(sys_c))

    # ==== 2. Rời rạc hóa ====
    Ts = 0.0005  # 0.5 ms
    sys_d = discretize(sys_c, Ts)
    print(sys_d)
    A_d, B_d, C_d, D_d = sys_d.A, sys_d.B, sys_d.C, sys_d.D
    # print ("Matrix A_d: ", A_d)
    print("Discrete poles:", ct.poles(sys_d), "dt =", sys_d.dt)
    print("Discrete zeros:", ct.zeros(sys_d), "dt =", sys_d.dt)

    # ==== 3. Mô phỏng step điện áp + step tải ====
    T_end = 0.2
    n_steps = int(T_end / Ts)
    t = np.arange(n_steps) * Ts

    u_step = 12.0        # 12 V
    T_step = 0.2        # 0.02 N.m (ví dụ)

    load_step_time = 0.05  # [s] thời điểm xuất hiện tải

    x = np.zeros(2)        # [i, omega]
    i_hist = np.zeros(n_steps)
    w_hist = np.zeros(n_steps)
    u_hist = np.zeros(n_steps)
    Tl_hist = np.zeros(n_steps)

    for k in range(n_steps):
        tk = t[k]
        u = u_step
        # Tải: 0 trước load_step_time, T_step sau đó
        Tl = T_step if tk >= load_step_time else 0.0

        u_vec = np.array([u, Tl])       # [u, T_L]

        # x_{k+1} = A_d x_k + B_d u_k
        x = A_d @ x + B_d @ u_vec
        y = C_d @ x   # [i, omega]

        i_hist[k] = y[0]
        w_hist[k] = y[1]
        u_hist[k] = u
        Tl_hist[k] = Tl

    # ==== 4. Vẽ kết quả ====
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(t, u_hist)
    axs[0].set_ylabel("u [V]")
    axs[0].set_title("Armature voltage")

    axs[1].plot(t, Tl_hist)
    axs[1].set_ylabel("T_L [N.m]")
    axs[1].set_title("Load torque")

    axs[2].plot(t, w_hist, label="omega")
    axs[2].set_ylabel("ω [rad/s]")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_title("Speed response with load step at t = %.3f s" % load_step_time)
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
    
    
    
    
