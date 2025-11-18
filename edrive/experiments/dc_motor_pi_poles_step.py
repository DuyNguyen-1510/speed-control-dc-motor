import numpy as np
import control as ct
import matplotlib.pyplot as plt

from edrive.plants.motor.pmdc_state_space import pmdc_ss, discretize


sys_c = pmdc_ss(r_a = 1.0, l_a = 1e-3 , k_e = 0.1, k_t = 0.1,
        j_rotor = 1e-3, b = 1e-4)

# Đánh giá nhanh tính ổn định, đáp ứng nhanh/ chậm của hệ thông qua poles so với trục ảo.
print("Open-loop poles (continuous plant):", ct.poles(sys_c))
print("Open-loop zeros (continuous plant):", ct.zeros(sys_c))

# # Bước kiểm tra thông tin không gian trạng thái để chọn kênh SISO cho phù hợp.
# print("sys_full:", sys_c)
# print("Number of inputs :", sys_c.ninputs)
# print("Number of outputs:", sys_c.noutputs)

# Trong python-control, sys_c[output_index, input_index]
# Tạo plant SISO (điều khiển tốc độ thông qua điện áp vào)
# từ MIMO (state space) dùng để thiết kế PI tốc độ
sys_u_w = sys_c[1, 0]

# Rời rạc hóa hệ liên tục theo thời gian.
sys_u_w_d = ct.sample_system(sys_u_w, 0.001, "zoh")

print("Opent-loop poles (discrete branch u-> w):", ct.poles(sys_u_w_d))
print("Opent-loop zeros (discrete branch u-> w):", ct.zeros(sys_u_w_d))


print(ct.tf(sys_u_w))

# ========== 2. Định nghĩa PI controller: C(s) = Kp + Ki/s ==========
def make_pi_controller(Kp, Ki):
    """
    C(s) = Kp + Ki/s = (Kp s + Ki) / s
    """
    num = [Kp, Ki]
    den = [1.0, 0.0]  # s
    C = ct.tf(num, den)
    return C

def make_closed_loop_pi(Kp, Ki):
    C = make_pi_controller(Kp, Ki)
    L = C * sys_u_w         # loop transfer function
    T = ct.feedback(L, 1)    # closed-loop từ ref -> omega
    return T

# ========== 3. Thử vài bộ PI khác nhau ==========
# Bạn có thể chỉnh lại các giá trị này để cảm nhận
pi_params = [
    (1.0,  50.0),
    (5.0,  200.0),
    (10.0, 400.0),
]

T_end = 0.5
t = np.linspace(0, T_end, 2000)

plt.figure(figsize=(8, 5))

for (Kp, Ki) in pi_params:
    T_pi = make_closed_loop_pi(Kp, Ki)
    poles = ct.poles(T_pi)
    zeros = ct.zeros(T_pi)

    print(f"\nPI: Kp = {Kp}, Ki = {Ki}")
    print("  Closed-loop poles:", poles)
    print("  Closed-loop zeros:", zeros)

    t_out, y_out = ct.step_response(T_pi, t)
    label = f"Kp={Kp}, Ki={Ki}"
    plt.plot(t_out, y_out, label=label)

plt.title("DC Motor Speed: Step Response with Different PI Gains")
plt.xlabel("Time [s]")
plt.ylabel("Speed ω [rad/s] (ref = 1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
