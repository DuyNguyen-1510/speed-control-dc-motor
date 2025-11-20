import numpy as np
import control as ct
import matplotlib.pyplot as plt

from edrive.plants.motor.pmdc_state_space import pmdc_ss, discretize

# state-space full MIMO
# sử dụng tham số động cơ theo luận án của giáo sư.
sys_c = pmdc_ss(r_a = 0.25, l_a = 0.00224 , k_e = 0.035,
                k_t = 0.035, j_rotor = 0.0023, b = 0.0005)

# SISO u-> omega
sys_u_w = sys_c[1, 0]

# Kiểm tra nhanh tính ổn định qua poles, zeros
print("Open-loop poles (continuous plant u-> omega): ", ct.poles(sys_u_w))
print("Open-loop zeros (continuous plant u-> omega): ", ct.zeros(sys_u_w))

# step response (open-loop, plant: u-> omega, u là 1V)
T_end = 5
t = np.linspace(0, T_end, num=1000)
t_out, y_out = ct.step_response(sys_u_w, t)

# hệ số khuếch đại tĩnh, dùng để xác định hệ số bộ điều khiển Kp = 1 / G(0)
dc_gain = ct.dcgain(sys_u_w)
print("DC gain G(0) (omega per V):", float(dc_gain))

plt.figure(figsize=(7,4))
plt.plot(t_out, y_out, label='omega')
plt.grid(True)
plt.title('Step respone open-loop, plant u-> omega, u = 1V')
plt.xlabel('Time [s]')
plt.ylabel('Omega [rad/s] for 1 V step input')
plt.tight_layout()
plt.show()



