f"""
Lý thuyết Bode-Plot:
Trong thực tế chúng ta không biết hàm truyền của các hệ thống, cần đưa các tín hiệu sin với tần số
khác nhau vào hệ thống để kiểm tra biên độ và độ lệch pha đầu ra, qua đó vẽ đồ thị Bode-Plot.

tần số cắt (ωc): tại đó biên độ bằng 0 dB
biên pha dự trữ (PM): khoảng cách từ pha tại ωc tới -180°
biên độ dự trữ (GM): lượng độ lợi cần thêm để đạt 0 dB khi pha-180°

Ứng dụng của Bode-Plot:
    - Thiết kế mức dự trữ ổn định tối ưu.
    - Ứng dụng trong chuẩn đoán, bảo trì.

Quan hệ trực tiếp giữa PM và tính ổn định hệ thống:
    - PM lớn (45-60°) → Cực nằm sâu trong nửa trái mặt phẳng → ít dao động, ổn định cao
    - PM nhỏ (5-15°) → Cực gần trục ảo → dao động nhiều, thời gian thiết lập lâu

Lưu ý: Các note này được ghi lại trong quá trình học tập từ Youtube và AI.
"""

import numpy as np
import control as ct
import matplotlib.pyplot as plt

from edrive.plants.motor.pmdc_state_space import pmdc_ss, discretize
# ===== B1. BODE PLOT & MARGIN =====

# state-space full MIMO
# sử dụng tham số động cơ theo luận án của giáo sư.
sys_c = pmdc_ss(r_a = 0.25, l_a = 0.00224 , k_e = 0.035,
                k_t = 0.035, j_rotor = 0.0023, b = 0.0005)

# SISO u-> omega
sys_u_w = sys_c[1, 0]

# Chọn dải tần hợp lý (tùy chỉnh thêm nếu muốn)
omega = np.logspace(-2, 5)  # 10^-2 rad/s đến 10^5 rad/s

mag, phase, omega = ct.bode_plot(
    sys_u_w,
    omega=omega,
    dB=True,
    Hz=False,
    deg=True,
    plot=True
)

gm, pm, wg, wp = ct.margin(sys_u_w)

print("\n=== Bode margins (plant with unity feedback) ===")
print(f"Gain margin (gm)      = {gm:.3g} (linear)")
print(f"Phase margin (pm)     = {pm:.2f} deg")
print(f"Gain crossover (wg)   = {wg:.2f} rad/s")
print(f"Phase crossover (wp)  = {wp:.2f} rad/s")

import matplotlib.pyplot as plt
plt.show()
