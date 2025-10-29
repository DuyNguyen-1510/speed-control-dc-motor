# src/runtime/sim_open_loop.py
from __future__ import annotations
from ast import arg
from cProfile import label
import os, time, argparse, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Add project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from edrive.plants.boost.averaged import BoostParams, BoostParasitics, BoostAveragedPlant, step_at

def steady_state_ic(Vin: float, R: float, D: float) -> tuple[float, float]:
    """Equilibrium ideal (điểm làm việc) để bỏ quá độ khởi động."""
    vo = Vin / max(1e-12, (1.0 - D))
    iL = Vin / (R * max(1e-12, (1.0 - D))**2)
    return iL, vo

def main():
    ap = argparse.ArgumentParser("Open-loop boost: continuous plant + ZOH (no save)")
    ap.add_argument("--L", type=float, default=200e-6)
    ap.add_argument("--C", type=float, default=330e-6)

    ap.add_argument("--Vin", type=float, default=24.0)
    ap.add_argument("--Vin_drop", type=float, default=10.0,
                help="Mức Vin khi sụt (bỏ qua nếu không đặt)")
    ap.add_argument("--t_drop", type=float, default=0.02,
                help="Thời điểm bắt đầu sụt [s]")
    ap.add_argument("--t_recover", type=float, default=0.04,
                help="Thời điểm khôi phục Vin ban đầu (bỏ qua nếu giữ nguyên mức sụt)")


    ap.add_argument("--R", type=float, default=100)
    ap.add_argument("--duty", type=float, default=0.504)
    ap.add_argument("--Ts", type=float, default=10e-6)
    ap.add_argument("--duration", type=float, default=0.06)
    ap.add_argument("--init", choices=["equil","zero"], default="zero",
                    help="equil: khởi tạo tại điểm làm việc; zero: iL=vo=0")
    
    DEFAULT_NON_IDEAL = True  # đổi True/False tùy lần chạy
    ap.add_argument("--non_ideal", action="store_true", default=DEFAULT_NON_IDEAL,
                help="bật mô hình phi lý tưởng")
    ap.add_argument("--ideal", dest="non_ideal", action="store_false",
                help="tắt non_ideal khi cần")

    ap.add_argument("--rL", type=float, default=0.35)
    ap.add_argument("--rDS", type=float, default=0.0)
    ap.add_argument("--VF", type=float, default=0.0)
    ap.add_argument("--RF", type=float, default=0.0)
    args = ap.parse_args()

    par = BoostParasitics(rL=args.rL, rDS=args.rDS, VF=args.VF, RF=args.RF)

    Vin_profile = args.Vin
    if args.Vin_drop is not None and args.t_drop is not None:
        drop_fun = step_at(args.t_drop, args.Vin, args.Vin_drop)
        if args.t_recover is None:
            Vin_profile = drop_fun
        else:
            def Vin_profile(t: float) -> float:
                if t < args.t_drop:
                    return args.Vin
                if t < args.t_recover:
                    return args.Vin_drop
                return args.Vin


    cfg = BoostParams(L=args.L, C=args.C, Vin=Vin_profile, R=args.R,
                      d_min=0.02, d_max=0.98, non_ideal=args.non_ideal, par=par)

    vin_source = cfg.Vin
    if callable(vin_source):
        vin_fun = vin_source
    else:
        vin_value = float(vin_source)
        vin_fun = lambda _t, _v=vin_value: _v


    plant = BoostAveragedPlant(cfg)

    if args.init == "zero":
        plant.reset((0.0, 0.0))
    else:
        # equilibrium ideal (gần đúng cả khi bật non_ideal)
        plant.reset(steady_state_ic(args.Vin, args.R, args.duty))

    Ts, T = args.Ts, args.duration
    n = int(T / Ts)
    t = 0.0
    t_list = np.empty(n); i_list = np.empty(n); v_list = np.empty(n)
    vin_list=np.empty(n); duty_list=np.empty(n)
    for k in range(n):
        x, _ = plant.propagate(t, t + Ts, duty=args.duty)  # ZOH duty
        t += Ts
        t_list[k], i_list[k], v_list[k] = t, x[0], x[1]
        vin_list[k], duty_list[k] = vin_fun(t), args.duty

    # Vẽ không lưu file
    mpl.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 10,                 # cỡ chữ cố định (pt)
    })

    fig, ax = plt.subplots(2, 1, figsize=(5.01,4.2), dpi=300, sharex=True)
    
    # ax.plot(t_list, i_list); ax[0].set_ylabel("iL [A]")
    # ax.grid(True, alpha=0.3)

    ax[0].plot(t_list, v_list, 'r-', label="Vo - Выходное напряжение", linewidth=0.7)
    ax[0].plot(t_list, vin_list, 'b-', label="Vi - Входное напряжение", linewidth=0.7)
    ax[0].axvspan(args.t_drop, args.t_recover, alpha=0.2, color='gray', label='Период проседания напряжения')
    ax[0].set_ylabel("Напряжения [В]")
    # ax[0].set_xlabel("Время, t [с]")
    ax[0].legend()
    # ax.set_title("Boost open-loop: output vs input profile")
    ax[0].grid(True, alpha=0.3)
    # fig.suptitle("Boost open-loop (continuous plant, ZOH duty)")

    ax[1].plot(t_list, duty_list, 'm-', label="d - Скважность", linewidth=0.7)
    ax[1].axvspan(args.t_drop, args.t_recover, alpha=0.2, color='gray', label='Период проседания напряжения')
    ax[1].set_ylabel("Скважность, d")
    ax[1].set_xlabel("Время, t [с]")
    ax[1].set_ylim(0, 1)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
