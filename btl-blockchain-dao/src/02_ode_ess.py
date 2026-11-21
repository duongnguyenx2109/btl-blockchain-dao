import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==============================================================================
# 1. NẠP DỮ LIỆU THẬT TỪ BƯỚC 1 ĐỂ HIỆU CHỈNH ĐIỀU KIỆN BAN ĐẦU x(0)
# ==============================================================================
def load_real_dao_calibration():
    voters_path = "data/real_dao_voters.csv"
    df = pd.read_csv(voters_path)
    # Cử tri tích cực (Active): những ví tham gia >= 20% số đề xuất thực tế
    active_ratio = float(np.mean(df["participation_rate"] >= 0.20))
    active_ratio = float(np.clip(active_ratio, 0.05, 0.45))
    # Giả định 5% mầm mống nhóm thông đồng ban đầu, phần còn lại là nhóm ủy quyền thụ động
    collude_init = 0.05
    delegating_ratio = round(1.0 - active_ratio - collude_init, 4)

    print(f"[OK] Da nap du lieu that ENS DAO ({len(df)} vi):")
    print(f"     -> Ty le khoi tao x(0) = [xA={active_ratio:.2%}, xD={delegating_ratio:.2%}, xC={collude_init:.2%}]")
    return [active_ratio, delegating_ratio, collude_init]

# ==============================================================================
# 2. HÀM LỢI ÍCH & HỆ PHƯƠNG TRÌNH VI PHÂN REPLICATOR DYNAMICS
# ==============================================================================
def payoffs(xA, xD, xC, p):
    # Quyền biểu quyết hiệu dụng của phe trung thực và phe thông đồng
    w_honest = xA + (1.0 - p["alpha"]) * xD
    w_collude = xC + p["alpha"] * xD
    p_win = 1.0 / (1.0 + np.exp(-p["k"] * (w_collude - w_honest)))

    pi_A = (1.0 - p_win) * p["R"] - p_win * p["L"] - p["c_v"]
    pi_D = (1.0 - p_win) * p["R"] - p_win * p["L"] - p["c_d"]
    pi_C = p_win * (p["B"] / (xC + 0.05) - p["L"]) + (1.0 - p_win) * p["R"] - p["c_c"]
    return pi_A, pi_D, pi_C, p_win

def replicator_ode(t, x, p):
    xA, xD, xC = np.clip(x, 1e-9, 1.0)
    s = xA + xD + xC
    xA, xD, xC = xA / s, xD / s, xC / s

    pi_A, pi_D, pi_C, _ = payoffs(xA, xD, xC, p)
    pi_bar = xA * pi_A + xD * pi_D + xC * pi_C

    return [
        xA * (pi_A - pi_bar),
        xD * (pi_D - pi_bar),
        xC * (pi_C - pi_bar)
    ]

def check_ess_stability(x_eq, p, eps=1e-6):
    """
    Rút gọn hệ 3 biến về không gian 2D (xA, xC) với xD = 1 - xA - xC.
    Tính ma trận Jacobian J(x*) và xét dấu phần thực các giá trị riêng.
    """
    def f_2d(u):
        xA, xC = u
        xD = max(0.0, 1.0 - xA - xC)
        dxA, _, dxC = replicator_ode(0, [xA, xD, xC], p)
        return np.array([dxA, dxC])

    u0 = np.array([x_eq[0], x_eq[2]])
    J = np.zeros((2, 2))
    for j in range(2):
        du = np.zeros(2)
        du[j] = eps
        J[:, j] = (f_2d(u0 + du) - f_2d(u0 - du)) / (2 * eps)

    eigvals = np.linalg.eigvals(J)
    max_real = float(np.max(np.real(eigvals)))
    is_ess = max_real < 0.0
    return eigvals, is_ess

# ==============================================================================
# 3. CHẠY 3 KỊCH BẢN & XUẤT BẢNG + BIỂU ĐỒ VÀO THƯ MỤC results/
# ==============================================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    x0 = load_real_dao_calibration()

    scenarios = {
        "KB1": {
            "label": "KB1: Hien trang ENS (c_v=1.5, alpha=0.45)",
            "params": {"R": 10.0, "L": 15.0, "B": 8.0, "c_v": 1.5, "c_d": 0.1, "c_c": 1.0, "alpha": 0.45, "k": 8.0}
        },
        "KB2": {
            "label": "KB2: Ca map thao tung uy quyen (alpha=0.65)",
            "params": {"R": 10.0, "L": 15.0, "B": 8.0, "c_v": 1.5, "c_d": 0.1, "c_c": 0.8, "alpha": 0.65, "k": 8.0}
        },
        "KB3": {
            "label": "KB3: Hoan gas & Tran uy quyen (c_v=0.15, alpha=0.20)",
            "params": {"R": 10.0, "L": 15.0, "B": 8.0, "c_v": 0.15, "c_d": 0.1, "c_c": 1.2, "alpha": 0.20, "k": 8.0}
        }
    }

    t_eval = np.linspace(0, 30, 300)
    summary_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)

    for idx, (key, sc) in enumerate(scenarios.items()):
        p = sc["params"]
        sol = solve_ivp(replicator_ode, [0, 30], x0, args=(p,), t_eval=t_eval, method="RK45")
        x_end = np.clip(sol.y[:, -1], 0.0, 1.0)
        x_end /= np.sum(x_end)

        _, _, _, p_win_end = payoffs(x_end[0], x_end[1], x_end[2], p)
        eigvals, is_ess = check_ess_stability(x_end, p)

        summary_rows.append({
            "Kich_ban": sc["label"],
            "xA_cuoi": round(float(x_end[0]), 4),
            "xD_cuoi": round(float(x_end[1]), 4),
            "xC_cuoi": round(float(x_end[2]), 4),
            "P_tan_cong_thang": round(float(p_win_end), 4),
            "Eigenvalues_J": f"{eigvals[0]:.3f}, {eigvals[1]:.3f}",
            "Ket_luan_ESS": "On dinh ESS" if is_ess else "Chua hoi tu han"
        })

        ax = axes[idx]
        ax.plot(sol.t, sol.y[0], label="Active Voters (xA)", color="#1f77b4", lw=2.2)
        ax.plot(sol.t, sol.y[1], label="Delegating (xD)", color="#ff7f0e", lw=2.2)
        ax.plot(sol.t, sol.y[2], label="Colluding (xC)", color="#d62728", lw=2.2)
        ax.set_title(sc["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Thoi gian tien hoa (t)")
        if idx == 0:
            ax.set_ylabel("Ty le quan the")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8.5)

    plt.tight_layout()
    plt.savefig("results/fig1_ode_trajectories.png", dpi=300)
    plt.close()

    # Biểu đồ 2: Giản đồ pha (Phase Portrait) trên mặt phẳng (xA, xC) cho Kịch bản 2
    p_attack = scenarios["KB2"]["params"]
    grid = np.linspace(0.02, 0.96, 16)
    XA, XC = np.meshgrid(grid, grid)
    U, V = np.zeros_like(XA), np.zeros_like(XC)

    for i in range(XA.shape[0]):
        for j in range(XA.shape[1]):
            xa, xc = XA[i, j], XC[i, j]
            if xa + xc <= 0.98:
                xd = 1.0 - xa - xc
                dxa, _, dxc = replicator_ode(0, [xa, xd, xc], p_attack)
                U[i, j], V[i, j] = dxa, dxc
            else:
                U[i, j], V[i, j] = np.nan, np.nan

    plt.figure(figsize=(6.5, 5.5))
    plt.quiver(XA, XC, U, V, color="#2c3e50", alpha=0.75)
    plt.plot([0, 1], [1, 0], "r--", lw=1.5, label="Bien don hinh (xA + xC = 1)")
    plt.fill_between([0, 1], [1, 0], 1, color="#f2f2f2")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Ty le cu tri tich cuc (xA)")
    plt.ylabel("Ty le nhom thong dong (xC)")
    plt.title("Gian do pha Replicator Dynamics (Kich ban tan cong)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/fig2_phase_portrait.png", dpi=300)
    plt.close()

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/ode_ess_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== BANG KET QUA PHAN TICH CAN BANG ESS ===")
    print(df_summary.to_string(index=False))
    print("\n==> Da luu 2 bieu do PNG va 1 file CSV vao thu muc results/!")