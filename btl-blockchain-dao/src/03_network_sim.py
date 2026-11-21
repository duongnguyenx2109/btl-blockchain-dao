import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ==============================================================================
# 1. NẠP 447 VÍ THẬT CỦA ENS DAO VÀ DỰNG ĐỒ THỊ QUAN HỆ XÃ HỘI (NETWORKX)
# ==============================================================================
def load_real_ens_network(seed=42):
    df = pd.read_csv("data/real_dao_voters.csv")
    N = len(df)
    raw_vp = df["max_vp"].values.astype(float)
    # Chuẩn hóa tổng quyền biểu quyết về 1.0 (100%) để dễ theo dõi tỷ trọng thao túng
    norm_vp = raw_vp / np.sum(raw_vp)
    participation = df["participation_rate"].values.astype(float)

    # Xây dựng mạng xã hội Scale-Free có trọng số ưu tiên kết nối tới các đại biểu lớn
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Gán thuộc tính thật lên từng nút của đồ thị NetworkX
    for i in range(N):
        G.nodes[i]["address"] = df.loc[i, "voter"]
        G.nodes[i]["vp_share"] = float(norm_vp[i])
        G.nodes[i]["participation"] = float(participation[i])

    # Tạo cạnh kết nối: mỗi ví nhỏ kết nối tới 3 đại biểu/láng giềng theo xác suất tỷ lệ với uy tín (vp)
    weights = np.power(norm_vp + 1e-6, 0.45)
    weights /= np.sum(weights)
    for u in range(N):
        targets = rng.choice(N, size=4, replace=False, p=weights)
        for v in targets:
            if u != v:
                G.add_edge(u, int(v))

    return G, norm_vp, participation

def gini_coefficient(values):
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[arr > 0]
    if len(arr) == 0 or np.sum(arr) == 0:
        return 0.0
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * arr) / (n * np.sum(arr)))

# ==============================================================================
# 2. MÔ PHỎNG 1.000 VÒNG BỎ PHIẾU ỦY QUYỀN LỎNG TRÊN ĐỒ THỊ NETWORKX
# ==============================================================================
def run_network_simulation(
    G, norm_vp, participation,
    rounds=1000, c_v=1.5, c_d=0.1, c_c=1.0,
    alpha=0.45, delegation_cap=None, beta=0.6, mu=0.005, seed=42
):
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    neighbors_list = [list(G.neighbors(u)) for u in range(N)]

    # Khởi tạo chiến lược từ hành vi thật: 0 = Active (A), 1 = Delegating (D), 2 = Colluding (C)
    strategies = np.where(participation >= 0.20, 0, 1)
    # Gieo 5% mầm mống nhóm thông đồng ban đầu (bao gồm một vài đại biểu nhóm trung bình)
    n_collude_init = max(1, int(0.05 * N))
    collude_idx = rng.choice(np.arange(5, min(80, N)), size=n_collude_init, replace=False)
    strategies[collude_idx] = 2

    hist_xA, hist_xD, hist_xC = [], [], []
    hist_wC, hist_gini, hist_wins = [], [], []
    last_delegate_map = {}

    R, L, B = 10.0, 15.0, 8.0

    for r in range(rounds):
        # Bước A: Các nút nhóm D chọn người nhận ủy quyền trong mạng lưới
        delegate_to = np.full(N, -1, dtype=int)
        for u in range(N):
            if strategies[u] == 1 and neighbors_list[u]:
                nbrs = neighbors_list[u]
                nbr_strats = strategies[nbrs]
                c_nbrs = [nbrs[i] for i, s in enumerate(nbr_strats) if s == 2]
                a_nbrs = [nbrs[i] for i, s in enumerate(nbr_strats) if s == 0]

                if c_nbrs and rng.random() < alpha:
                    delegate_to[u] = int(rng.choice(c_nbrs))
                elif a_nbrs:
                    delegate_to[u] = int(rng.choice(a_nbrs))
                else:
                    delegate_to[u] = int(rng.choice(nbrs))

        # Bước B: Lan truyền ủy quyền bắc cầu & đóng băng chu trình (Cycle Freezing)
        eff_power = np.zeros(N, dtype=float)
        for u in range(N):
            curr = u
            visited = set()
            frozen = False
            while strategies[curr] == 1 and delegate_to[curr] != -1:
                if curr in visited:
                    frozen = True
                    break
                visited.add(curr)
                curr = delegate_to[curr]
            if not frozen and strategies[curr] != 1:
                eff_power[curr] += norm_vp[u]

        # Bước C: Áp dụng Trần ủy quyền (Delegation Cap) nếu có
        if delegation_cap is not None:
            eff_power = np.minimum(eff_power, delegation_cap)

        # Bước D: Kiểm phiếu vòng r
        w_A = float(np.sum(eff_power[strategies == 0]))
        w_C = float(np.sum(eff_power[strategies == 2]))
        total_cast = w_A + w_C
        share_wC = (w_C / total_cast) if total_cast > 0 else 0.0
        collude_win = 1.0 if w_C > w_A else 0.0

        # Bước E: Tính lợi ích từng nút
        num_C = max(1, int(np.sum(strategies == 2)))
        reward_C = (B * (N / 10.0)) / num_C
        base_payoff = (1.0 - collude_win) * R - collude_win * L

        payoffs = np.full(N, base_payoff, dtype=float)
        payoffs[strategies == 0] -= c_v
        payoffs[strategies == 1] -= c_d
        payoffs[strategies == 2] = (
            collude_win * (reward_C - L) + (1.0 - collude_win) * R - c_c
        )

        # Bước F: Cập nhật chiến lược theo quy tắc bắt chước Fermi
        new_strategies = strategies.copy()
        for u in range(N):
            if rng.random() < mu:
                new_strategies[u] = int(rng.integers(0, 3))
                continue
            nbrs = neighbors_list[u]
            if nbrs:
                v = int(rng.choice(nbrs))
                diff = np.clip(payoffs[v] - payoffs[u], -30.0, 30.0)
                prob = 1.0 / (1.0 + np.exp(-beta * diff))
                if rng.random() < prob:
                    new_strategies[u] = strategies[v]
        strategies = new_strategies

        hist_xA.append(float(np.mean(strategies == 0)))
        hist_xD.append(float(np.mean(strategies == 1)))
        hist_xC.append(float(np.mean(strategies == 2)))
        hist_wC.append(share_wC)
        hist_gini.append(gini_coefficient(eff_power))
        hist_wins.append(collude_win)

        if r == rounds - 1:
            last_delegate_map = {
                "strategies": strategies.copy(),
                "eff_power": eff_power.copy(),
                "delegate_to": delegate_to.copy()
            }

    return {
        "xA": np.array(hist_xA),
        "xD": np.array(hist_xD),
        "xC": np.array(hist_xC),
        "wC": np.array(hist_wC),
        "gini": np.array(hist_gini),
        "wins": np.array(hist_wins),
        "snapshot": last_delegate_map
    }

# ==============================================================================
# 3. CHẠY THỰC NGHIỆM 3 KỊCH BẢN & XUẤT BIỂU ĐỒ CHO BÁO CÁO
# ==============================================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    G, norm_vp, participation = load_real_ens_network(seed=42)
    print(f"[OK] Da khoi tao do thi NetworkX: {G.number_of_nodes()} nut, {G.number_of_edges()} canh.")

    configs = {
        "KB1_HienTrang": {
            "label": "KB1: Hien trang (Khong tran uy quyen)",
            "params": {"c_v": 1.5, "c_d": 0.1, "c_c": 1.0, "alpha": 0.45, "delegation_cap": None},
            "color": "#ff7f0e"
        },
        "KB2_TanCong": {
            "label": "KB2: Ca map thong dong (alpha=0.65)",
            "params": {"c_v": 1.5, "c_d": 0.1, "c_c": 0.8, "alpha": 0.65, "delegation_cap": None},
            "color": "#d62728"
        },
        "KB3_PhongVe": {
            "label": "KB3: Hoan gas + Tran uy quyen 8%",
            "params": {"c_v": 0.15, "c_d": 0.1, "c_c": 1.2, "alpha": 0.20, "delegation_cap": 0.08},
            "color": "#2ca02c"
        }
    }

    outputs = {}
    summary_rows = []

    for key, cfg in configs.items():
        print(f"Dang chay mo phong 1.000 vong cho {cfg['label']}...")
        res = run_network_simulation(G, norm_vp, participation, rounds=1000, **cfg["params"])
        outputs[key] = res

        summary_rows.append({
            "Kich_ban": cfg["label"],
            "Ty_le_Active_cuoi": f"{res['xA'][-1]:.2%}",
            "Ty_le_Delegating_cuoi": f"{res['xD'][-1]:.2%}",
            "Ty_le_Colluding_cuoi": f"{res['xC'][-1]:.2%}",
            "Quyen_bieu_quyet_phe_C_TB": f"{np.mean(res['wC'][-200:]):.2%}",
            "He_so_Gini_quyen_luc_TB": round(float(np.mean(res['gini'][-200:])), 4),
            "Ty_le_tan_cong_thanh_cong": f"{np.mean(res['wins']):.2%}"
        })

    # --- VẼ BIỂU ĐỒ 3: DIỄN BIẾN 1.000 VÒNG BỎ PHIẾU (4 SUBPLOTS) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    rounds_axis = np.arange(1, 1001)

    # Subplot 1: Tỷ trọng quyền biểu quyết phe thông đồng (wC)
    for key, cfg in configs.items():
        # Làm mượt đường cong bằng trung bình trượt 15 vòng cho dễ nhìn
        smooth_wC = pd.Series(outputs[key]["wC"]).rolling(15, min_periods=1).mean()
        axes[0, 0].plot(rounds_axis, smooth_wC * 100, label=cfg["label"], color=cfg["color"], lw=2)
    axes[0, 0].axhline(50, color="black", linestyle="--", lw=1.2, label="Nguong qua ban (50%)")
    axes[0, 0].set_title("1. Quyen bieu quyet phe thong dong qua 1.000 vong", fontweight="bold")
    axes[0, 0].set_ylabel("Ty trong quyen bieu quyet (%)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(fontsize=8.5)

    # Subplot 2: Tỷ lệ cử tri tích cực (xA)
    for key, cfg in configs.items():
        axes[0, 1].plot(rounds_axis, outputs[key]["xA"] * 100, label=cfg["label"], color=cfg["color"], lw=2)
    axes[0, 1].set_title("2. Dien bien ty le cu tri tich cuc (Active Voters)", fontweight="bold")
    axes[0, 1].set_ylabel("Ty le so vi (%)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(fontsize=8.5)

    # Subplot 3: Hệ số bất bình đẳng quyền lực (Gini)
    for key, cfg in configs.items():
        smooth_gini = pd.Series(outputs[key]["gini"]).rolling(15, min_periods=1).mean()
        axes[1, 0].plot(rounds_axis, smooth_gini, label=cfg["label"], color=cfg["color"], lw=2)
    axes[1, 0].set_title("3. Muc do tap trung quyen luc (He so Gini hieu dung)", fontweight="bold")
    axes[1, 0].set_xlabel("Vong bo phieu (t)")
    axes[1, 0].set_ylabel("He so Gini")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(fontsize=8.5)

    # Subplot 4: Biểu đồ cột tỷ lệ tấn công thành công
    labels_short = ["KB1: Hien trang", "KB2: Ca map thong dong", "KB3: Hoan gas + Tran 8%"]
    win_rates = [np.mean(outputs[k]["wins"]) * 100 for k in configs]
    colors_bar = [configs[k]["color"] for k in configs]
    bars = axes[1, 1].bar(labels_short, win_rates, color=colors_bar, width=0.5)
    axes[1, 1].set_title("4. Ty le tan cong thao tung quy thanh cong (%)", fontweight="bold")
    axes[1, 1].set_ylabel("Ty le thang cu cua phe C (%)")
    axes[1, 1].set_ylim(0, 110)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, win_rates):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, val + 2.5, f"{val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/fig3_network_1000_rounds.png", dpi=300)
    plt.close()

    # --- VẼ BIỂU ĐỒ 4: CẤU TRÚC MẠNG LƯỚI TOP 60 ĐẠI BIỂU (KB2 vs KB3) ---
    top_nodes = list(range(60))
    subG = G.subgraph(top_nodes)
    pos = nx.spring_layout(subG, seed=42, k=0.45)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    color_map = {0: "#1f77b4", 1: "#ffbb78", 2: "#d62728"}

    for idx, (key, title) in enumerate([
        ("KB2_TanCong", "Mang luoi Top 60 vi - KB2 (Bi nhom C thao tung)"),
        ("KB3_PhongVe", "Mang luoi Top 60 vi - KB3 (On dinh nho Tran uy quyen)")
    ]):
        snap = outputs[key]["snapshot"]
        node_colors = [color_map[int(snap["strategies"][u])] for u in top_nodes]
        node_sizes = [120 + snap["eff_power"][u] * 4500 for u in top_nodes]

        nx.draw_networkx_edges(subG, pos, ax=axes[idx], alpha=0.22, edge_color="#7f8c8d")
        nx.draw_networkx_nodes(subG, pos, ax=axes[idx], node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=0.6)
        axes[idx].set_title(title, fontsize=10.5, fontweight="bold")
        axes[idx].axis("off")

    # Chú thích màu sắc
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Active Voters (A)", markerfacecolor="#1f77b4", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Delegating (D)", markerfacecolor="#ffbb78", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Colluding (C)", markerfacecolor="#d62728", markersize=9),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("results/fig4_dao_delegation_graph.png", dpi=300)
    plt.close()

    df_net = pd.DataFrame(summary_rows)
    df_net.to_csv("results/network_sim_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== BANG KET QUA MO PHONG MANG LUOI 1.000 VONG (447 VI ENS) ===")
    print(df_net.to_string(index=False))
    print("\n==> Da luu fig3_network_1000_rounds.png, fig4_dao_delegation_graph.png va network_sim_summary.csv!")