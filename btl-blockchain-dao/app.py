import time
import os
import requests
import numpy as np
import pandas as pd
import networkx as nx
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import streamlit as st

st.set_page_config(
    page_title="Mô phỏng Quản trị DAO - Liquid Democracy",
    page_icon="🏛️",
    layout="wide"
)

SNAPSHOT_GRAPHQL_URL = "https://hub.snapshot.org/graphql"

# ==============================================================================
# CÁC HÀM XỬ LÝ DỮ LIỆU & TÍNH TOÁN LÕI
# ==============================================================================
def gini_coefficient(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[arr > 0]
    if len(arr) == 0 or np.sum(arr) == 0:
        return 0.0
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * arr) / (n * np.sum(arr)))

@st.cache_data(show_spinner=False)
def fetch_live_snapshot_data(space: str, limit: int = 25):
    q_prop = """
    query Proposals($space: String!, $first: Int!) {
      proposals(first: $first, where: { space_in: [$space], state: "closed" }, orderBy: "created", orderDirection: desc) {
        id title created votes scores_total
      }
    }
    """
    r1 = requests.post(
        SNAPSHOT_GRAPHQL_URL,
        json={"query": q_prop, "variables": {"space": space, "first": limit}},
        timeout=25
    )
    r1.raise_for_status()
    proposals = r1.json()["data"]["proposals"]
    if not proposals:
        raise ValueError(f"Không tìm thấy dữ liệu đề xuất cho DAO '{space}'.")
    df_prop = pd.DataFrame(proposals)
    df_prop["created_date"] = pd.to_datetime(df_prop["created"], unit="s")
    prop_ids = df_prop["id"].tolist()

    q_votes = """
    query Votes($pids: [String!]!, $first: Int!, $skip: Int!) {
      votes(first: $first, skip: $skip, where: { proposal_in: $pids }, orderBy: "vp", orderDirection: desc) {
        voter vp proposal { id }
      }
    }
    """
    all_votes = []
    for skip in range(0, 5000, 1000):
        r2 = requests.post(
            SNAPSHOT_GRAPHQL_URL,
            json={"query": q_votes, "variables": {"pids": prop_ids, "first": 1000, "skip": skip}},
            timeout=25
        )
        r2.raise_for_status()
        batch = r2.json()["data"]["votes"]
        if not batch:
            break
        for v in batch:
            all_votes.append({
                "proposal_id": v["proposal"]["id"],
                "voter": v["voter"].lower(),
                "vp": float(v["vp"])
            })
        if len(batch) < 1000:
            break
        time.sleep(0.2)

    df_votes = pd.DataFrame(all_votes)
    df_voters = (
        df_votes.groupby("voter", as_index=False)
        .agg(max_vp=("vp", "max"), avg_vp=("vp", "mean"), proposals_voted=("proposal_id", "nunique"))
        .sort_values("max_vp", ascending=False)
    )
    df_voters["participation_rate"] = df_voters["proposals_voted"] / len(df_prop)
    return df_voters, df_prop

@st.cache_data(show_spinner=False)
def load_initial_data():
    for v_path, p_path in [
        ("data/real_dao_voters.csv", "data/real_dao_proposals.csv"),
        ("real_dao_voters.csv", "real_dao_proposals.csv")
    ]:
        if os.path.exists(v_path):
            df_voters = pd.read_csv(v_path)
            df_proposals = pd.read_csv(p_path) if os.path.exists(p_path) else pd.DataFrame()
            return df_voters, df_proposals, f"ENS DAO (ens.eth - Dữ liệu thật {len(df_voters)} ví)"

    try:
        df_voters, df_proposals = fetch_live_snapshot_data("ens.eth", limit=25)
        return df_voters, df_proposals, f"ENS DAO (ens.eth - Trực tiếp Snapshot API, {len(df_voters)} ví)"
    except Exception:
        rng = np.random.default_rng(42)
        vp = (rng.pareto(a=1.35, size=300) + 1.0) * 1000.0
        part = rng.uniform(0.04, 0.85, size=300)
        df_voters = pd.DataFrame({
            "voter": [f"0x{i:040x}" for i in range(300)],
            "max_vp": np.sort(vp)[::-1],
            "avg_vp": np.sort(vp)[::-1],
            "proposals_voted": np.maximum(1, (part * 25).astype(int)),
            "participation_rate": part
        })
        return df_voters, pd.DataFrame(), "Dữ liệu mô phỏng Pareto dự phòng (Mất kết nối API)"

# --- MÔ HÌNH ODE & ESS ---
def payoffs(xA, xD, xC, p):
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
    return [xA * (pi_A - pi_bar), xD * (pi_D - pi_bar), xC * (pi_C - pi_bar)]

def check_ess_stability(x_eq, p, eps=1e-6):
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
    is_ess = float(np.max(np.real(eigvals))) < 0.0
    return eigvals, is_ess

# --- MÔ PHỎNG MẠNG LƯỚI NETWORKX ---
@st.cache_data(show_spinner=False)
def run_network_sim_cached(
    vp_tuple, part_tuple, rounds, c_v, c_d, c_c, alpha,
    delegation_cap, use_quadratic, beta=0.6, mu=0.005, seed=42
):
    rng = np.random.default_rng(seed)
    raw_vp = np.array(vp_tuple, dtype=float)
    participation = np.array(part_tuple, dtype=float)
    N = len(raw_vp)

    eff_base_vp = np.sqrt(raw_vp) if use_quadratic else raw_vp.copy()
    norm_vp = eff_base_vp / np.sum(eff_base_vp)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    weights = np.power(norm_vp + 1e-6, 0.45)
    weights /= np.sum(weights)
    for u in range(N):
        targets = rng.choice(N, size=min(4, N - 1), replace=False, p=weights)
        for v in targets:
            if u != v:
                G.add_edge(u, int(v))

    neighbors_list = [list(G.neighbors(u)) for u in range(N)]
    strategies = np.where(participation >= 0.20, 0, 1)
    n_collude_init = max(1, int(0.05 * N))
    pool_idx = np.arange(min(5, N - 1), min(80, N))
    if len(pool_idx) >= n_collude_init:
        collude_idx = rng.choice(pool_idx, size=n_collude_init, replace=False)
        strategies[collude_idx] = 2

    hist_xA, hist_xD, hist_xC = [], [], []
    hist_wC, hist_gini, hist_wins = [], [], []
    snapshot = {}
    R, L, B = 10.0, 15.0, 8.0

    for r in range(rounds):
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

        if delegation_cap is not None and delegation_cap > 0:
            eff_power = np.minimum(eff_power, delegation_cap)

        w_A = float(np.sum(eff_power[strategies == 0]))
        w_C = float(np.sum(eff_power[strategies == 2]))
        total_cast = w_A + w_C
        share_wC = (w_C / total_cast) if total_cast > 0 else 0.0
        collude_win = 1.0 if w_C > w_A else 0.0

        num_C = max(1, int(np.sum(strategies == 2)))
        reward_C = (B * (N / 10.0)) / num_C
        base_payoff = (1.0 - collude_win) * R - collude_win * L

        payoffs_arr = np.full(N, base_payoff, dtype=float)
        payoffs_arr[strategies == 0] -= c_v
        payoffs_arr[strategies == 1] -= c_d
        payoffs_arr[strategies == 2] = collude_win * (reward_C - L) + (1.0 - collude_win) * R - c_c

        new_strategies = strategies.copy()
        for u in range(N):
            if rng.random() < mu:
                new_strategies[u] = int(rng.integers(0, 3))
                continue
            nbrs = neighbors_list[u]
            if nbrs:
                v = int(rng.choice(nbrs))
                diff = np.clip(payoffs_arr[v] - payoffs_arr[u], -30.0, 30.0)
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
            snapshot = {"strategies": strategies.copy(), "eff_power": eff_power.copy()}

    return G, {
        "xA": np.array(hist_xA),
        "xD": np.array(hist_xD),
        "xC": np.array(hist_xC),
        "wC": np.array(hist_wC),
        "gini": np.array(hist_gini),
        "wins": np.array(hist_wins),
        "snapshot": snapshot
    }

# ==============================================================================
# KHỞI TẠO TRẠNG THÁI & CALLBACK ĐỒNG BỘ KỊCH BẢN
# ==============================================================================
if "df_voters" not in st.session_state:
    with st.spinner("Đang nạp dữ liệu quản trị DAO..."):
        df_v, df_p, src_name = load_initial_data()
        st.session_state["df_voters"] = df_v
        st.session_state["df_proposals"] = df_p
        st.session_state["data_source"] = src_name

default_sliders = {
    "preset_box": "KB1: Hiện trạng (Phí vote cao)",
    "c_v": 1.50,
    "c_d": 0.10,
    "c_c": 1.00,
    "alpha": 0.45,
    "cap_pct": 0,
    "use_quadratic": False
}
for k_state, v_state in default_sliders.items():
    if k_state not in st.session_state:
        st.session_state[k_state] = v_state

def on_preset_change():
    choice = st.session_state["preset_box"]
    if choice == "KB1: Hiện trạng (Phí vote cao)":
        st.session_state.update({"c_v": 1.50, "c_d": 0.10, "c_c": 1.00, "alpha": 0.45, "cap_pct": 0, "use_quadratic": False})
    elif choice == "KB2: Cá mập thao túng ủy quyền":
        st.session_state.update({"c_v": 1.50, "c_d": 0.10, "c_c": 0.80, "alpha": 0.65, "cap_pct": 0, "use_quadratic": False})
    elif choice == "KB3: Hoàn phí Gas + Trần ủy quyền 8%":
        st.session_state.update({"c_v": 0.15, "c_d": 0.10, "c_c": 1.20, "alpha": 0.20, "cap_pct": 8, "use_quadratic": False})

# ==============================================================================
# GIAO DIỆN CHÍNH STREAMLIT
# ==============================================================================
st.title("🏛️ Mô phỏng Động lực học Tiến hóa & Thông đồng trong Quản trị DAO")
st.caption("Đề tài 2 • Mô hình Replicator Dynamics (ESS), Mạng lưới Ủy quyền lỏng (NetworkX) & Đối chiếu Smart Contract")

# --- SIDEBAR: ĐIỀU CHỈNH THAM SỐ TRỰC TIẾP ---
with st.sidebar:
    st.header("⚙️ Bảng điều khiển Tham số")

    st.selectbox(
        "Chọn nhanh kịch bản mẫu:",
        [
            "KB1: Hiện trạng (Phí vote cao)",
            "KB2: Cá mập thao túng ủy quyền",
            "KB3: Hoàn phí Gas + Trần ủy quyền 8%",
            "Tùy chỉnh thủ công"
        ],
        key="preset_box",
        on_change=on_preset_change
    )

    st.subheader("1. Chi phí & Lợi ích (Payoff)")
    c_v = st.slider(r"Chi phí tự bỏ phiếu ($c_v$)", 0.05, 3.00, step=0.05, key="c_v")
    c_d = st.slider(r"Chi phí ủy quyền ($c_d$)", 0.00, 1.00, step=0.05, key="c_d")
    c_c = st.slider(r"Chi phí phối hợp thông đồng ($c_c$)", 0.10, 3.00, step=0.10, key="c_c")
    alpha = st.slider(r"Hệ số lôi kéo ủy quyền ($\alpha$)", 0.00, 1.00, step=0.05, key="alpha")

    with st.expander(r"Tham số vĩ mô DAO ($R, L, B, k$)"):
        R = st.number_input(r"Lợi ích quản trị minh bạch ($R$)", value=10.0, step=1.0)
        L = st.number_input(r"Thiệt hại khi quỹ bị rút ($L$)", value=15.0, step=1.0)
        B = st.number_input(r"Ngân sách trục lợi chia nhóm C ($B$)", value=8.0, step=1.0)
        k_sens = st.number_input(r"Độ nhạy hàm thắng cử Sigmoid ($k$)", value=8.0, step=1.0)

    st.subheader("2. Cơ chế phòng vệ Mạng lưới")
    rounds = st.slider("Số vòng mô phỏng NetworkX", 200, 1000, 1000, 100)
    cap_pct = st.slider("Trần ủy quyền tối đa (% tổng quyền)", 0, 30, step=1, key="cap_pct", help="0% = Không giới hạn trần ủy quyền")
    use_quadratic = st.checkbox("Áp dụng Quadratic Voting (Căn bậc hai VP)", key="use_quadratic")

# --- TABS NỘI DUNG ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Dữ liệu DAO Thực tế",
    "📐 2. Mô hình ODE & Điểm cân bằng ESS",
    "🕸️ 3. Mô phỏng Mạng lưới (NetworkX)",
    "⛓️ 4. Smart Contract & Chi phí Gas"
])

# ==============================================================================
# TAB 1: DỮ LIỆU THỰC TẾ
# ==============================================================================
with tab1:
    st.subheader(f"Nguồn dữ liệu đang nạp: {st.session_state['data_source']}")

    if "fetch_msg" in st.session_state:
        st.success(st.session_state.pop("fetch_msg"))

    col_fetch1, col_fetch2 = st.columns([2, 1])
    with col_fetch1:
        dao_input = st.text_input(
            "Kéo trực tiếp dữ liệu DAO khác từ Snapshot API (ví dụ: ens.eth, arbitrumfoundation.eth, opcollective.eth):",
            value="ens.eth"
        )
    with col_fetch2:
        st.write("")
        st.write("")
        if st.button("🔄 Tải dữ liệu mới từ Snapshot"):
            with st.spinner(f"Đang gọi GraphQL API của {dao_input}..."):
                try:
                    df_v_new, df_p_new = fetch_live_snapshot_data(dao_input.strip(), limit=25)
                    st.session_state["df_voters"] = df_v_new
                    st.session_state["df_proposals"] = df_p_new
                    st.session_state["data_source"] = f"{dao_input.strip()} (Trực tiếp Snapshot API, {len(df_v_new)} ví)"
                    st.session_state["fetch_msg"] = f"Đã tải thành công {len(df_v_new)} ví cử tri từ {dao_input.strip()}!"
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi tải API: {e}")

    df_voters = st.session_state["df_voters"]
    total_vp = df_voters["max_vp"].sum()
    top5_share = df_voters.head(5)["max_vp"].sum() / total_vp
    gini_val = gini_coefficient(df_voters["max_vp"].values)
    avg_part = df_voters["participation_rate"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng số ví duy nhất", f"{len(df_voters):,}")
    m2.metric("Tỷ lệ tham gia TB", f"{avg_part:.2%}")
    m3.metric("Hệ số bất bình đẳng Gini", f"{gini_val:.4f}")
    m4.metric(
        "Quyền lực Top 5 đại biểu",
        f"{top5_share:.2%}",
        delta="Vượt ngưỡng 50%!" if top5_share > 0.5 else "An toàn",
        delta_color="inverse"
    )

    c_left, c_right = st.columns(2)
    with c_left:
        fig_top, ax_top = plt.subplots(figsize=(6.5, 4))
        top12 = df_voters.head(12).copy()
        labels_addr = [f"{a[:6]}...{a[-4:]}" for a in top12["voter"]]
        shares_pct = (top12["max_vp"] / total_vp) * 100
        ax_top.barh(labels_addr[::-1], shares_pct[::-1], color="#1f77b4")
        ax_top.set_xlabel("Tỷ trọng quyền biểu quyết (%)")
        ax_top.set_title("Top 12 Đại biểu nắm quyền biểu quyết lớn nhất", fontweight="bold")
        ax_top.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_top)
        plt.close(fig_top)

    with c_right:
        fig_lorenz, ax_lorenz = plt.subplots(figsize=(6.5, 4))
        sorted_vp = np.sort(df_voters["max_vp"].values)
        cum_vp = np.cumsum(sorted_vp) / np.sum(sorted_vp)
        cum_pop = np.arange(1, len(sorted_vp) + 1) / len(sorted_vp)
        ax_lorenz.plot(cum_pop * 100, cum_vp * 100, color="#d62728", lw=2.2, label=f"Đường cong Lorenz (Gini = {gini_val:.3f})")
        ax_lorenz.plot([0, 100], [0, 100], "k--", lw=1.2, label="Bình đẳng tuyệt đối")
        ax_lorenz.fill_between(cum_pop * 100, cum_vp * 100, cum_pop * 100, color="#d62728", alpha=0.12)
        ax_lorenz.set_xlabel("Tỷ lệ tích lũy số ví (%)")
        ax_lorenz.set_ylabel("Tỷ lệ tích lũy quyền biểu quyết (%)")
        ax_lorenz.set_title("Đường cong Lorenz phân phối quyền lực DAO", fontweight="bold")
        ax_lorenz.legend(loc="upper left")
        ax_lorenz.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_lorenz)
        plt.close(fig_lorenz)

    with st.expander(f"Xem bảng dữ liệu chi tiết {len(df_voters):,} ví cử tri"):
        st.dataframe(df_voters, use_container_width=True)

# ==============================================================================
# TAB 2: MÔ HÌNH GIẢI TÍCH REPLICATOR DYNAMICS & ESS
# ==============================================================================
with tab2:
    st.subheader("Hệ phương trình vi phân tiến hóa Replicator Dynamics & Kiểm định ESS")

    active_init = float(np.clip(np.mean(df_voters["participation_rate"] >= 0.20), 0.05, 0.45))
    c_init1, c_init2 = st.columns(2)
    with c_init1:
        use_real_x0 = st.checkbox(r"Hiệu chỉnh $x(0)$ tự động từ dữ liệu DAO thực tế", value=True)
    with c_init2:
        xC_0 = st.slider(r"Tỷ lệ mầm mống thông đồng ban đầu $x_C(0)$", 0.01, 0.30, 0.05, 0.01)

    if use_real_x0:
        xA_0 = active_init
    else:
        xA_0 = st.slider(r"Tỷ lệ cử tri tích cực ban đầu $x_A(0)$", 0.05, 0.80, 0.35, 0.05)
    xD_0 = max(0.0, 1.0 - xA_0 - xC_0)
    x0 = [xA_0, xD_0, xC_0]

    ode_params = {"R": R, "L": L, "B": B, "c_v": c_v, "c_d": c_d, "c_c": c_c, "alpha": alpha, "k": k_sens}
    t_eval = np.linspace(0, 30, 300)
    sol = solve_ivp(replicator_ode, [0, 30], x0, args=(ode_params,), t_eval=t_eval, method="RK45")

    x_end = np.clip(sol.y[:, -1], 0.0, 1.0)
    x_end /= np.sum(x_end)
    _, _, _, p_win_end = payoffs(x_end[0], x_end[1], x_end[2], ode_params)
    eigvals, is_ess = check_ess_stability(x_end, ode_params)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(r"Cử tri tích cực cuối ($x_A^*$)", f"{x_end[0]:.2%}")
    k2.metric(r"Ủy quyền thụ động cuối ($x_D^*$)", f"{x_end[1]:.2%}")
    k3.metric(r"Nhóm thông đồng cuối ($x_C^*$)", f"{x_end[2]:.2%}")
    k4.metric("Xác suất tấn công thắng", f"{p_win_end:.2%}")

    if is_ess:
        st.success(rf"✅ Điểm cân bằng $x^* = ({x_end[0]:.3f}, {x_end[1]:.3f}, {x_end[2]:.3f})$ là **Chiến lược ổn định tiến hóa (ESS)** vì mọi giá trị riêng Jacobian đều có phần thực âm: $\lambda = [{eigvals[0]:.3f}, {eigvals[1]:.3f}]$.")
    else:
        st.warning(rf"⚠️ Hệ đang dịch chuyển (chưa hội tụ hẳn tại $t=30$): $\lambda = [{eigvals[0]:.3f}, {eigvals[1]:.3f}]$.")

    col_ode1, col_ode2 = st.columns([1.3, 1])
    with col_ode1:
        fig_ode, ax_ode = plt.subplots(figsize=(7.5, 4.5))
        ax_ode.plot(sol.t, sol.y[0] * 100, label=r"Active Voters ($x_A$)", color="#1f77b4", lw=2.5)
        ax_ode.plot(sol.t, sol.y[1] * 100, label=r"Delegating Free-riders ($x_D$)", color="#ff7f0e", lw=2.5)
        ax_ode.plot(sol.t, sol.y[2] * 100, label=r"Colluding Attackers ($x_C$)", color="#d62728", lw=2.5)
        ax_ode.set_title("Quỹ đạo tiến hóa 3 nhóm chiến lược theo thời gian", fontweight="bold")
        ax_ode.set_xlabel(r"Thời gian tiến hóa ($t$)")
        ax_ode.set_ylabel("Tỷ lệ quần thể (%)")
        ax_ode.set_ylim(-2, 104)
        ax_ode.grid(True, linestyle="--", alpha=0.5)
        ax_ode.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig_ode)
        plt.close(fig_ode)

    with col_ode2:
        fig_ph, ax_ph = plt.subplots(figsize=(5.5, 4.5))
        grid = np.linspace(0.02, 0.96, 15)
        XA, XC = np.meshgrid(grid, grid)
        U, V = np.zeros_like(XA), np.zeros_like(XC)
        for i in range(XA.shape[0]):
            for j in range(XA.shape[1]):
                xa, xc = XA[i, j], XC[i, j]
                if xa + xc <= 0.98:
                    xd = 1.0 - xa - xc
                    dxa, _, dxc = replicator_ode(0, [xa, xd, xc], ode_params)
                    U[i, j], V[i, j] = dxa, dxc
                else:
                    U[i, j], V[i, j] = np.nan, np.nan
        ax_ph.quiver(XA, XC, U, V, color="#2c3e50", alpha=0.75)
        ax_ph.plot(sol.y[0], sol.y[2], color="#d62728", lw=2.2, label="Quỹ đạo hiện tại")
        ax_ph.scatter([x_end[0]], [x_end[2]], color="gold", edgecolor="black", s=90, zorder=5, label=r"Điểm hội tụ $x^*$")
        ax_ph.plot([0, 1], [1, 0], "k--", lw=1)
        ax_ph.fill_between([0, 1], [1, 0], 1, color="#f2f2f2")
        ax_ph.set_xlim(0, 1)
        ax_ph.set_ylim(0, 1)
        ax_ph.set_xlabel(r"Tỷ lệ Active Voters ($x_A$)")
        ax_ph.set_ylabel(r"Tỷ lệ Colluding ($x_C$)")
        ax_ph.set_title("Giản đồ pha (Phase Portrait)", fontweight="bold")
        ax_ph.legend(loc="upper right", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_ph)
        plt.close(fig_ph)

# ==============================================================================
# TAB 3: MÔ PHỎNG MẠNG LƯỚI 1.000 VÒNG (NETWORKX)
# ==============================================================================
with tab3:
    st.subheader(f"Mô phỏng ủy quyền bắc cầu qua {rounds} vòng trên {len(df_voters):,} ví thật")

    cap_val = (cap_pct / 100.0) if cap_pct > 0 else None
    with st.spinner("Đang chạy mô phỏng mạng lưới NetworkX..."):
        G_net, res_net = run_network_sim_cached(
            tuple(df_voters["max_vp"].tolist()),
            tuple(df_voters["participation_rate"].tolist()),
            rounds=rounds, c_v=c_v, c_d=c_d, c_c=c_c, alpha=alpha,
            delegation_cap=cap_val, use_quadratic=use_quadratic
        )

    tail_window = min(200, rounds)
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Tỷ lệ tấn công thành công", f"{np.mean(res_net['wins']):.2%}")
    n2.metric("Quyền biểu quyết phe C (TB)", f"{np.mean(res_net['wC'][-tail_window:]):.2%}")
    n3.metric("Hệ số Gini hiệu dụng (TB)", f"{np.mean(res_net['gini'][-tail_window:]):.4f}")
    n4.metric("Cử tri Active vòng cuối", f"{res_net['xA'][-1]:.2%}")

    col_net1, col_net2 = st.columns([1.25, 1])
    with col_net1:
        fig_sim, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
        r_axis = np.arange(1, rounds + 1)
        smooth_wC = pd.Series(res_net["wC"]).rolling(15, min_periods=1).mean()
        ax1.plot(r_axis, smooth_wC * 100, color="#d62728", lw=2, label="Quyền biểu quyết phe thông đồng (%)")
        ax1.axhline(50, color="black", linestyle="--", lw=1.2, label="Ngưỡng quá bán (50%)")
        ax1.set_ylabel("Quyền biểu quyết (%)")
        ax1.set_title("Diễn biến quyền lực phe thông đồng & Hệ số Gini qua các vòng", fontweight="bold")
        ax1.legend(loc="best", fontsize=8.5)
        ax1.grid(True, linestyle="--", alpha=0.5)

        smooth_gini = pd.Series(res_net["gini"]).rolling(15, min_periods=1).mean()
        ax2.plot(r_axis, smooth_gini, color="#ff7f0e", lw=2, label="Hệ số bất bình đẳng Gini hiệu dụng")
        ax2.plot(r_axis, res_net["xA"], color="#1f77b4", lw=1.8, label=r"Tỷ lệ cử tri Active ($x_A$)")
        ax2.set_xlabel(r"Vòng bỏ phiếu ($t$)")
        ax2.set_ylabel("Giá trị chỉ số")
        ax2.legend(loc="best", fontsize=8.5)
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_sim)
        plt.close(fig_sim)

    with col_net2:
        fig_g, ax_g = plt.subplots(figsize=(6, 6))
        top_n = min(60, G_net.number_of_nodes())
        subG = G_net.subgraph(list(range(top_n)))
        pos = nx.spring_layout(subG, seed=42, k=0.45)
        snap = res_net["snapshot"]
        cmap = {0: "#1f77b4", 1: "#ffbb78", 2: "#d62728"}
        n_colors = [cmap[int(snap["strategies"][u])] for u in range(top_n)]
        n_sizes = [100 + snap["eff_power"][u] * 4500 for u in range(top_n)]

        nx.draw_networkx_edges(subG, pos, ax=ax_g, alpha=0.22, edge_color="#7f8c8d")
        nx.draw_networkx_nodes(subG, pos, ax=ax_g, node_color=n_colors, node_size=n_sizes, edgecolors="black", linewidths=0.6)
        ax_g.set_title(f"Cấu trúc quyền lực Top {top_n} đại biểu (Vòng {rounds})", fontweight="bold")
        ax_g.axis("off")
        legend_el = [
            Line2D([0], [0], marker="o", color="w", label="Active (A)", markerfacecolor="#1f77b4", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Delegating (D)", markerfacecolor="#ffbb78", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Colluding (C)", markerfacecolor="#d62728", markersize=9),
        ]
        ax_g.legend(handles=legend_el, loc="lower center", ncol=3, frameon=True, fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_g)
        plt.close(fig_g)

# ==============================================================================
# TAB 4: SMART CONTRACT & PHÂN TÍCH CHI PHÍ GAS (MỤC 5)
# ==============================================================================
with tab4:
    st.subheader("Kiến trúc Hybrid (On-chain vs Off-chain) & Mô phỏng giới hạn Gas EVM")

    col_gas1, col_gas2 = st.columns(2)
    with col_gas1:
        st.markdown(f"""
        **Phân tách kiến trúc triển khai (Mục 5):**
        * **On-chain (`LiquidGovernance.sol`):** Chỉ lưu trữ trạng thái cốt lõi (số dư token, địa chỉ nhận ủy quyền, kết quả bỏ phiếu cuối cùng) để đảm bảo tính minh bạch và bất biến.
        * **Off-chain (`Python / Snapshot`):** Thực hiện giải hệ phương trình vi phân ODE và mô phỏng tiến hóa mạng lưới nhiều vòng lặp.
        
        **3 lý do kỹ thuật không chạy mô phỏng trực tiếp trên Blockchain:**
        1. **Giới hạn Block Gas Limit (30 triệu gas/block):** Duyệt đệ quy cây ủy quyền sâu hoặc lặp 1.000 vòng trên {len(df_voters):,} ví sẽ vượt trần gas và bị hoàn tác giao dịch (*Out of Gas*).
        2. **Chi phí giao dịch đắt đỏ:** Mỗi lượt gọi `delegate()` (kèm kiểm tra chu trình) tốn ~72.000 gas, khiến cử tri nhỏ lẻ nảy sinh tâm lý thờ ơ.
        3. **Tính tất định của EVM:** Máy ảo Ethereum không hỗ trợ số thực dấu phẩy động (`float`), giải tích vi phân hay chọn ngẫu nhiên nội tại (quy tắc Fermi).
        """)

    with col_gas2:
        st.markdown("**Trình ước tính chi phí Gas khi mở rộng quy mô On-chain:**")
        sim_voters = st.slider("Số lượng cử tri tham gia mạng lưới", 50, 1000, len(df_voters), 50)
        sim_rounds = st.slider("Số vòng lặp mô phỏng muốn đưa lên On-chain", 1, 50, 10, 1)

        gas_per_round = sim_voters * 72000
        total_gas = gas_per_round * sim_rounds
        block_limit = 30_000_000
        blocks_needed = total_gas / block_limit

        g1, g2 = st.columns(2)
        g1.metric("Tổng Gas tiêu tốn ước tính", f"{total_gas:,.0f} gas")
        g2.metric("Số Block Ethereum cần chiếm trọn", f"{blocks_needed:.1f} blocks")

        if total_gas > block_limit:
            st.error(f"🚨 Vượt quá giới hạn 1 Block Ethereum ({block_limit:,} gas) gấp {blocks_needed:.1f} lần! Bắt buộc xử lý Off-chain.")
        else:
            st.success("✅ Nằm trong giới hạn 1 Block Ethereum.")

    if os.path.exists("contracts/LiquidGovernance.sol"):
        with st.expander("Xem mã nguồn hợp đồng thông minh contracts/LiquidGovernance.sol"):
            with open("contracts/LiquidGovernance.sol", "r", encoding="utf-8") as f:
                st.code(f.read(), language="solidity")
