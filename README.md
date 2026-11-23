# Bài Tập Lớn: Tiền Số & Blockchain

## Đề tài 2: Động lực học tiến hóa của sự thờ ơ cử tri và thông đồng trong quản trị DAO sử dụng Ủy quyền lỏng (Liquid Democracy)

> 🌐 **Demo Web Dashboard tương tác trực tuyến:** [https://btl-blockchain-dao-prf8kodqm7gmd3semga8tr.streamlit.app/](https://btl-blockchain-dao-prf8kodqm7gmd3semga8tr.streamlit.app/)

### 1. Cấu trúc thư mục dự án

```text
btl-blockchain-dao/
├── contracts/
│   └── LiquidGovernance.sol          # Hợp đồng thông minh cài đặt ủy quyền lỏng bắc cầu & kiểm tra chu trình
├── data/
│   ├── real_dao_proposals.csv        # Dữ liệu thực 25 đề xuất quản trị gần nhất của ENS DAO (ens.eth)
│   ├── real_dao_votes.csv            # Chi tiết từng phiếu bầu và quyền biểu quyết thực tế (Voting Power)
│   └── real_dao_voters.csv           # Hồ sơ hành vi của 447 ví cử tri/đại biểu duy nhất
├── results/
│   ├── fig1_ode_trajectories.png     # Quỹ đạo hội tụ hệ phương trình vi phân Replicator Dynamics (3 kịch bản)
│   ├── fig2_phase_portrait.png       # Giản đồ pha trường vector hội tụ trạng thái tấn công
│   ├── fig3_network_1000_rounds.png  # Diễn biến mô phỏng 1.000 vòng bỏ phiếu trên 447 ví thật (NetworkX)
│   ├── fig4_dao_delegation_graph.png # Cấu trúc mạng lưới quyền lực Top 60 đại biểu (Tấn công vs Phòng vệ)
│   ├── ode_ess_summary.csv           # Bảng tổng hợp điểm cân bằng & giá trị riêng ma trận Jacobian (ESS)
│   └── network_sim_summary.csv       # Bảng tổng hợp kết quả thực nghiệm mạng lưới 1.000 vòng
├── src/
│   ├── 01_fetch_data.py              # Script gọi Snapshot GraphQL API lấy dữ liệu thực tế
│   ├── 02_ode_ess.py                 # Giải hệ Replicator Dynamics & kiểm định ổn định ESS
│   └── 03_network_sim.py             # Mô phỏng ủy quyền bắc cầu 1.000 vòng trên đồ thị NetworkX
├── app.py                            # Ứng dụng Web Dashboard tương tác trực quan (Streamlit)
├── requirements.txt                  # Danh sách thư viện Python cần thiết để chạy & deploy ứng dụng
└── README.md
```

---

### 2. Quy trình thực hiện từ đầu đến cuối

#### Giai đoạn 1: Thu thập dữ liệu quản trị thực tế & Chạy mô phỏng (Off-chain / Python)

1. **Cài đặt môi trường & thư viện:**
   Yêu cầu Python 3.9+ cùng các thư viện tính toán khoa học và phân tích đồ thị:
   ```bash
   pip install requests pandas numpy scipy networkx matplotlib
   ```

2. **Bước 1 — Kéo dữ liệu thực từ Snapshot GraphQL API (`src/01_fetch_data.py`):**
   ```bash
   python src/01_fetch_data.py
   ```
   - **Hoạt động:** Gửi truy vấn tới endpoint công khai `https://hub.snapshot.org/graphql` để lấy dữ liệu 25 đề xuất đã đóng gần nhất của **ENS DAO (`ens.eth`)**, trích xuất toàn bộ lịch sử bỏ phiếu (`vp` - Voting Power) và tổng hợp hồ sơ **447 ví cử tri duy nhất**.
   - **Đầu ra:** Lưu 3 file `real_dao_proposals.csv`, `real_dao_votes.csv`, `real_dao_voters.csv` vào thư mục `data/`.

3. **Bước 2 — Giải hệ phương trình Replicator Dynamics & Kiểm định ESS (`src/02_ode_ess.py`):**
   ```bash
   python src/02_ode_ess.py
   ```
   - **Hoạt động:** Đọc tỷ lệ cử tri tích cực thực tế từ `real_dao_voters.csv` để hiệu chỉnh trạng thái khởi tạo **x(0) = [38,26%, 56,74%, 5,00%]**. Giải hệ phương trình vi phân bằng phương pháp Runge-Kutta (`RK45`) cho 3 kịch bản (Hiện trạng phí cao, Cá mập thao túng, Giải pháp hoàn gas), đồng thời tính ma trận Jacobian tại điểm cân bằng để xét dấu giá trị riêng (**max Re(λ) < 0**).
   - **Đầu ra:** Xuất `fig1_ode_trajectories.png`, `fig2_phase_portrait.png` và bảng `ode_ess_summary.csv` vào `results/`.

4. **Bước 3 — Mô phỏng mạng lưới ủy quyền lỏng 1.000 vòng (`src/03_network_sim.py`):**
   ```bash
   python src/03_network_sim.py
   ```
   - **Hoạt động:** Nạp trọng số quyền biểu quyết thật của 447 ví ENS DAO vào đồ thị mạng xã hội `NetworkX` (447 nút, 1.774 cạnh). Thực thi quy trình ủy quyền bắc cầu, tự động phát hiện và đóng băng chu trình ủy quyền (cycle freezing), áp dụng cơ chế **Trần ủy quyền 8% (Delegation Cap)** và cập nhật chiến lược theo quy tắc bắt chước Fermi qua 1.000 vòng bỏ phiếu liên tiếp.
   - **Đầu ra:** Xuất `fig3_network_1000_rounds.png`, `fig4_dao_delegation_graph.png` và bảng `network_sim_summary.csv` vào `results/`.

---

#### Giai đoạn 2: Triển khai Smart Contract & Đo chi phí Gas trên Remix IDE (On-chain / Solidity)

Để đối chiếu giữa kiến trúc On-chain và Off-chain, hợp đồng `contracts/LiquidGovernance.sol` được triển khai thử nghiệm trên môi trường **Remix VM** theo 5 bước sau:

1. **Biên dịch (Compile):**
   - Truy cập [remix.ethereum.org](https://remix.ethereum.org), tạo file `LiquidGovernance.sol` và dán mã nguồn từ thư mục `contracts/`.
   - Mở tab **Solidity Compiler** (biểu tượng chữ S bên trái), chọn phiên bản compiler `0.8.20+` và bấm **Compile LiquidGovernance.sol**.
2. **Triển khai (Deploy):**
   - Chuyển sang tab **Deploy & Run Transactions** (biểu tượng Ethereum), chọn môi trường **Remix VM** và bấm nút **Deploy** (mặc định `Account 1` đóng vai trò Admin).
3. **Cấp quyền biểu quyết cho 3 ví mẫu (`registerVoter`):**
   - Giữ nguyên `Account 1` ở mục *Account*, lần lượt copy địa chỉ của `Account 1`, `Account 2`, `Account 3` và gọi hàm `registerVoter(_voter, _weight)` với số lượng token tương ứng là `100`, `200` và `500`.
4. **Tạo đề xuất (`createProposal`) & Thực hiện ủy quyền bắc cầu (`delegate`):**
   - Gọi `createProposal("Cap ngan sach DAO")` bằng `Account 1` (tạo ra đề xuất có `_proposalId = 0`).
   - Dùng `Account 1` gọi `delegate(<Địa_chỉ_Account_2>)`, sau đó chuyển sang `Account 2` gọi `delegate(<Địa_chỉ_Account_3>)`. Lúc này quyền biểu quyết của Ví 1 và Ví 2 tự động cộng dồn về Ví 3 (`100 + 200 + 500 = 800`).
5. **Bỏ phiếu đại diện (`vote`) & Kiểm tra chi phí Gas:**
   - Chuyển sang `Account 3`, gọi hàm `vote(0, true)`.
   - Mở rộng chi tiết giao dịch trong khung **Terminal** phía dưới Remix để ghi nhận thông số `execution cost` (gas) cho từng thao tác ủy quyền/bỏ phiếu và kiểm tra sự kiện `Voted(..., effectivePower: 800)`.

---

#### Giai đoạn 3: Trực quan hóa tương tác với Streamlit Dashboard (`app.py`)

1. **Cài đặt đầy đủ thư viện từ `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Khởi chạy giao diện web tương tác trên máy cục bộ (Localhost):**
   ```bash
   python -m streamlit run app.py
   ```
   - **Hoạt động:** Mở bảng điều khiển trực quan tại `http://localhost:8501` gồm 4 tab chức năng:
     - *Tab 1 (Dữ liệu DAO Thực tế):* Tải trực tiếp dữ liệu live từ Snapshot API cho bất kỳ DAO nào (`ens.eth`, `arbitrumfoundation.eth`, `opcollective.eth`...), hiển thị biểu đồ Top 12 đại biểu và đường cong Lorenz.
     - *Tab 2 (Mô hình ODE & ESS):* Kéo thanh trượt thay đổi chi phí (c<sub>v</sub>, c<sub>d</sub>, c<sub>c</sub>) và hệ số lôi kéo (α) để quan sát quỹ đạo hội tụ và kiểm định ESS theo thời gian thực.
     - *Tab 3 (Mô phỏng Mạng lưới NetworkX):* Tùy chỉnh số vòng lặp, bật/tắt **Trần ủy quyền (Delegation Cap)** hoặc **Quadratic Voting** để theo dõi biến động quyền biểu quyết phe thông đồng và hệ số Gini.
     - *Tab 4 (Smart Contract & Chi phí Gas):* Ước tính chi phí Gas EVM khi mở rộng quy mô cử tri và số vòng lặp on-chain.

---

### 3. Tóm tắt kết quả thực nghiệm & Kiến trúc Hybrid

- **Thực trạng dữ liệu ENS DAO (`447 ví`):** Tỷ lệ tham gia bỏ phiếu trung bình chỉ đạt **23,04%**, hệ số bất bình đẳng quyền lực **Gini = 0,9556**, và **Top 5 đại biểu lớn nhất nắm 56,49%** tổng quyền biểu quyết (vượt ngưỡng quá bán 50%).
- **Kết quả giải tích ODE & ESS:**
  - *Kịch bản hiện trạng (c<sub>v</sub> = 1,5; α = 0,45) & Tấn công (α = 0,65):* Hệ hội tụ về điểm cân bằng ESS bị thâu tóm bởi nhóm thông đồng (x<sub>C</sub> = 100%, xác suất tấn công thành công **99,97%**).
  - *Kịch bản hoàn gas & hạn chế lôi kéo (c<sub>v</sub> = 0,15; α = 0,20):* Hệ hội tụ về trạng thái cân bằng ESS an toàn (x<sub>A</sub> = 13,08%; x<sub>D</sub> = 86,92%; x<sub>C</sub> = 0%), xác suất tấn công giảm còn **0,54%**.
- **Kết quả mô phỏng mạng lưới 1.000 vòng trên 447 ví thật:** Do phân phối token thực tế bị phân hóa "cá mập" cực đoan, giải pháp hoàn gas kết hợp **Trần ủy quyền 8%** giúp giảm tỷ lệ tấn công thành công từ **99,80%** xuống **77,20%** và giảm hệ số Gini hiệu dụng từ `0,9549` xuống `0,9207`.
- **Giải thích ranh giới On-chain vs Off-chain:** Thực nghiệm trên Remix cho thấy mỗi giao dịch `delegate` (kèm duyệt vòng lặp phát hiện chu trình) và `vote` (đệ quy tính tổng quyền lực cây ủy quyền) tiêu tốn hàng chục nghìn đơn vị Gas. Do đó, việc chạy mô phỏng tiến hóa 1.000 vòng trên đồ thị 447 nút không thể thực thi trực tiếp trên EVM (vượt *Block Gas Limit* và thiếu tính toán số thực/ngẫu nhiên), dẫn tới kiến trúc chuẩn cho DAO thực tế là **Mô phỏng & giám sát rủi ro Off-chain kết hợp Thực thi bỏ phiếu & lưu trữ trạng thái On-chain**.
