# Bài Tập Lớn: Tiền Số & Blockchain

## Đề tài: Động lực học tiến hóa của sự thờ ơ cử tri và thông đồng trong quản trị DAO sử dụng Ủy quyền lỏng (Liquid Democracy)

### Cấu trúc thư mục dự án

```text
btl-blockchain-dao/
├── contracts/
│   └── LiquidGovernance.sol         # Hợp đồng thông minh mẫu cài đặt ủy quyền lỏng & kiểm tra chu trình (Mục 5)
├── data/
│   ├── real_dao_proposals.csv       # Dữ liệu thực 25 đề xuất quản trị gần nhất của ENS DAO (ens.eth)
│   ├── real_dao_votes.csv           # Chi tiết phiếu bầu và quyền biểu quyết thực tế (Voting Power)
│   └── real_dao_voters.csv          # Hồ sơ hành vi của 447 ví cử tri/đại biểu duy nhất
├── results/
│   ├── fig1_ode_trajectories.png    # Quỹ đạo hội tụ hệ phương trình vi phân Replicator Dynamics (3 kịch bản)
│   ├── fig2_phase_portrait.png      # Giản đồ pha trường vector hội tụ trạng thái tấn công
│   ├── fig3_network_1000_rounds.png # Diễn biến mô phỏng 1.000 vòng bỏ phiếu trên 447 ví thật (NetworkX)
│   ├── fig4_dao_delegation_graph.png# Cấu trúc mạng lưới quyền lực Top 60 đại biểu (Tấn công vs Phòng vệ)
│   ├── ode_ess_summary.csv          # Bảng tổng hợp điểm cân bằng & giá trị riêng ma trận Jacobian (ESS)
│   └── network_sim_summary.csv      # Bảng tổng hợp kết quả thực nghiệm mạng lưới 1.000 vòng
├── src/
│   ├── 01_fetch_data.py             # Script gọi Snapshot GraphQL API lấy dữ liệu thực tế
│   ├── 02_ode_ess.py                # Giải hệ Replicator Dynamics & kiểm định ổn định ESS
│   └── 03_network_sim.py            # Mô phỏng ủy quyền bắc cầu 1.000 vòng trên đồ thị NetworkX
└── README.md
```
