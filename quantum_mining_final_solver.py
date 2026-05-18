# ==========================================================
# 智慧矿山设备资源配置 · 量子最终求解与验证主程序
# 功能：QUBO建模 → 读取预处理Ising矩阵 → 量子真机求解 → 结果解码 → 约束验证
# 硬件平台：玻色量子 CPQC-550 相干伊辛机
# ==========================================================

import numpy as np
import pandas as pd
import kaiwu as kw
import time

def load_qubo_matrix(file_path):
    """加载 QUBO/Ising 矩阵（兼容不规范 CSV）"""
    try:
        return np.loadtxt(file_path, delimiter=',')
    except ValueError:
        with open(file_path, 'r') as f:
            data = []
            for line in f:
                cleaned_line = line.strip().replace('\n', '').replace(' ', '')
                if cleaned_line:
                    try:
                        row = [float(x) for x in cleaned_line.split(',')]
                        data.append(row)
                    except ValueError:
                        continue
            return np.array(data)

# ===================== 程序计时开始 =====================
start = time.perf_counter()

# ===================== 1. 矿山设备参数定义 =====================
# 挖掘机参数
excavator_param = np.array([
    [0.9, 190/60, 28/60, 10.0, 7000/9600, 1000/9600, 7],
    [1.2, 175/60, 30/60, 14.0, 7500/9600, 1500/9600, 15],
    [1.8, 165/60, 34/60, 20.0, 8500/9600, 2000/9600, 15],
    [2.1, 150/60, 38/60, 32.0, 9000/9600, 3000/9600, 7]
])

# 矿车参数
truck_param = np.array([
    [18/60, 6000/9600, 2000/9600, 7],
    [22/60, 7000/9600, 3000/9600, 7],
    [27/60, 8000/9600, 4000/9600, 3]
])

# 协同约束矩阵
match_matrix = np.array([
    [1, 0, 0],
    [2, 1, 0],
    [2, 2, 1],
    [0, 2, 1]
])

# 收益计算矩阵
profit_matrix = np.array([
    [1, 1, 1],
    [2, 1, 1],
    [2, 2, 1],
    [1, 2, 1]
])

# ===================== 2. 优化变量定义 =====================
# 挖掘机数量变量
X = [kw.qubo.Integer(f"X{i}", 0, excavator_param[i][6]) for i in range(4)]

# 挖掘机启用状态变量
Y = [kw.qubo.Binary(f"Y{i}") for i in range(4)]

# 矿车分配变量
k = [kw.qubo.Integer(f"k{i}{j}", 0, truck_param[j][3]) for i in range(4) for j in range(3)]
k = np.array(k).reshape(4, 3)
zero_pos = (match_matrix == 0)
k[zero_pos] = 0

# ===================== 3. QUBO 目标函数 + 约束项构建 =====================
Qubo_Model = kw.qubo.QuboModel()

# 收益
s = 0
for i in range(4):
    total_sum = 0
    for j in range(3):
        total_sum += X[i] * excavator_param[i][1] - (X[i] * profit_matrix[i][j] - k[i][j]) * excavator_param[i][1] / profit_matrix[i][j]
    s += 20 * excavator_param[i][0] * total_sum

# 挖掘机成本
w = 0
for i in range(4):
    w += 1 * X[i] * (excavator_param[i][4] + excavator_param[i][5]) + 7 * X[i] * excavator_param[i][2]

# 矿车成本
t = 0
for j in range(3):
    total_sum = 0
    for i in range(4):
        total_sum += k[i][j]
    t += total_sum * (1 * (truck_param[j][1] + truck_param[j][2]) + 7 * truck_param[j][0])

obj = -(s - w - t)

# ===================== 4. 约束惩罚项构建 =====================
# 预算约束
cost = kw.qubo.quicksum(a * b for a, b in zip(excavator_param[:, 3], X))
slack_cost = kw.qubo.Integer("slack_cost", 0, 240)
P = kw.qubo.get_min_penalty_for_equal_constraint(obj, cost + slack_cost - 240)
cost_constraint = P * 7.5 * (cost + slack_cost - 240) ** 2

# 设备类型数量约束
slack_type = kw.qubo.Integer("slack_type", 0, 1)
P = kw.qubo.get_min_penalty_for_equal_constraint(obj, kw.qubo.quicksum(Y) - 3 - slack_type)
typenum_constraint = P * (kw.qubo.quicksum(Y) - 3 - slack_type) ** 2

# 数量-启用关联约束
slack_xy = [kw.qubo.Integer(f"slack_xy{i}", 0, excavator_param[i][6]) for i in range(4)]
XY1_constraint = []
for i in range(4):
    P = kw.qubo.get_min_penalty_for_equal_constraint(obj, excavator_param[i][6] * Y[i] - X[i] - slack_xy[i])
    XY1_constraint.append(P * (excavator_param[i][6] * Y[i] - X[i] - slack_xy[i]) ** 2)

# 矿车分配约束
slack_kx = [kw.qubo.Integer(f"slack_kx{i}{j}", 0, 14) for i in range(4) for j in range(3)]
slack_kx = np.array(slack_kx).reshape(4, 3)
slack_kx[zero_pos] = 0
kX_constraint = []
for j in range(3):
    for i in range(4):
        if k[i][j] != 0:
            P = kw.qubo.get_min_penalty_for_equal_constraint(obj, k[i][j] + slack_kx[i][j] - match_matrix[i][j] * X[i])
            if i != 1:
                kX_constraint.append(0.01 * P * (k[i][j] + slack_kx[i][j] - match_matrix[i][j] * X[i]) ** 2)
            else:
                kX_constraint.append(0.2 * P * (k[i][j] + slack_kx[i][j] - match_matrix[i][j] * X[i]) ** 2)

# 矿车总量约束
slack_kt = [kw.qubo.Integer(f"slack_kt{j}", 0, truck_param[j][3]) for j in range(3)]
kT_constraint = []
for j in range(3):
    P = kw.qubo.get_min_penalty_for_equal_constraint(obj, kw.qubo.quicksum(k[:, j]) + slack_kt[j] - truck_param[j][3])
    kT_constraint.append(0.35 * P * (kw.qubo.quicksum(k[:, j]) + slack_kt[j] - truck_param[j][3]) ** 2)

# 分配一致性约束
ckx_constraint = []
for i in range(4):
    P = kw.qubo.get_min_penalty_for_equal_constraint(obj, kw.qubo.quicksum(k[i, :]) - X[i])
    ckx_constraint.append(P * (kw.qubo.quicksum(k[i, :]) - X[i]) ** 2)

# ===================== 5. 总目标函数 =====================
obj += cost_constraint
for c in ckx_constraint: obj += c
for c in kX_constraint: obj += c
for c in kT_constraint: obj += c

Qubo_Model.set_objective(obj)
qubo_matrix = Qubo_Model.get_qubo_matrix(bit_width=None)

# ===================== 6. 加载预处理完成的 Ising 矩阵 =====================
Ising_matrix = load_qubo_matrix("int_Ising.csv")

# ===================== 7. 玻色量子真机求解（已脱敏） =====================
kw.common.CheckpointManager.save_dir = '/tmp'

# 量子平台接口（请填写自己的账号信息）
optimizer = kw.cim.CIMOptimizer(
    user_id="YOUR_USER_ID",
    sdk_code="YOUR_SDK_CODE",
    task_name="quantum_mining_optimization",
    machine_name="CPQC-550"
)
solution = optimizer.solve(Ising_matrix)

# ===================== 8. 读取云平台返回的最优解 =====================
solution = np.loadtxt("DR_reduce_Ising_and_round_solution.csv")
opt = kw.sampler.optimal_sampler(Ising_matrix, np.array([solution]), 0, negtail_ff=False)
spin_best = opt[0][0] * opt[0][0, -1]
binary_best = kw.sampler.spin_to_binary(spin_best)

# ===================== 9. 结果解码 =====================
sol_dict = Qubo_Model.get_sol_dict(binary_best)
var_dict = {name: sol_dict[name] for name in sol_dict if not name.startswith('_slack')}

# 挖掘机数量解码
X_values = [
    var_dict['X0[0]'] + 2*var_dict['X0[1]'] + 4*var_dict['X0[2]'],
    var_dict['X1[0]'] + 2*var_dict['X1[1]'] + 4*var_dict['X1[2]'] + 8*var_dict['X1[3]'],
    var_dict['X2[0]'] + 2*var_dict['X2[1]'] + 4*var_dict['X2[2]'] + 8*var_dict['X2[3]'],
    var_dict['X3[0]'] + 2*var_dict['X3[1]'] + 4*var_dict['X3[2]']
]

# 矿车分配矩阵
K = np.zeros((4, 3), dtype=int)
for i in range(4):
    for j in range(3):
        if k[i][j] != 0:
            bits = [var_dict[f'k{i}{j}[{k}]'] for k in range(3) if f'k{i}{j}[{k}]' in var_dict]
            K[i, j] = sum(b * (2**idx) for idx, b in enumerate(bits))

# ===================== 10. 输出最优配置 =====================
print("=" * 60)
print("【智慧矿山量子优化最优配置】")
print(f"挖掘机配置数量: {X_values}")
print("矿车分配矩阵:")
print(K)
print("=" * 60)

# ===================== 11. 约束验证 =====================
total_cost = sum(excavator_param[i, 3] * X_values[i] for i in range(4))
print("\n约束验证结果：")
print(f"总成本 = {total_cost:.2f} (≤240) → {'通过' if total_cost <= 240 else '失败'}")

# ===================== 运行时间 =====================
end = time.perf_counter()
print(f"\n程序总运行时间: {end - start:.6f} 秒")