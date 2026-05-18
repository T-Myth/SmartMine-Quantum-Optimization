# -*- coding: utf-8 -*-
"""
智慧矿山设备资源配置优化模型构建脚本
功能：构建目标函数+全约束，生成QUBO矩阵与Ising矩阵并本地导出
适用流程：建模生成矩阵 → 精度压缩处理 → 量子平台求解 → 结果解析
"""
import numpy as np
import pandas as pd
import kaiwu as kw

# ===================== 设备基础参数定义 =====================
# 挖掘机参数：[斗容,作业效率,单位油耗,采购单价,月人工成本,月维护成本,最大保有数量]
excavator_param = np.array([
    [0.9, 190/60, 28/60, 1.00, 7000/9600, 1000/9600, 7],
    [1.2, 175/60, 30/60, 1.40, 7500/9600, 1500/9600, 15],
    [1.8, 165/60, 34/60, 2.00, 8500/9600, 2000/9600, 15],
    [2.1, 150/60, 38/60, 3.20, 9000/9600, 3000/9600, 7]
])

# 矿车参数：[单位油耗,月人工成本,月维护成本,现有保有总量]
truck_param = np.array([
    [18/60, 6000/9600, 2000/9600, 7],
    [22/60, 7000/9600, 3000/9600, 7],
    [27/60, 8000/9600, 4000/9600, 3]
])

# 设备协同配比约束矩阵
match_constraint_mat = np.array([
    [1, 0, 0],
    [2, 1, 0],
    [2, 2, 1],
    [0, 2, 1]
])

# 收益计算专用配比矩阵
profit_calc_mat = np.array([
    [1, 1, 1],
    [2, 1, 1],
    [2, 2, 1],
    [1, 2, 1]
])

# ===================== 优化决策变量初始化 =====================
# 四类挖掘机配置数量整数变量
excavator_num = [kw.qubo.Integer(f"X{i}", 0, excavator_param[i][6]) for i in range(4)]

# 挖掘机型号启用标识0-1变量
excavator_flag = [kw.qubo.Binary(f"Y{i}") for i in range(4)]

# 挖掘机向矿车分配数量变量
alloc_num = [kw.qubo.Integer(f"k{i}{j}", 0, truck_param[j][3])
             for i in range(4) for j in range(3)]
alloc_num = np.array(alloc_num).reshape(4, 3)

# 屏蔽无协同关系的分配变量
zero_area = (match_constraint_mat == 0)
alloc_num[zero_area] = 0

# ===================== 初始化QUBO模型 =====================
model = kw.qubo.QuboModel()

# ===================== 构建目标函数 收益-总成本 =====================
# 计算矿山作业整体收益
total_profit = 0
for i in range(4):
    work_eff_sum = 0
    for j in range(3):
        work_eff_sum += excavator_num[i] * excavator_param[i][1] - \
                        (excavator_num[i] * profit_calc_mat[i][j] - alloc_num[i][j]) * \
                        excavator_param[i][1] / profit_calc_mat[i][j]
    total_profit += 20 * excavator_param[i][0] * work_eff_sum

# 挖掘机整体运营成本
excavator_cost = 0
for i in range(4):
    excavator_cost += excavator_num[i] * (excavator_param[i][4] + excavator_param[i][5]) + \
                      7 * excavator_num[i] * excavator_param[i][2]

# 矿车整体运营成本
truck_cost = 0
for j in range(3):
    total_alloc = sum(alloc_num[:, j])
    truck_cost += total_alloc * ((truck_param[j][1] + truck_param[j][2]) + 7 * truck_param[j][0])

# 最大化收益转为最小化目标
target_func = -(total_profit - excavator_cost - truck_cost)
model.set_objective(target_func)

# ===================== 添加全部约束条件 =====================
# 约束1：设备采购总资金约束
purchase_cost = kw.qubo.quicksum(price * num for price, num in zip(excavator_param[:, 3], excavator_num))
slack_budget = kw.qubo.Integer("slack_budget", 0, 24)
penalty_budget = kw.qubo.get_min_penalty_for_equal_constraint(target_func, purchase_cost + slack_budget - 24)
model.add_constraint(purchase_cost + slack_budget - 24 == 0, "budget_limit", penalty=penalty_budget)

# 输出惩罚系数参考值
print("资金约束最优惩罚系数下界：", penalty_budget)
kw.qubo.details(model.make())

# 约束2：启用挖掘机型号数量下限约束
slack_type_cnt = kw.qubo.Integer("slack_type_cnt", 0, 1)
penalty_type = kw.qubo.get_min_penalty_for_equal_constraint(target_func, kw.qubo.quicksum(excavator_flag) - 3 - slack_type_cnt)
model.add_constraint(kw.qubo.quicksum(excavator_flag) - 3 - slack_type_cnt == 0, "type_num_limit", penalty=penalty_type)

# 约束3：设备数量与启用标识联动约束
slack_link = [kw.qubo.Integer(f"slack_link{i}", 0, excavator_param[i][6]) for i in range(4)]
for i in range(4):
    expr = excavator_param[i][6] * excavator_flag[i] - excavator_num[i] - slack_link[i]
    pen = kw.qubo.get_min_penalty_for_equal_constraint(target_func, expr)
    model.add_constraint(expr == 0, f"link_constraint_{i}", penalty=pen)

# 约束4：设备协同配比分配约束
slack_alloc_limit = np.array([
    kw.qubo.Integer(f"slack_alloc{i}{j}", 0, 7)
    for i in range(4) for j in range(3)
]).reshape(4, 3)
slack_alloc_limit[zero_area] = 0

for i in range(4):
    for j in range(3):
        if alloc_num[i][j] != 0:
            expr = alloc_num[i][j] + slack_alloc_limit[i][j] - match_constraint_mat[i][j] * excavator_num[i]
            pen = kw.qubo.get_min_penalty_for_equal_constraint(target_func, expr)
            model.add_constraint(expr == 0, f"match_alloc_{i}_{j}", penalty=pen)

# 约束5：各类矿车分配总量不超过现有存量
slack_truck_total = [kw.qubo.Integer(f"slack_truck{j}", 0, truck_param[j][3]) for j in range(3)]
for j in range(3):
    expr = kw.qubo.quicksum(alloc_num[:, j].tolist()) + slack_truck_total[j] - truck_param[j][3]
    pen = kw.qubo.get_min_penalty_for_equal_constraint(target_func, expr)
    model.add_constraint(expr == 0, f"truck_stock_limit_{j}", penalty=pen)

# ===================== 生成并导出矩阵文件 =====================
kw.qubo.details(model.make())
# 导出标准QUBO矩阵
qubo_result_mat = model.get_qubo_matrix(bit_width=None)
# QUBO映射转为Ising自旋矩阵
ising_result_mat, _ = kw.qubo.qubo_matrix_to_ising_matrix(qubo_result_mat)

# 本地保存矩阵文件
pd.DataFrame(qubo_result_mat).to_csv("QUBO.csv", index=False, header=False)
pd.DataFrame(ising_result_mat).to_csv("Ising.csv", index=False, header=False)

print("矩阵生成完成，已导出 QUBO.csv 与 Ising.csv")
