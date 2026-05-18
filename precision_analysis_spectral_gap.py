# ==========================================================
# QUBO矩阵谱间隙计算与最低精度分析脚本
# 功能：完全匹配精度压缩流程图，计算谱间隙γ_Q、最优缩放系数α*、最低可降整数精度m
# 输入：原始QUBO矩阵CSV文件
# 输出：谱间隙、α*、最低精度m等核心参数
# ==========================================================
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

def load_qubo_from_csv(file_path):
    """
    从CSV文件加载QUBO矩阵，转换为标准上三角形式
    参数：file_path - QUBO矩阵CSV文件路径
    返回：上三角QUBO方阵
    """
    df = pd.read_csv(file_path, header=None)
    matrix = df.to_numpy(dtype=float)
    
    # 确保矩阵是方阵
    assert matrix.shape[0] == matrix.shape[1], "QUBO矩阵必须为方阵"
    
    # 转换为QUBO标准上三角形式
    for i in range(matrix.shape[0]):
        for j in range(i):
            matrix[i][j] = 0.0
    return matrix

def compute_spectral_gap(Q):
    """
    计算QUBO问题的谱间隙γ_Q（最优解与次优解的目标值差值）
    参数：Q - 上三角QUBO矩阵
    返回：谱间隙γ_Q
    """
    n = Q.shape[0]
    model = gp.Model("QUBO_Spectral_Gap")
    # 关闭Gurobi输出日志
    model.setParam('OutputFlag', 0)
    
    # 定义二进制决策变量
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # 构建QUBO目标函数
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(i, n):
            obj += Q[i, j] * x[i] * x[j]
    model.setObjective(obj, GRB.MINIMIZE)
    
    # 设置求解参数
    model.setParam('NonConvex', 2)
    model.setParam('TimeLimit', 90)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        # 获取最优解与最优目标值y1
        x_opt = [x[i].X for i in range(n)]
        y1 = model.ObjVal
        
        # 添加约束：排除已找到的最优解，求解次优解
        xor_expr = gp.LinExpr()
        for i in range(n):
            if x_opt[i] == 0:
                xor_expr += x[i]
            else:
                xor_expr += (1 - x[i])
        model.addConstr(xor_expr >= 1, "exclude_optimal_sol")
        
        # 重新求解次优解
        model.optimize()
        if model.status == GRB.OPTIMAL:
            y2 = model.ObjVal
            # 谱间隙 = 次优解目标值 - 最优解目标值
            return y2 - y1
        else:
            raise Exception("次优解求解失败，问题可能无解或时间不足")
    else:
        raise Exception("QUBO问题原始最优解求解失败")

# ===================== 主程序 =====================
if __name__ == "__main__":
    try:
        # 1. 加载原始QUBO矩阵
        Q = load_qubo_from_csv("DR_reduce_QUBO.csv")
        n = Q.shape[0]
        print(f"✅ 成功加载QUBO矩阵，维度：{n}×{n}")
        
        # 2. 计算QUBO矩阵的谱间隙γ_Q
        gamma = compute_spectral_gap(Q)
        print(f"✅ 计算得到谱间隙 γ_Q = {gamma:.6f}")
        
        # 3. 计算最优缩放系数α*（保证最优解不丢失的最小缩放系数）
        alpha = (n**2 + n) / (4 * gamma)
        print(f"✅ 最优缩放系数 α* = {alpha:.6f}")
        
        # 4. 计算QUBO矩阵的动态范围Dmax
        max_element = np.max(Q)
        min_element = np.min(Q)
        Dmax = abs(max_element - min_element)
        print(f"✅ QUBO矩阵动态范围 Dmax = {Dmax:.6f}")
        
        # 5. 计算QUBO矩阵最低可降低的整数精度m（bit）
        m = np.log2(Dmax * alpha)
        print(f"✅ QUBO矩阵最低可降整数精度 m = {m:.2f} bit")
        
        # 6. 输出最终结论
        print("\n" + "="*60)
        print("📊 精度分析结论：")
        print(f"1. 若硬件支持m bit精度，可直接对矩阵进行缩放取整")
        print(f"2. 若硬件精度不足m bit，需进行变量拆分适配")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 运行错误: {str(e)}")
        print("\n请检查：")
        print("1. CSV文件路径是否正确")
        print("2. 矩阵是否为方阵")
        print("3. Gurobi许可证是否正常激活")
        print("4. 求解时间是否充足")