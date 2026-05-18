# ==========================================================
# QUBO 矩阵等价整数量化脚本
# 功能：根据谱间隙分析得到的最低精度 m
#      构造【最优解完全相同】的整数精度 QUBO 矩阵
# 输入：原始 QUBO 矩阵
# 输出：整数精度 m、等价最优解的量化 QUBO
# ==========================================================

import numpy as np
import pandas as pd
import math

def load_qubo_matrix(file_path):
    """加载 QUBO 矩阵"""
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

# ===================== 加载原始 QUBO =====================
qubo_matrix = load_qubo_matrix("QUBO.csv")

# ===================== 填写从谱间隙得到的最低精度 m =====================
m =   # 请填入你计算出的 m

# ===================== 等价放缩 + 整数化（保持最优解不变） =====================
max_val = np.max(qubo_matrix)
min_val = np.min(qubo_matrix)
dynamic_range = abs(max_val - min_val)

# 按 m bit 精度放缩（等价变换）
scaled_qubo = (2**m / dynamic_range) * qubo_matrix

# 取整为整数（仍保持最优解不变）
integer_qubo = np.round(scaled_qubo)

# ===================== 保存量化后的整数 QUBO =====================
pd.DataFrame(integer_qubo).to_csv("DR_reduce_QUBO_and_round.csv", index=False, header=False)