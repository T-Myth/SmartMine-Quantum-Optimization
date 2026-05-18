# ==========================================================
# 动态范围(DR)压缩与精度量化脚本
# 功能：将 Ising 矩阵进行数值缩放 + 整数化，适配量子硬件运行
# 输入：int_Ising.csv   原始 Ising 矩阵
# 输出：DR_reduce_Ising.csv   压缩量化后的矩阵
# ==========================================================

import numpy as np
import pandas as pd
import kaiwu as kw
import math

def load_qubo_matrix(file_path):
    """Load QUBO/Ising matrix from CSV with error handling"""
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

# ===================== 加载待压缩矩阵 =====================
ising_matrix = load_qubo_matrix("QUBO.csv")

# ===================== 查看原始动态范围 DR =====================
print("原始动态范围 DR：", kw.preprocess.get_dynamic_range_metric(ising_matrix))
# ===================== 降低 DR =====================
DR_reduce_Ising = kw.preprocess.perform_precision_adaption_mutate(ising_matrix)

# ===================== 保存压缩结果 =====================
df = pd.DataFrame(ising_matrix)
df.to_csv("DR_reduce_QUBO.csv", index=False, header=False)

print("DR 压缩完成，已输出：DR_reduce_QUBO.csv")