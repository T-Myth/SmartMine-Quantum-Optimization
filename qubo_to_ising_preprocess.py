# ==========================================================
# QUBO 转 Ising 矩阵 + 系数放大 + 变量拆分预处理
# 功能：将整数 QUBO 转为 Ising，并通过放大系数避免小数误差
#      最终通过变量拆分将精度降至 8bit，适配量子硬件
# 输入：DR_reduce_QUBO_and_round.csv (整数等价QUBO)
# 输出：int_Ising.csv (8bit 整数Ising，可直接上传真机)
# ==========================================================

import numpy as np
import pandas as pd
import kaiwu as kw
import math

def load_qubo_matrix(file_path):
    """加载 QUBO/Ising 矩阵，兼容不规范 CSV"""
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

# ===================== 1. 加载整数 QUBO 矩阵 =====================
QUBO_matrix = load_qubo_matrix("DR_reduce_QUBO_and_round.csv")

# ===================== 2. QUBO → Ising 转换 =====================
Ising_matrix, _ = kw.qubo.qubo_matrix_to_ising_matrix(QUBO_matrix)

# ===================== 3. 系数放大（关键！避免小数） =====================
# 乘以 8，保证 QUBO 整数精度 m → 转为 Ising 无小数
# 此时 Ising 精度 = m + 3
Ising_matrix = Ising_matrix * 8

# ===================== 4. 变量拆分 → 精度降到 8bit =====================
# 变量拆分：扩大问题规模，把高精度 Ising → 8bit 整数
Int_Ising, _ = kw.preprocess.perform_precision_adaption_split(Ising_matrix, 8)

# ===================== 5. 输出最终 8bit Ising =====================
df = pd.DataFrame(Int_Ising)
df.to_csv("int_Ising.csv", index=False, header=False)
