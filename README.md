# 基于光量子计算机的智慧矿山设备资源配置优化

## 项目简介
本项目以**智慧矿山设备资源配置**为研究对象，构建多目标混合整数线性规划（MILP）模型，将其转化为二次无约束二值优化（QUBO）模型并映射至Ising模型，采用**相干伊辛机（CIM，光量子计算机）**实现快速全局最优求解。实验表明，CIM可在**8.154ms**内获得最优可行解，显著优于传统优化算法。

## 一、研究背景
智慧矿山设备配置属于**大规模NP难组合优化问题**，传统遗传算法、禁忌搜索、分支定界存在**求解慢、易陷入局部最优**问题。相干伊辛机（CIM）作为光量子计算机，具备**室温运行、全连接、相干并行**优势，适用于工业级组合优化求解。

## 二、数学模型（MILP）
### 目标函数
$$
\max f = SuC - \sum_{k}\sum_{i} N_i^{E_k}Pu_i^{E_k} + \sum_{k}\sum_{i}\left[N_i^{E_k}T(p_i^{E_k})-(Ef_i^{E_k}+Ep_i^{E_k}+Em_i^{E_k})Y_i^{E_k}N_i^{E_k}\right]
$$

### 约束条件
1. 协同配比约束

$$
k_{ij}^{E_k,E_w} \le N_i^{E_k}C_{ij}^{E_k,E_w}
$$

3. 矿车数量上限
   
$$
\sum_{i}k_{ij}^{E_k,E_w} \le N_j^{E_w}
$$

5. 配比平衡约束
   
$$
\sum_{j}k_{ij}^{E_k,E_w} = N_i^{E_k}
$$

7. 最低设备类型约束
   
$$
\sum_{i}I_i^{E_k} \ge El_{E_k}
$$

9. 数量-指示变量约束
    
$$
N_i^{E_k} \le MI_i^{E_k},\quad I_i^{E_k} \le N_i^{E_k}
$$

11. 预算约束
    
$$
\sum_{k}\sum_{i}N_i^{E_k}Pu_i^{E_k} \le SuC
$$

## 三、QUBO模型构建
### 3.1 整数二进制表示

$$
n = \sum_{j=1}^{d}2^{j-1}B_j,\quad B_j\in\{0,1\}
$$

### 3.2 松弛变量表示

$$
slack = (k-2^{\lfloor\log_2k\rfloor}+1)slack_e + \sum_{i=0}^{\lfloor\log_2k\rfloor-1}2^islack_i
$$

### 3.3 惩罚项构建

$$
H_{pen} = P(Ax-b)^2
$$

### 3.4 完整QUBO目标

$$
\min H_{total} = -f + \sum H_{pen}
$$

### 3.5 惩罚系数理论下界

$$
P > \frac{\Delta f_{max}}{\Delta penalty_{min}}
$$

$$
\Delta f_{max} = \max_{i}\left\{\sum_{Q_{ij}\ge0}Q_{ij},\sum_{Q_{ij}\le0}Q_{ij}\right\}
$$

$$
\Delta penalty_{min} = \min_{x\in F,x_f\in\overline{F}}\left\{x_f^TQ_{pen}x_f-x^TQ_{pen}x\right\}
$$

$$
P^* = \left\lceil\frac{\Delta f_{max}}{\Delta penalty_{min}}+1\right\rceil
$$

## 四、QUBO→Ising模型映射
### 变量映射

$$
x_i = \frac{\sigma_i+1}{2},\quad \sigma_i\in\{\pm1\}
$$

### 矩阵转换

$$
J_{ij} = \frac{Q_{ij}}{4},\quad h_i = \frac{Q_{ii}}{2}+\frac{1}{4}\sum_{k\ne i}(Q_{ik}+Q_{ki})
$$

### Ising哈密顿量

$$
H = \sum_{i<j}J_{ij}\sigma_i\sigma_j + \sum_{i}h_i\sigma_i
$$

## 五、Ising矩阵精度降低方法
### 5.1 动态范围

$$
DR(Q) = \log_2\frac{\overline{D}(Q)}{\breve{D}(Q)}
$$

$$
\overline{D}(Q) = \max_{Q_{ij}\ne Q_{kl}}|Q_{ij}-Q_{kl}|
$$

$$
\breve{D}(Q) = \min_{Q_{ij}\ne Q_{kl}}|Q_{ij}-Q_{kl}|
$$

### 5.2 谱间隙

$$
\gamma_Q = x^{**T}Qx^{**}-x^{*T}Qx^{*}
$$

### 5.3 最小放缩系数

$$
\alpha^* = \frac{n_Q^2+n_Q}{4\gamma_Q}
$$

### 5.4 精度位数

$$
m = \left\lceil\log_2(\alpha^*\overline{D}(Q))\right\rceil
$$

### 5.5 矩阵缩放取整

$$
Q' = \lfloor\alpha Q\rceil
$$

### 5.6 变量拆分（超8bit）

$$
\min f(x,x')+M\sum(x_k-x'_k)^2
$$

$$
f(x,x') = \sum\left\lceil\frac{Q_{ij}}{2}\right\rceil x_ix_j+\sum\left\lfloor\frac{Q_{ij}}{2}\right\rfloor x'_ix'_j+\sum Q_{ii}x_i
$$

## 六、实验结果
### 测试场景
4类挖掘机、3类矿车协同配置，预算2400万元。

### 求解对比
- CIM（CPQC-550）：8.154ms，全局最优可行解
- 遗传算法：27.319s，最优可行解
- 分支定界：93.72s，最优可行解
- 禁忌搜索：0.07s，非可行解

### 最优配置
- 挖掘机：1型×7、2型×7、3型×2、4型×1
- 矿车：1型×7、2型×7、3型×3
- 最大收益：60908.64万元

## 七、核心创新
1. 首次将CIM应用于智慧矿山设备配置
2. 完整MILP→QUBO→Ising建模链
3. 提出惩罚系数理论下界
4. 设计Ising矩阵精度压缩方法
5. 实现毫秒级全局最优求解

## 八、文件结构
