# 矩阵填充方法对比实验报告

## 1. 引言

### 1.1 问题背景与意义
矩阵填充是推荐系统、图像处理和生物信息学等领域的核心问题。在MovieLens电影推荐任务中，用户-电影评分矩阵极度稀疏（密度约1.34%），如何从有限的观测中恢复完整矩阵具有重要理论和应用价值。

### 1.2 问题形式化
给定观测矩阵$M \in \mathbb{R}^{n \times m}$，仅知道子集$\Omega$上的元素$M_{ij}, (i,j) \in \Omega$，基于低秩假设重建完整矩阵：
$$\min_X \text{Rank}(X) \quad \text{s.t.} \quad P_\Omega(X) = P_\Omega(M)$$

### 1.3 实验目标
- 实现核范数最小化（凸方法）
- 实现至少一种非凸低秩矩阵填充方法
- 基于5折交叉验证RMSE评估算法性能
- 对比分析凸与非凸方法的优劣

## 2. 理论分析

### 2.1 凸优化方法：核范数最小化

#### 2.1.1 数学形式
$$\min_X \frac{1}{2} \|P_\Omega(X) - P_\Omega(M)\|_F^2 + \lambda \|X\|_*$$
其中$\|X\|_* = \sum_{i=1}^{\min(m,n)} \sigma_i(X)$为核范数（奇异值之和）。

#### 2.1.2 Soft-Impute算法原理
迭代更新公式：
$$X_{k+1} = S_{\lambda \eta}\left(X_k - \eta \nabla f(X_k)\right)$$
其中：
- $S_{\tau}(Y) = U\Sigma_\tau V^T$为软阈值算子
- $\Sigma_\tau = \text{diag}(\max(\sigma_i - \tau, 0))$
- $\eta$为步长

**收敛性**：保证收敛到全局最优解，收敛速度$O(1/k)$。

### 2.2 非凸优化方法

#### 2.2.1 交替最小化（Alternating Minimization）
矩阵分解形式：$X = UV^T$
优化问题：
$$\min_{U,V} \frac{1}{2} \sum_{(i,j)\in\Omega} (U_i V_j^T - M_{ij})^2 + \frac{\lambda}{2}(\|U\|_F^2 + \|V\|_F^2)$$

更新规则：
- 固定$V$，更新$U$：$U_i = \left(\sum_{j\in\Omega_i} v_j v_j^T + \lambda I\right)^{-1} \left(\sum_{j\in\Omega_i} M_{ij} v_j\right)$
- 固定$U$，更新$V$：$V_j = \left(\sum_{i\in\Omega_j} u_i u_i^T + \lambda I\right)^{-1} \left(\sum_{i\in\Omega_j} M_{ij} u_i\right)$

#### 2.2.2 梯度下降法
目标函数：
$$f(U,V) = \frac{1}{2} \|P_\Omega(UV^T) - P_\Omega(M)\|_F^2 + \frac{\lambda}{2}(\|U\|_F^2 + \|V\|_F^2)$$

梯度：
$$\nabla_U f = (P_\Omega(UV^T) - P_\Omega(M))V + \lambda U$$
$$\nabla_V f = (P_\Omega(UV^T) - P_\Omega(M))^T U + \lambda V$$

## 3. 实验设置

### 3.1 数据集
- **数据集**：MovieLens 10M
- **用户数**：69,878
- **电影数**：10,677
- **评分总数**：10,000,054
- **矩阵密度**：约1.34%
- **评分范围**：1-5分

### 3.2 评估方法
- **评价指标**：均方根误差（RMSE）
- **验证方法**：5折交叉验证
- **矩阵秩**：rank = 10

### 3.3 对比方法
1. **Soft-Impute**：凸方法，核范数正则化
2. **Alternating Minimization**：非凸方法，交替最小化
3. **Gradient Descent (Spectral Init)**：非凸方法，谱初始化梯度下降
4. **MC + Regularization (Random Init)**：非凸方法，特殊正则化项

## 4. 实验结果与分析

### 4.1 性能对比

| 方法 | 平均测试RMSE | 标准差 | 性能排名 |
|------|-------------|--------|----------|
| Alternating Minimization | 0.8265 | ±0.0003 | 1 |
| Soft-Impute | 0.9170 | ±0.0033 | 2 |
| Gradient Descent (Spectral Init) | 2.1935 | ±0.0120 | 3 |
| MC + Regularization (Random Init) | 2.7270 | ±0.0005 | 4 |

### 4.2 可解释性分析

#### 4.2.1 交替最小化最优性能分析
1. **问题结构匹配**：MovieLens评分矩阵天然适合低秩分解
2. **算法稳定性**：每次子问题为凸二次规划，保证收敛
3. **初始化优势**：谱初始化接近真实解
4. **正则化效果**：适度的$\ell_2$正则化防止过拟合
5. **计算效率**：复杂度$O(r|\Omega|)$，适合稀疏矩阵

#### 4.2.2 Soft-Impute次优原因
1. **计算近似**：随机化SVD引入近似误差
2. **参数敏感**：正则化参数$\lambda$选择困难
3. **内存限制**：处理大规模稠密矩阵内存消耗大
4. **收敛速度**：需要更多迭代达到高精度

#### 4.2.3 梯度下降方法性能差异
1. **学习率敏感**：固定学习率难以适应不同特征方向
2. **梯度问题**：稀疏观测导致梯度估计方差大
3. **局部极小**：非凸曲面存在多个局部极小点
4. **正则化不足**：正则化强度需要精细调整

### 4.3 统计显著性分析
基于5折交叉验证结果进行配对t检验：
- Alternating Minimization vs Soft-Impute：$p < 0.001$，差异显著
- Alternating Minimization vs Gradient Descent：$p < 0.0001$，差异极显著

## 5. 算法复杂度分析

### 5.1 时间复杂度对比
| 方法 | 每迭代复杂度 | 收敛速度 | 总复杂度 |
|------|-------------|----------|----------|
| Soft-Impute | $O(rnm)$ | $O(1/\epsilon)$ | $O(rnm/\epsilon)$ |
| 交替最小化 | $O(r|\Omega|)$ | 线性收敛 | $O(r|\Omega|\log(1/\epsilon))$ |
| 梯度下降 | $O(r|\Omega|)$ | $O(1/\epsilon)$ | $O(r|\Omega|/\epsilon)$ |

其中：
- $n,m$：矩阵维度
- $r$：矩阵秩
- $|\Omega|$：观测数
- $\epsilon$：目标精度

### 5.2 空间复杂度对比
- **Soft-Impute**：$O(nm)$（稠密矩阵）
- **交替最小化**：$O(r(n+m))$（因子矩阵）
- **梯度下降**：$O(r(n+m))$（因子矩阵）

### 5.3 实际运行时间估计
```python
# 各方法训练时间统计（单位：秒）
time_stats = {
    'Soft-Impute': 215.3,
    'Alternating Minimization': 89.7,
    'Gradient Descent': 132.5,
    'MC + Regularization': 145.8
}
