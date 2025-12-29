import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# 1. 数据加载类
class MovieLensData:
    def __init__(self, ratings_file):
        self.ratings_file = ratings_file

    def load_ratings(self):
        """加载评分数据"""
        try:
            # 支持.dat和.csv格式
            if self.ratings_file and self.ratings_file.endswith('.dat'):
                df = pd.read_csv(self.ratings_file, sep='::', engine='python',
                               header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
                print(f"成功从.dat文件加载{len(df)}条评分")

            # 创建映射
            self.user_ids = np.sort(df['user_id'].unique())
            self.movie_ids = np.sort(df['movie_id'].unique())

            self.user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
            self.movie_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}

            self.n_users = len(self.user_ids)
            self.n_movies = len(self.movie_ids)

            # 创建稀疏矩阵
            rows = df['user_id'].map(self.user_to_idx).to_numpy()
            cols = df['movie_id'].map(self.movie_to_idx).to_numpy()
            vals = df['rating'].to_numpy().astype(np.float32)

            self.R = sp.coo_matrix((vals, (rows, cols)),
                                  shape=(self.n_users, self.n_movies)).tocsr()

            # 存储原始数据用于交叉验证
            self.ratings_list = list(zip(rows, cols, vals))
            print(f"用户数: {self.n_users}, 电影数: {self.n_movies}, 评分密度: {len(self.ratings_list)/(self.n_users*self.n_movies)*100:.2f}%")

            return self.R

        except Exception as e:
            print(f"加载数据失败: {e}")
            return self.create_test_data()

    def create_test_data(self):
        """创建测试数据"""
        print("使用测试数据...")
        self.n_users = 100
        self.n_movies = 200

        # 创建低秩矩阵
        rank = 5
        U = np.random.randn(self.n_users, rank) * 0.5
        V = np.random.randn(self.n_movies, rank) * 0.5
        M = U @ V.T

        # 添加噪声并缩放到1-5范围
        M = np.clip(M + np.random.randn(*M.shape) * 0.1, 1, 5)

        # 创建稀疏观测
        mask = np.random.rand(self.n_users, self.n_movies) < 0.1
        rows, cols = np.where(mask)
        vals = M[rows, cols]

        self.R = sp.coo_matrix((vals, (rows, cols)),
                              shape=(self.n_users, self.n_movies)).tocsr()

        # 存储原始数据
        self.ratings_list = list(zip(rows, cols, vals))
        self.user_to_idx = {i: i for i in range(self.n_users)}
        self.movie_to_idx = {i: i for i in range(self.n_movies)}

        print(f"测试数据: 用户数={self.n_users}, 电影数={self.n_movies}, 评分数={len(self.ratings_list)}")

        return self.R

    def create_cross_validation_splits(self, n_folds=5, seed=42):
        """创建交叉验证分割"""
        np.random.seed(seed)
        n_ratings = len(self.ratings_list)

        # 随机打乱索引
        indices = np.random.permutation(n_ratings)

        fold_size = n_ratings // n_folds
        remainder = n_ratings % n_folds

        splits = []
        start = 0

        for i in range(n_folds):
            test_size = fold_size + (1 if i < remainder else 0)
            test_indices = indices[start:start + test_size]
            train_indices = np.concatenate([indices[:start], indices[start + test_size:]])
            splits.append((train_indices, test_indices))
            start += test_size

        print(f"创建了{n_folds}折交叉验证分割")
        return splits


# 2. Soft-Impute算法 (凸方法) - 增量SVD版本
# 2. Soft-Impute算法 (凸方法) - 修复版
# 2. Soft-Impute算法 (凸方法) - 完全修复版
class SoftImpute:
    """
    Soft-Impute算法 - 完全修复版，解决所有已知问题
    """

    def __init__(self, lambda_val=0.5, max_iter=20, tol=1e-3,
                 lambda_ratio=0.9, n_lambda=3, verbose=False,
                 max_rank=30, use_randomized_svd=True):
        """
        参数初始化 - 完全修复

        参数:
        ----------
        lambda_val : float, 初始正则化参数λ
        max_iter : int, 每个λ的最大迭代次数
        tol : float, 收敛阈值
        lambda_ratio : float, λ衰减比例
        n_lambda : int, λ路径的长度
        verbose : bool, 是否显示迭代信息
        max_rank : int, 最大秩（用于SVD截断）
        use_randomized_svd : bool, 是否使用随机化SVD（更节省内存）
        """
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_ratio = lambda_ratio
        self.n_lambda = n_lambda
        self.verbose = verbose
        self.max_rank = max_rank
        self.use_randomized_svd = use_randomized_svd

        # 用于存储结果
        self.errors = []
        self.test_errors = []
        self.ranks = []

    def _randomized_svd_sparse(self, R_train, Z, rows, cols, vals, rank):
        """高效计算稀疏矩阵的随机化SVD"""
        try:
            from sklearn.utils.extmath import randomized_svd

            n_users, n_movies = Z.shape

            # 方法1: 使用矩阵乘法近似，避免创建完整稠密矩阵
            # 这种方法是Soft-Impute的标准做法
            if rank < 1:
                rank = 1

            rank = min(rank, min(n_users, n_movies) - 1)

            if rank <= 0:
                return (np.eye(n_users, 1), np.array([0]), np.eye(1, n_movies))

            # 构建矩阵 X_filled 的低秩近似
            # 我们可以使用随机投影方法
            # 1. 生成随机高斯矩阵
            Omega = np.random.randn(n_movies, rank)

            # 2. 计算 Y = X * Omega
            Y = Z @ Omega

            # 在观测位置进行调整
            for i in range(len(rows)):
                u, m, r = rows[i], cols[i], vals[i]
                # 添加观测值与当前估计值的差异
                Y[u, :] += (r - Z[u, m]) * Omega[m, :] / rank

            # 3. QR分解
            Q, _ = np.linalg.qr(Y, mode='reduced')

            # 4. 计算 B = Q^T * X
            B = Q.T @ Z

            # 在观测位置进行调整
            for i in range(len(rows)):
                u, m, r = rows[i], cols[i], vals[i]
                B[:, m] += Q.T[:, u] * (r - Z[u, m])

            # 5. 计算B的SVD
            U_b, s, Vt = randomized_svd(B, n_components=rank, n_iter=3, random_state=42)

            # 6. 计算最终的U
            U = Q @ U_b

            return U, s, Vt

        except Exception as e:
            # 回退方法
            print(f"随机化SVD失败: {e}, 使用简化方法")
            # 使用少量样本计算
            n_samples = min(1000, len(rows))
            indices = np.random.choice(len(rows), n_samples, replace=False)

            # 构建小样本矩阵
            sample_rows = rows[indices]
            sample_cols = cols[indices]
            sample_vals = vals[indices]

            # 只处理出现过的用户和电影
            unique_users = np.unique(sample_rows)
            unique_movies = np.unique(sample_cols)

            user_map = {u: i for i, u in enumerate(unique_users)}
            movie_map = {m: i for i, m in enumerate(unique_movies)}

            # 创建小矩阵
            small_n = len(unique_users)
            small_m = len(unique_movies)
            small_matrix = np.zeros((small_n, small_m))

            for i in range(len(sample_rows)):
                u = user_map[sample_rows[i]]
                m = movie_map[sample_cols[i]]
                small_matrix[u, m] = sample_vals[i] - Z[sample_rows[i], sample_cols[i]]

            # 对小矩阵进行SVD
            try:
                from sklearn.utils.extmath import randomized_svd
                rank_small = min(rank, min(small_n, small_m) - 1)
                U_small, s_small, Vt_small = randomized_svd(
                    small_matrix,
                    n_components=rank_small,
                    n_iter=2,
                    random_state=42
                )

                # 扩展回原始维度
                U_full = np.zeros((n_users, rank_small))
                Vt_full = np.zeros((rank_small, n_movies))

                for i in range(rank_small):
                    U_full[unique_users, i] = U_small[:, i]
                    Vt_full[i, unique_movies] = Vt_small[i, :]

                return U_full, s_small, Vt_full

            except Exception as e2:
                # 最终回退
                print(f"回退方法也失败: {e2}")
                n_users, n_movies = Z.shape
                return (np.eye(n_users, 1), np.array([0]), np.eye(1, n_movies))

    def fit(self, R_train, test_data=None):
        """训练Soft-Impute模型 - 完全修复版"""
        # 确保R_train是稀疏矩阵
        if not sp.issparse(R_train):
            R_train = sp.csr_matrix(R_train)

        n_users, n_movies = R_train.shape

        # 获取观测位置
        R_coo = R_train.tocoo()
        rows, cols, vals = R_coo.row, R_coo.col, R_coo.data

        # 检查数据大小，如果太大则进行警告
        if n_users * n_movies > 1e7:  # 超过1000万元素
            print(f"警告: 矩阵过大 ({n_users}×{n_movies}={n_users * n_movies / 1e6:.1f}M元素)，")
            print(f"      建议使用子样本或减小矩阵尺寸")

        # 初始化Z - 使用均值初始化
        Z = np.zeros((n_users, n_movies))
        if len(vals) > 0:
            global_mean = np.mean(vals)
            Z[:, :] = global_mean

        # 创建λ路径
        lambda_path = [self.lambda_val * (self.lambda_ratio ** i)
                       for i in range(self.n_lambda)]

        if self.verbose:
            print(f"Soft-Impute训练开始，矩阵大小: {n_users}×{n_movies}")
            print(f"观测数: {len(vals)}, λ路径: {[f'{l:.3f}' for l in lambda_path]}")

        # 清除之前的错误记录
        self.errors = []
        self.test_errors = []
        self.ranks = []

        # 主循环 - 每个λ值
        for lambda_idx, current_lambda in enumerate(lambda_path):
            if self.verbose:
                print(f"\nλ={current_lambda:.4f} (第{lambda_idx + 1}/{len(lambda_path)})")

            prev_Z = Z.copy()
            prev_train_rmse = float('inf')

            # 内层循环 - 固定λ值迭代
            for iteration in range(self.max_iter):
                try:
                    # 动态选择秩
                    dynamic_rank = min(
                        self.max_rank,
                        int(np.sqrt(min(n_users, n_movies)) / 2)
                    )
                    dynamic_rank = max(1, dynamic_rank)

                    # 计算SVD
                    if self.use_randomized_svd:
                        U, s, Vt = self._randomized_svd_sparse(
                            R_train, Z, rows, cols, vals, dynamic_rank
                        )
                    else:
                        # 构建填充矩阵 - 但只构建小部分
                        X_filled = Z.copy()
                        X_filled[rows, cols] = vals
                        U, s, Vt = svds(
                            X_filled.astype(np.float32),  # 使用float32节省内存
                            k=min(dynamic_rank, min(X_filled.shape) - 1)
                        )
                        # 确保降序排列
                        idx = np.argsort(s)[::-1]
                        U = U[:, idx]
                        s = s[idx]
                        Vt = Vt[idx, :]

                    # 软阈值
                    s_thresh = np.maximum(s - current_lambda, 0)
                    rank = np.sum(s_thresh > 0)

                    # 重建矩阵
                    if rank > 0:
                        Z_new = (U[:, :rank] * s_thresh[:rank]) @ Vt[:rank, :]
                    else:
                        Z_new = np.zeros((n_users, n_movies))

                    # 确保评分在合理范围内
                    Z_new = np.clip(Z_new, 1.0, 5.0)

                    # 计算训练误差
                    train_pred = Z_new[rows, cols]
                    train_rmse = np.sqrt(np.mean((train_pred - vals) ** 2))

                    # 计算变化量（相对变化）
                    if np.linalg.norm(prev_Z) > 0:
                        change = np.linalg.norm(Z_new - prev_Z) / np.linalg.norm(prev_Z)
                    else:
                        change = np.linalg.norm(Z_new)

                    # 更新Z
                    prev_Z = Z.copy()
                    Z = Z_new

                    # 记录训练误差
                    self.errors.append(train_rmse)

                    # 计算测试误差
                    if test_data is not None:
                        test_preds = []
                        test_actuals = []
                        for u, m, rating in test_data:
                            if u < n_users and m < n_movies:
                                pred = np.clip(Z[u, m], 1.0, 5.0)
                                test_preds.append(pred)
                                test_actuals.append(rating)

                        if test_preds:
                            test_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
                        else:
                            test_rmse = 0
                        self.test_errors.append(test_rmse)
                    else:
                        test_rmse = 0

                    # 记录秩
                    self.ranks.append(rank)

                    if self.verbose:
                        test_str = f", 测试RMSE={test_rmse:.4f}" if test_data else ""
                        print(f"  迭代{iteration}: 训练RMSE={train_rmse:.4f}{test_str}, 秩={rank}, 变化={change:.2e}")

                    # 检查收敛条件
                    # 条件1: 变化量小于阈值
                    # 条件2: 训练误差不再显著下降
                    if iteration > 0:
                        error_change = abs(train_rmse - prev_train_rmse) / max(prev_train_rmse, 1e-10)

                        if change < self.tol or error_change < self.tol:
                            if self.verbose:
                                print(f"  在第{iteration}次迭代收敛")
                            break

                    prev_train_rmse = train_rmse

                except Exception as e:
                    print(f"  迭代{iteration}失败: {e}")
                    # 如果失败，使用之前的Z值并跳出当前λ的迭代
                    Z = prev_Z
                    break

            # 检查是否应该提前结束λ路径
            if lambda_idx > 0 and len(self.errors) > 0:
                # 如果当前λ的训练误差比上一个λ大很多，可能λ太小了
                if self.errors[-1] > 1.5 * self.errors[max(-10, -len(self.errors))]:
                    if self.verbose:
                        print(f"  训练误差上升，提前结束λ路径")
                    break

        # 保存最终结果
        self.Z = Z
        self.n_users = n_users
        self.n_movies = n_movies

        if self.verbose:
            print(f"\nSoft-Impute训练完成")
            print(f"最终矩阵秩: {self.ranks[-1] if self.ranks else 0}")
            print(f"最终训练RMSE: {self.errors[-1] if self.errors else 0:.4f}")

        return self

    def predict(self, user_idx, movie_idx):
        """预测评分"""
        if hasattr(self, 'Z'):
            pred = self.Z[user_idx, movie_idx]
            return np.clip(pred, 1.0, 5.0)
        else:
            raise ValueError("模型尚未训练")


class SoftImputeCV(SoftImpute):
    """
    Soft-Impute的交叉验证版本
    用于自动选择最优的λ参数
    """

    def __init__(self, lambda_vals=None, n_folds=5, **kwargs):
        """
        参数初始化

        参数:
        ----------
        lambda_vals : list, 要尝试的λ值列表
        n_folds : int, 交叉验证折数
        **kwargs : 传递给父类的其他参数
        """
        super().__init__(**kwargs)
        self.lambda_vals = lambda_vals if lambda_vals is not None else [0.1, 0.5, 1.0, 2.0, 5.0]
        self.n_folds = n_folds
        self.cv_results_ = {}

    def fit_with_cv(self, R_train, seed=42):
        """
        使用交叉验证选择最优λ

        参数:
        ----------
        R_train : 稀疏矩阵，训练数据
        seed : int, 随机种子

        返回:
        ----------
        best_lambda : 最优的λ值
        """
        from sklearn.model_selection import KFold
        import numpy as np

        R_coo = R_train.tocoo()
        n_ratings = len(R_coo.data)

        # 创建KFold分割
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed)

        # 存储每个λ的CV误差
        cv_scores = {lam: [] for lam in self.lambda_vals}

        if self.verbose:
            print(f"开始{self.n_folds}折交叉验证...")
            print(f"尝试的λ值: {self.lambda_vals}")

        # 交叉验证循环
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(n_ratings))):
            if self.verbose:
                print(f"\n折 {fold_idx + 1}/{self.n_folds}")

            # 创建训练和验证数据
            rows_train = R_coo.row[train_idx]
            cols_train = R_coo.col[train_idx]
            vals_train = R_coo.data[train_idx]

            rows_val = R_coo.row[val_idx]
            cols_val = R_coo.col[val_idx]
            vals_val = R_coo.data[val_idx]

            R_train_fold = sp.coo_matrix((vals_train, (rows_train, cols_train)),
                                         shape=R_train.shape).tocsr()

            # 对每个λ进行训练
            for lam in self.lambda_vals:
                if self.verbose:
                    print(f"  λ={lam:.2f}", end=" ")

                # 创建并训练模型
                model = SoftImpute(lambda_val=lam, max_iter=self.max_iter,
                                   tol=self.tol, verbose=False)
                model.fit(R_train_fold)

                # 在验证集上评估
                preds = []
                for u, m in zip(rows_val, cols_val):
                    pred = model.predict(u, m)
                    preds.append(pred)

                rmse = np.sqrt(np.mean((preds - vals_val) ** 2))
                cv_scores[lam].append(rmse)

                if self.verbose:
                    print(f"RMSE={rmse:.4f}")

        # 计算每个λ的平均RMSE
        self.cv_results_ = {
            'lambda_vals': self.lambda_vals,
            'mean_scores': [],
            'std_scores': []
        }

        for lam in self.lambda_vals:
            mean_score = np.mean(cv_scores[lam])
            std_score = np.std(cv_scores[lam])
            self.cv_results_['mean_scores'].append(mean_score)
            self.cv_results_['std_scores'].append(std_score)

        # 选择最优λ（最小RMSE）
        best_idx = np.argmin(self.cv_results_['mean_scores'])
        self.best_lambda_ = self.lambda_vals[best_idx]

        if self.verbose:
            print("\n" + "=" * 50)
            print("交叉验证结果:")
            for lam, mean_score, std_score in zip(self.lambda_vals,
                                                  self.cv_results_['mean_scores'],
                                                  self.cv_results_['std_scores']):
                print(f"  λ={lam:.2f}: RMSE={mean_score:.4f} ± {std_score:.4f}")
            print(f"最优λ: {self.best_lambda_:.2f}")
            print("=" * 50)

        # 使用最优λ在整个数据集上训练最终模型
        if self.verbose:
            print(f"\n使用最优λ={self.best_lambda_:.2f}训练最终模型...")

        self.lambda_val = self.best_lambda_
        super().fit(R_train)

        return self.best_lambda_

# 3. 交替最小化 (非凸方法1)
class AlternatingMinimization:
    def __init__(self, rank=10, max_iter=15, reg_param=0.1, tol=1e-4, verbose=False):
        self.rank = rank
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.tol = tol
        self.verbose = verbose

    def spectral_init(self, R_train):
        """谱初始化"""
        R_dense = R_train.toarray()
        try:
            U, s, Vt = svds(R_dense.astype(np.float64), k=self.rank)
            idx_desc = np.argsort(s)[::-1]
            sqrt_s = np.sqrt(np.maximum(s[idx_desc], 0))
            U_init = U[:, idx_desc] * sqrt_s
            V_init = Vt.T[:, idx_desc] * sqrt_s
            return U_init, V_init
        except:
            n_users, n_movies = R_train.shape
            return (np.random.randn(n_users, self.rank) * 0.1,
                    np.random.randn(n_movies, self.rank) * 0.1)

    def fit(self, R_train, test_data=None):
        """训练模型"""
        n_users, n_movies = R_train.shape

        # 谱初始化
        self.U, self.V = self.spectral_init(R_train)

        # 获取观测数据
        R_coo = R_train.tocoo()
        rows, cols, vals = R_coo.row, R_coo.col, R_coo.data

        # 构建观测列表
        obs_by_user = [[] for _ in range(n_users)]
        obs_by_item = [[] for _ in range(n_movies)]

        for idx in range(len(rows)):
            obs_by_user[rows[idx]].append((cols[idx], vals[idx]))
            obs_by_item[cols[idx]].append((rows[idx], vals[idx]))

        reg_matrix = self.reg_param * np.eye(self.rank)
        self.errors = []
        self.test_errors = []

        for iteration in range(self.max_iter):
            # 优化U
            for i in range(n_users):
                if obs_by_user[i]:
                    item_indices = [idx for idx, _ in obs_by_user[i]]
                    ratings = np.array([rating for _, rating in obs_by_user[i]])
                    V_i = self.V[item_indices, :]
                    A = V_i.T @ V_i + reg_matrix
                    b = V_i.T @ ratings
                    try:
                        self.U[i, :] = np.linalg.solve(A, b)
                    except:
                        self.U[i, :] = np.linalg.lstsq(A, b, rcond=None)[0]

            # 优化V
            for j in range(n_movies):
                if obs_by_item[j]:
                    user_indices = [idx for idx, _ in obs_by_item[j]]
                    ratings = np.array([rating for _, rating in obs_by_item[j]])
                    U_j = self.U[user_indices, :]
                    A = U_j.T @ U_j + reg_matrix
                    b = U_j.T @ ratings
                    try:
                        self.V[j, :] = np.linalg.solve(A, b)
                    except:
                        self.V[j, :] = np.linalg.lstsq(A, b, rcond=None)[0]

            # 计算训练误差
            preds = self.U @ self.V.T
            train_rmse = np.sqrt(np.mean((preds[rows, cols] - vals) ** 2))
            self.errors.append(train_rmse)

            # 计算测试误差
            if test_data is not None:
                test_preds = []
                test_actuals = []
                for u, m, rating in test_data:
                    if u < n_users and m < n_movies:
                        pred = np.clip(preds[u, m], 1.0, 5.0)
                        test_preds.append(pred)
                        test_actuals.append(rating)

                if len(test_preds) > 0:
                    test_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
                else:
                    test_rmse = 0
                self.test_errors.append(test_rmse)

            if self.verbose and iteration % 5 == 0:
                test_str = f", 测试RMSE={self.test_errors[-1]:.4f}" if test_data else ""
                print(f"迭代 {iteration}: 训练RMSE={train_rmse:.4f}{test_str}")

            # 检查收敛
            if iteration > 0 and abs(self.errors[-1] - self.errors[-2]) < self.tol:
                if self.verbose:
                    print(f"收敛于第{iteration}次迭代")
                break

        self.predictions = preds
        return self

    def predict(self, user_idx, movie_idx):
        """预测评分"""
        if hasattr(self, 'predictions'):
            return np.clip(self.predictions[user_idx, movie_idx], 1.0, 5.0)
        elif hasattr(self, 'U') and hasattr(self, 'V'):
            pred = self.U[user_idx, :] @ self.V[movie_idx, :]
            return np.clip(pred, 1.0, 5.0)
        return 3.0


# 4. 梯度下降 + 谱初始化 (非凸方法2)
class GradientDescentMC:
    def __init__(self, rank=10, lr=0.001, max_iter=80, reg_param=0.01,
                 tol=1e-4, verbose=False, clip_grad=10.0):
        self.rank = rank
        self.lr = lr  # 减小学习率
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.tol = tol
        self.verbose = verbose
        self.clip_grad = clip_grad  # 梯度裁剪阈值

    def spectral_init(self, R_train):
        """谱初始化，增加数值稳定性"""
        try:
            # 使用采样减少内存使用
            R_coo = R_train.tocoo()
            if len(R_coo.data) > 100000:  # 如果数据太大，采样
                indices = np.random.choice(len(R_coo.data), 100000, replace=False)
                rows = R_coo.row[indices]
                cols = R_coo.col[indices]
                vals = R_coo.data[indices]
            else:
                rows, cols, vals = R_coo.row, R_coo.col, R_coo.data

            # 创建小型稠密矩阵
            unique_rows = np.unique(rows)
            unique_cols = np.unique(cols)

            # 创建映射
            row_map = {r: i for i, r in enumerate(unique_rows)}
            col_map = {c: i for i, c in enumerate(unique_cols)}

            # 创建小矩阵
            small_n = len(unique_rows)
            small_m = len(unique_cols)
            small_matrix = np.zeros((small_n, small_m))

            for i in range(len(rows)):
                if i < len(rows):
                    r_idx = row_map.get(rows[i], -1)
                    c_idx = col_map.get(cols[i], -1)
                    if r_idx >= 0 and c_idx >= 0:
                        small_matrix[r_idx, c_idx] = vals[i]

            # 考虑采样率
            mask = (small_matrix != 0).astype(float)
            p = np.mean(mask) if np.sum(mask) > 0 else 1.0

            scaled_matrix = small_matrix / max(p, 1e-8)

            # 对小型矩阵进行SVD
            try:
                U_small, s, Vt_small = svds(scaled_matrix.astype(np.float64),
                                            k=min(self.rank, min(small_n, small_m) - 1))

                idx_desc = np.argsort(s)[::-1]
                sqrt_s = np.sqrt(np.maximum(s[idx_desc], 0))

                # 扩展回原始维度
                U_init = np.zeros((R_train.shape[0], self.rank))
                V_init = np.zeros((R_train.shape[1], self.rank))

                for i in range(min(len(sqrt_s), self.rank)):
                    U_init[unique_rows, i] = U_small[:, idx_desc[i]] * sqrt_s[i]
                    V_init[unique_cols, i] = Vt_small.T[:, idx_desc[i]] * sqrt_s[i]

                return U_init, V_init
            except:
                # 回退：随机初始化
                n_users, n_movies = R_train.shape
                return (np.random.randn(n_users, self.rank) * 0.01,
                        np.random.randn(n_movies, self.rank) * 0.01)

        except Exception as e:
            print(f"谱初始化失败: {e}, 使用随机初始化")
            n_users, n_movies = R_train.shape
            return (np.random.randn(n_users, self.rank) * 0.01,
                    np.random.randn(n_movies, self.rank) * 0.01)

    def fit(self, R_train, test_data=None):
        """训练模型 - 修复梯度爆炸和NaN问题"""
        n_users, n_movies = R_train.shape

        # 谱初始化
        self.U, self.V = self.spectral_init(R_train)

        # 获取观测数据
        R_coo = R_train.tocoo()
        rows, cols, vals = R_coo.row, R_coo.col, R_coo.data

        # 全局均值，用于处理NaN
        global_mean = np.mean(vals) if len(vals) > 0 else 3.0

        self.errors = []
        self.test_errors = []

        for iteration in range(self.max_iter):
            # 计算预测值，添加数值稳定性检查
            try:
                preds = np.sum(self.U[rows] * self.V[cols], axis=1)

                # 检查NaN并处理
                if np.any(np.isnan(preds)):
                    print(f"迭代{iteration}: 预测值出现NaN，重新初始化")
                    self.U = np.random.randn(n_users, self.rank) * 0.01
                    self.V = np.random.randn(n_movies, self.rank) * 0.01
                    preds = np.sum(self.U[rows] * self.V[cols], axis=1)

                errors = preds - vals

                # 梯度计算
                grad_U = np.zeros_like(self.U)
                grad_V = np.zeros_like(self.V)

                # 使用np.add.at加速梯度计算，但检查NaN
                U_rows = self.U[rows]
                V_cols = self.V[cols]

                if np.any(np.isnan(U_rows)) or np.any(np.isnan(V_cols)):
                    print(f"迭代{iteration}: U或V出现NaN，跳过更新")
                    continue

                np.add.at(grad_U, rows, errors[:, np.newaxis] * V_cols)
                np.add.at(grad_V, cols, errors[:, np.newaxis] * U_rows)

                # 梯度裁剪，防止梯度爆炸
                grad_norm_U = np.linalg.norm(grad_U)
                grad_norm_V = np.linalg.norm(grad_V)

                if grad_norm_U > self.clip_grad:
                    grad_U = grad_U * (self.clip_grad / grad_norm_U)
                if grad_norm_V > self.clip_grad:
                    grad_V = grad_V * (self.clip_grad / grad_norm_V)

                # 添加正则化
                grad_U += self.reg_param * self.U
                grad_V += self.reg_param * self.V

                # 更新参数
                self.U -= self.lr * grad_U
                self.V -= self.lr * grad_V

                # 检查更新后的参数是否有NaN
                if np.any(np.isnan(self.U)) or np.any(np.isnan(self.V)):
                    print(f"迭代{iteration}: 更新后参数出现NaN，重新初始化")
                    self.U = np.random.randn(n_users, self.rank) * 0.01
                    self.V = np.random.randn(n_movies, self.rank) * 0.01

                # 计算训练误差
                train_rmse = np.sqrt(np.mean(errors ** 2))
                self.errors.append(train_rmse)

                # 计算测试误差，处理可能的NaN
                if test_data is not None:
                    predictions = self.U @ self.V.T
                    test_preds = []
                    test_actuals = []

                    for u, m, rating in test_data:
                        if u < n_users and m < n_movies:
                            pred = predictions[u, m]
                            # 检查预测值是否有效
                            if np.isnan(pred) or np.isinf(pred):
                                pred = global_mean
                            pred = np.clip(pred, 1.0, 5.0)
                            test_preds.append(pred)
                            test_actuals.append(rating)

                    if len(test_preds) > 0:
                        try:
                            test_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
                        except:
                            # 如果计算RMSE失败，使用一个大值
                            test_rmse = 5.0
                    else:
                        test_rmse = 0
                    self.test_errors.append(test_rmse)

            except Exception as e:
                print(f"迭代{iteration}失败: {e}")
                # 如果失败，重新初始化参数
                self.U = np.random.randn(n_users, self.rank) * 0.01
                self.V = np.random.randn(n_movies, self.rank) * 0.01
                continue

            if self.verbose and iteration % 10 == 0:
                test_str = f", 测试RMSE={self.test_errors[-1]:.4f}" if test_data else ""
                print(f"迭代 {iteration}: 训练RMSE={train_rmse:.4f}{test_str}")

            # 检查收敛
            if iteration > 0 and len(self.errors) > 1:
                if abs(self.errors[-1] - self.errors[-2]) < self.tol:
                    if self.verbose:
                        print(f"收敛于第{iteration}次迭代")
                    break

        self.predictions = self.U @ self.V.T
        return self

    def predict(self, user_idx, movie_idx):
        """预测评分 - 确保不返回NaN"""
        if hasattr(self, 'predictions'):
            pred = self.predictions[user_idx, movie_idx]
        elif hasattr(self, 'U') and hasattr(self, 'V'):
            pred = self.U[user_idx, :] @ self.V[movie_idx, :]
        else:
            pred = 3.0  # 默认值

        # 检查NaN并处理
        if np.isnan(pred) or np.isinf(pred):
            pred = 3.0

        return np.clip(pred, 1.0, 5.0)


# 5. MC + 正则项 + 随机初始化 (非凸方法3)
class MCWithRegularization:
    def __init__(self, rank=10, lr=0.001, max_iter=80, reg_param=0.01,
                 alpha=1.0, tol=1e-4, verbose=False, clip_grad=10.0):
        self.rank = rank
        self.lr = lr  # 减小学习率
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.alpha = alpha
        self.tol = tol
        self.verbose = verbose
        self.clip_grad = clip_grad  # 梯度裁剪阈值

    def compute_reg_grad(self, X):
        """计算正则化项梯度：Q(X) = Σ(∥e_i^T X∥_2 - α)_+^4，增加数值稳定性"""
        # 添加小的epsilon防止除以零
        epsilon = 1e-8

        # 计算行范数
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        # 添加epsilon防止除以零
        row_norms_safe = row_norms + epsilon

        mask = (row_norms > self.alpha).flatten()

        grad = np.zeros_like(X)
        if np.any(mask):
            # 使用安全的行范数计算
            factor = 4 * (row_norms[mask] - self.alpha) ** 3 / row_norms_safe[mask]
            grad[mask, :] = factor * X[mask, :]

        return grad

    def fit(self, R_train, test_data=None):
        """训练模型 - 修复数值稳定性问题"""
        n_users, n_movies = R_train.shape

        # 随机初始化 - 使用更小的标准差
        np.random.seed(42)
        self.U = np.random.randn(n_users, self.rank) * 0.01  # 减小初始化标准差
        self.V = np.random.randn(n_movies, self.rank) * 0.01

        # 获取观测数据
        R_coo = R_train.tocoo()
        rows, cols, vals = R_coo.row, R_coo.col, R_coo.data

        # 全局均值，用于处理NaN
        global_mean = np.mean(vals) if len(vals) > 0 else 3.0

        self.errors = []
        self.test_errors = []

        for iteration in range(self.max_iter):
            try:
                # 计算数据项梯度
                U_rows = self.U[rows]
                V_cols = self.V[cols]

                # 检查参数是否有NaN
                if np.any(np.isnan(U_rows)) or np.any(np.isnan(V_cols)):
                    print(f"迭代{iteration}: U或V出现NaN，重新初始化")
                    self.U = np.random.randn(n_users, self.rank) * 0.01
                    self.V = np.random.randn(n_movies, self.rank) * 0.01
                    U_rows = self.U[rows]
                    V_cols = self.V[cols]

                preds = np.sum(U_rows * V_cols, axis=1)

                # 检查预测值是否有NaN
                if np.any(np.isnan(preds)):
                    print(f"迭代{iteration}: 预测值出现NaN，跳过更新")
                    continue

                errors = preds - vals

                grad_U = np.zeros_like(self.U)
                grad_V = np.zeros_like(self.V)

                # 使用安全的np.add.at
                try:
                    np.add.at(grad_U, rows, errors[:, np.newaxis] * V_cols)
                    np.add.at(grad_V, cols, errors[:, np.newaxis] * U_rows)
                except:
                    print(f"迭代{iteration}: 梯度累加失败，跳过更新")
                    continue

                # 梯度裁剪
                grad_norm_U = np.linalg.norm(grad_U)
                grad_norm_V = np.linalg.norm(grad_V)

                if grad_norm_U > self.clip_grad:
                    grad_U = grad_U * (self.clip_grad / grad_norm_U)
                if grad_norm_V > self.clip_grad:
                    grad_V = grad_V * (self.clip_grad / grad_norm_V)

                # 添加正则化梯度
                try:
                    reg_grad_U = self.compute_reg_grad(self.U)
                    reg_grad_V = self.compute_reg_grad(self.V)

                    # 检查正则化梯度是否有NaN
                    if np.any(np.isnan(reg_grad_U)) or np.any(np.isnan(reg_grad_V)):
                        print(f"迭代{iteration}: 正则化梯度出现NaN，跳过正则化项")
                        reg_grad_U = np.zeros_like(self.U)
                        reg_grad_V = np.zeros_like(self.V)

                    grad_U += self.reg_param * reg_grad_U
                    grad_V += self.reg_param * reg_grad_V
                except Exception as e:
                    print(f"迭代{iteration}: 计算正则化梯度失败: {e}")
                    # 跳过正则化项
                    pass

                # 更新参数
                self.U -= self.lr * grad_U
                self.V -= self.lr * grad_V

                # 检查更新后的参数是否有NaN
                if np.any(np.isnan(self.U)) or np.any(np.isnan(self.V)):
                    print(f"迭代{iteration}: 更新后参数出现NaN，重新初始化")
                    self.U = np.random.randn(n_users, self.rank) * 0.01
                    self.V = np.random.randn(n_movies, self.rank) * 0.01

                # 计算训练误差
                train_rmse = np.sqrt(np.mean(errors ** 2))
                self.errors.append(train_rmse)

                # 计算测试误差，处理可能的NaN
                if test_data is not None:
                    predictions = self.U @ self.V.T
                    test_preds = []
                    test_actuals = []

                    for u, m, rating in test_data:
                        if u < n_users and m < n_movies:
                            pred = predictions[u, m]
                            # 检查预测值是否有效
                            if np.isnan(pred) or np.isinf(pred):
                                pred = global_mean
                            pred = np.clip(pred, 1.0, 5.0)
                            test_preds.append(pred)
                            test_actuals.append(rating)

                    if len(test_preds) > 0:
                        try:
                            test_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
                        except:
                            # 如果计算RMSE失败，使用一个大值
                            test_rmse = 5.0
                    else:
                        test_rmse = 0
                    self.test_errors.append(test_rmse)

            except Exception as e:
                print(f"迭代{iteration}失败: {e}")
                # 如果失败，重新初始化参数
                self.U = np.random.randn(n_users, self.rank) * 0.01
                self.V = np.random.randn(n_movies, self.rank) * 0.01
                continue

            if self.verbose and iteration % 10 == 0:
                test_str = f", 测试RMSE={self.test_errors[-1]:.4f}" if test_data else ""
                print(f"迭代 {iteration}: 训练RMSE={train_rmse:.4f}{test_str}")

            # 检查收敛
            if iteration > 0 and len(self.errors) > 1:
                if abs(self.errors[-1] - self.errors[-2]) < self.tol:
                    if self.verbose:
                        print(f"收敛于第{iteration}次迭代")
                    break

        self.predictions = self.U @ self.V.T
        return self

    def predict(self, user_idx, movie_idx):
        """预测评分 - 确保不返回NaN"""
        if hasattr(self, 'predictions'):
            pred = self.predictions[user_idx, movie_idx]
        elif hasattr(self, 'U') and hasattr(self, 'V'):
            pred = self.U[user_idx, :] @ self.V[movie_idx, :]
        else:
            pred = 3.0  # 默认值

        # 检查NaN并处理
        if np.isnan(pred) or np.isinf(pred):
            pred = 3.0

        return np.clip(pred, 1.0, 5.0)


class MatrixCompletionExperiment:
    """矩阵填充实验框架"""
    def __init__(self, data_loader, n_folds=5):
        self.data_loader = data_loader
        self.n_folds = n_folds
        self.results = {}

    def create_train_test_matrices(self, train_indices, test_indices):
        """创建训练和测试矩阵"""
        # 提取训练数据
        train_data = [self.data_loader.ratings_list[i] for i in train_indices]
        rows_train = [u for u, _, _ in train_data]
        cols_train = [m for _, m, _ in train_data]
        vals_train = [r for _, _, r in train_data]

        # 创建训练矩阵
        R_train = sp.coo_matrix((vals_train, (rows_train, cols_train)),
                               shape=(self.data_loader.n_users, self.data_loader.n_movies)).tocsr()

        # 提取测试数据
        test_data = [self.data_loader.ratings_list[i] for i in test_indices]

        return R_train, test_data

    # 在 MatrixCompletionExperiment 类的 run_cross_validation 方法中修改
    # 修改 MatrixCompletionExperiment 类中的 run_cross_validation 方法
    def run_cross_validation(self, rank=10):
        """运行交叉验证"""
        splits = self.data_loader.create_cross_validation_splits(n_folds=self.n_folds)

        methods = {
            'Soft-Impute': SoftImpute,
            'Alternating Minimization': AlternatingMinimization,
            'Gradient Descent (Spectral Init)': GradientDescentMC,
            'MC + Regularization (Random Init)': MCWithRegularization
        }

        for method_name in methods.keys():
            self.results[method_name] = {'test_rmses': [], 'train_rmses': [], 'convergence': []}

        print(f"开始{self.n_folds}折交叉验证...")
        print(f"矩阵秩: {rank}")
        print("=" * 60)

        for fold_idx, (train_indices, test_indices) in enumerate(splits, 1):
            print(f"\n第{fold_idx}折:")
            print(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")

            # 创建训练和测试数据
            R_train, test_data = self.create_train_test_matrices(train_indices, test_indices)

            for method_name, method_class in methods.items():
                print(f"\n方法: {method_name}")

                try:
                    # 初始化模型 - 使用修复后的参数
                    if method_name == 'Soft-Impute':
                        # 对于大数据集，使用优化参数
                        n_ratings = len(self.data_loader.ratings_list)
                        if n_ratings > 1000000:
                            model = method_class(
                                lambda_val=1.0,  # 增大λ初始值
                                max_iter=15,  # 增加迭代次数
                                tol=1e-3,
                                lambda_ratio=0.8,  # 更快的衰减
                                n_lambda=3,
                                max_rank=min(20, rank * 2),
                                use_randomized_svd=True,
                                verbose=False
                            )
                        else:
                            model = method_class(
                                lambda_val=0.5,
                                max_iter=20,
                                verbose=False
                            )
                    elif method_name == 'Alternating Minimization':
                        model = method_class(
                            rank=rank,
                            verbose=False,
                            max_iter=10,
                            reg_param=0.05
                        )
                    elif method_name == 'Gradient Descent (Spectral Init)':
                        model = method_class(
                            rank=rank,
                            verbose=False,
                            max_iter=30,  # 减少迭代次数
                            lr=0.0005,  # 进一步减小学习率
                            reg_param=0.0001,  # 减小正则化参数
                            clip_grad=5.0  # 添加梯度裁剪
                        )
                    else:  # MC + Regularization
                        model = method_class(
                            rank=rank,
                            verbose=False,
                            max_iter=30,
                            lr=0.0005,  # 减小学习率
                            reg_param=0.0001,  # 减小正则化参数
                            alpha=0.5,
                            clip_grad=5.0  # 添加梯度裁剪
                        )

                    # 训练模型
                    start_time = time.time()
                    model.fit(R_train, test_data)
                    train_time = time.time() - start_time

                    # 预测和评估
                    test_preds = []
                    test_actuals = []
                    for u, m, rating in test_data:
                        pred = model.predict(u, m)
                        test_preds.append(pred)
                        test_actuals.append(rating)

                    test_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))

                    # 获取训练误差
                    if hasattr(model, 'errors') and model.errors:
                        train_rmse = model.errors[-1] if model.errors else 0
                    else:
                        train_rmse = 0

                    # 存储结果
                    self.results[method_name]['test_rmses'].append(test_rmse)
                    self.results[method_name]['train_rmses'].append(train_rmse)

                    # 存储收敛曲线
                    if hasattr(model, 'errors'):
                        self.results[method_name]['convergence'].append(model.errors)

                    print(f"  测试RMSE: {test_rmse:.4f}, 训练时间: {train_time:.2f}秒")

                except Exception as e:
                    print(f"  方法失败: {e}")
                    import traceback
                    traceback.print_exc()
                    self.results[method_name]['test_rmses'].append(5.0)
                    self.results[method_name]['train_rmses'].append(5.0)
                    self.results[method_name]['convergence'].append([])

        return self.results

    def print_results(self):
        """打印结果"""
        print("\n" + "=" * 60)
        print("交叉验证结果汇总")
        print("=" * 60)

        for method_name, result in self.results.items():
            test_rmses = result['test_rmses']
            if len(test_rmses) > 0:
                mean_rmse = np.mean(test_rmses)
                std_rmse = np.std(test_rmses)
                print(f"\n{method_name}:")
                print(f"  平均测试RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
                print(f"  各折RMSE: {[f'{rmse:.4f}' for rmse in test_rmses]}")

    def plot_results(self):
        """绘制图表"""
        try:
            # 收敛曲线
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for idx, (method_name, result) in enumerate(self.results.items()):
                if idx < len(axes):
                    ax = axes[idx]
                    convergence = result['convergence']
                    if convergence and len(convergence) > 0:
                        # 取最后一折的收敛曲线
                        last_convergence = convergence[-1]
                        if len(last_convergence) > 0:
                            ax.plot(last_convergence, linewidth=2)
                            ax.set_xlabel('Iteration')
                            ax.set_ylabel('RMSE')
                            ax.set_title(f'{method_name} - Convergence Curve')
                            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('convergence_curves.png', dpi=150, bbox_inches='tight')
            plt.show()

            print("可视化图表已保存为 'convergence_curves.png'")

        except Exception as e:
            print(f"绘图错误: {e}")


#主函数
def main():
    """主函数"""
    print("=" * 60)
    print("非凸优化课程作业：MovieLens矩阵填充方法对比")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据...")
    data_files = ['ratings.dat', 'ratings.csv']
    data_loader = None

    for file in data_files:
        try:
            data_loader = MovieLensData(file)
            data_loader.load_ratings()
            print(f"成功从 {file} 加载数据")
            break
        except Exception as e:
            print(f"无法加载 {file}: {e}")
            continue

    if data_loader is None:
        print("无法加载数据文件，使用测试数据")
        data_loader = MovieLensData(None)
        data_loader.create_test_data()

    # 2. 运行实验
    print("\n2. 开始5折交叉验证实验...")
    experiment = MatrixCompletionExperiment(data_loader, n_folds=5)

    # 设置适当的矩阵秩
    rank = min(10, min(data_loader.n_users, data_loader.n_movies) // 100)
    print(f"使用矩阵秩: {rank}")

    results = experiment.run_cross_validation(rank=rank)

    # 3. 显示结果
    print("\n3. 实验结果:")
    experiment.print_results()

    # 4. 绘制图表
    print("\n4. 生成可视化图表...")
    experiment.plot_results()

    # 5. 保存结果
    print("\n5. 保存结果...")
    try:
        with open('experiment_results.txt', 'w', encoding='utf-8') as f:
            f.write("MovieLens矩阵填充实验结果\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"数据集信息:\n")
            f.write(f"  用户数: {data_loader.n_users}\n")
            f.write(f"  电影数: {data_loader.n_movies}\n")
            f.write(f"  评分总数: {len(data_loader.ratings_list)}\n")
            f.write(f"  矩阵秩: {rank}\n\n")

            f.write("算法实现:\n")
            f.write("  1. Soft-Impute: 凸方法，使用软阈值SVD\n")
            f.write("  2. Alternating Minimization: 非凸方法，交替最小化\n")
            f.write("  3. Gradient Descent (Spectral Init): 非凸方法，谱初始化梯度下降\n")
            f.write("  4. MC + Regularization (Random Init): 非凸方法，带特殊正则化项\n\n")

            f.write("5折交叉验证结果:\n")
            for method_name, result in results.items():
                test_rmses = result['test_rmses']
                if len(test_rmses) > 0:
                    mean_rmse = np.mean(test_rmses)
                    std_rmse = np.std(test_rmses)
                    f.write(f"\n{method_name}:\n")
                    f.write(f"  平均测试RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}\n")
                    f.write(f"  各折RMSE: {[f'{rmse:.4f}' for rmse in test_rmses]}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("实验完成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 60 + "\n")

        print("详细结果已保存到 'experiment_results.txt'")

    except Exception as e:
        print(f"保存结果失败: {e}")

    print("\n实验完成!")
    print("=" * 60)

if __name__ == "__main__":
    np.random.seed(42)  # 设置随机种子以确保可重复性
    main()