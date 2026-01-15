# metrics.py - 结构密度计算模块
"""
核心：从状态轨迹计算涌现的"市场结构"
1. K-means 发现协议/吸引子
2. 转移矩阵 + 熵度量复杂度
3. 在线计算，O(1) 更新
"""

import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, deque
from typing import List

class StructureMetrics:
    def __init__(self, window_size=5000, n_clusters=5, cluster_update_interval=500, n_init=3):
        """
        优化版本：降低聚类频率和初始化次数
        
        Args:
            window_size: 轨迹窗口大小
            n_clusters: 聚类数
            cluster_update_interval: 聚类更新间隔（ticks），从100改为500
            n_init: K-means初始化次数，从10改为3
        """
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.cluster_update_interval = cluster_update_interval
        self.n_init = n_init
        self.trajectory = deque(maxlen=window_size)
        self.cluster_assignments = deque(maxlen=window_size)
        self.last_cluster_update = 0
        self.cached_kmeans = None  # 缓存K-means模型
        
    def update_trajectory(self, state_features: np.ndarray):
        """更新轨迹，state_features = [price,vol,liq,imbalance]"""
        self.trajectory.append(state_features)
        
        # 优化：降低聚类频率（从每100改为每500 tick）
        # 并且只在有足够数据时才聚类
        should_update = (
            len(self.trajectory) >= 200 and 
            (len(self.trajectory) - self.last_cluster_update) >= self.cluster_update_interval
        )
        
        if should_update:
            trajectory_array = np.array(self.trajectory)
            
            # 优化：减少n_init（从10改为3），使用缓存模型初始化
            if self.cached_kmeans is not None and hasattr(self.cached_kmeans, 'cluster_centers_'):
                # 使用上次的聚类中心作为初始化（warm start）
                init_centers = self.cached_kmeans.cluster_centers_
                kmeans = KMeans(
                    n_clusters=self.n_clusters, 
                    random_state=42, 
                    n_init=1,  # warm start时只需要1次
                    init=init_centers,
                    max_iter=100  # 减少最大迭代次数
                )
            else:
                # 首次聚类或缓存失效
                kmeans = KMeans(
                    n_clusters=self.n_clusters, 
                    random_state=42, 
                    n_init=self.n_init,  # 从10降到3
                    max_iter=100
                )
            
            clusters = kmeans.fit_predict(trajectory_array)
            
            # 缓存模型
            self.cached_kmeans = kmeans
            
            # 更新聚类分配（只更新新增的部分）
            new_count = len(self.trajectory) - self.last_cluster_update
            new_assignments = clusters[-new_count:]
            for assignment in new_assignments:
                self.cluster_assignments.append(assignment)
            
            self.last_cluster_update = len(self.trajectory)
            
    def compute_complexity(self) -> float:
        """结构密度 = 协议数 + 转移熵 + 驻留均匀度"""
        if len(self.cluster_assignments) < 200:
            return 0.3  # 默认值
            
        clusters = np.array(self.cluster_assignments)
        
        # 1. 有效协议数 (活跃簇)
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        active_protocols = np.sum(counts > len(clusters) * 0.02)  # >2%活跃
        protocol_score = active_protocols / self.n_clusters
        
        # 2. 转移熵
        trans_count = defaultdict(int)
        for i in range(len(clusters)-1):
            trans_count[(clusters[i], clusters[i+1])] += 1
            
        trans_matrix = np.zeros((self.n_clusters, self.n_clusters))
        for (i,j), cnt in trans_count.items():
            if i < self.n_clusters and j < self.n_clusters:
                trans_matrix[i,j] = cnt
            
        # 行归一化
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_prob = np.divide(trans_matrix, row_sums, 
                              out=np.zeros_like(trans_matrix), 
                              where=row_sums!=0)
        
        # 熵 (忽略零行)
        entropy = -np.sum(trans_prob * np.log2(trans_prob + 1e-8), axis=1)
        entropy = entropy[row_sums.flatten() > 0]
        transfer_entropy = np.mean(entropy) / np.log2(self.n_clusters) if len(entropy) > 0 else 0
        
        # 3. 驻留均匀度
        stay_dist = counts / len(clusters)
        uniformity = 1.0 - np.max(stay_dist)
        
        # 综合复杂度
        complexity = (
            0.4 * protocol_score +
            0.4 * transfer_entropy + 
            0.2 * uniformity
        )
        
        return np.clip(complexity, 0.0, 1.0)
    
    def get_cluster_info(self) -> dict:
        """获取聚类信息（用于调试）"""
        if len(self.cluster_assignments) < 200:
            return {'n_clusters': 0, 'active_protocols': 0}
        
        clusters = np.array(self.cluster_assignments)
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        active_protocols = np.sum(counts > len(clusters) * 0.02)
        
        return {
            'n_clusters': len(unique_clusters),
            'active_protocols': active_protocols,
            'cluster_distribution': dict(zip(unique_clusters, counts))
        }

# 测试
if __name__ == "__main__":
    metrics = StructureMetrics()
    
    # 合成轨迹测试
    np.random.seed(42)
    for i in range(10000):
        state = np.random.rand(4)  # [price,vol,liq,imbalance]
        metrics.update_trajectory(state)
        
        if i % 2000 == 0:
            cpx = metrics.compute_complexity()
            info = metrics.get_cluster_info()
            print(f"t={i:5d}, complexity={cpx:.3f}, "
                  f"active_protocols={info['active_protocols']}/{info['n_clusters']}")
    
    print("\n结构密度计算测试通过！")
