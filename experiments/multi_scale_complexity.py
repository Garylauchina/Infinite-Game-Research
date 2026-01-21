# multi_scale_complexity.py - P2: 多尺度complexity观测
"""
P2修复：结构密度/complexity的观测口径做多尺度
- n_clusters ∈ {3,5,8,13}
- window_size ∈ {1000,5000,20000}
- 输出sensitivity表
"""

import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import Dict, List

def compute_complexity_multi_scale(trajectory: np.ndarray) -> Dict[str, float]:
    """
    在多个尺度上计算complexity
    
    Args:
        trajectory: shape (T, 4) - [price_norm, volatility, liquidity, imbalance]
    
    Returns:
        Dict with keys like 'cpx_k3_w1000', 'cpx_k5_w5000', etc.
    """
    n_clusters_list = [3, 5, 8, 13]
    window_sizes = [1000, 5000, 20000]
    
    results = {}
    
    for n_clusters in n_clusters_list:
        for window_size in window_sizes:
            if len(trajectory) < window_size:
                # 如果轨迹太短，使用全部数据
                traj_window = trajectory
            else:
                # 使用最后window_size个点
                traj_window = trajectory[-window_size:]
            
            if len(traj_window) < n_clusters * 2:
                results[f'cpx_k{n_clusters}_w{window_size}'] = 0.0
                continue
            
            try:
                # K-means聚类
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=100)
                clusters = km.fit_predict(traj_window)
                
                # 计算复杂度
                complexity = _compute_complexity_from_clusters(clusters, n_clusters)
                results[f'cpx_k{n_clusters}_w{window_size}'] = complexity
            except:
                results[f'cpx_k{n_clusters}_w{window_size}'] = 0.0
    
    return results

def _compute_complexity_from_clusters(clusters: np.ndarray, n_clusters: int) -> float:
    """从聚类分配计算复杂度"""
    if len(clusters) < 200:
        return 0.3
    
    # 1. 有效协议数
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    active_protocols = np.sum(counts > len(clusters) * 0.02)
    protocol_score = active_protocols / n_clusters
    
    # 2. 转移熵
    trans_count = defaultdict(int)
    for i in range(len(clusters)-1):
        trans_count[(clusters[i], clusters[i+1])] += 1
    
    trans_matrix = np.zeros((n_clusters, n_clusters))
    for (i,j), cnt in trans_count.items():
        if i < n_clusters and j < n_clusters:
            trans_matrix[i,j] = cnt
    
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_prob = np.divide(trans_matrix, row_sums, 
                          out=np.zeros_like(trans_matrix), 
                          where=row_sums!=0)
    
    entropy = -np.sum(trans_prob * np.log2(trans_prob + 1e-8), axis=1)
    entropy = entropy[row_sums.flatten() > 0]
    transfer_entropy = np.mean(entropy) / np.log2(n_clusters) if len(entropy) > 0 else 0
    
    # 3. 驻留均匀度
    stay_dist = counts / len(clusters)
    uniformity = 1.0 - np.max(stay_dist)
    
    complexity = 0.4 * protocol_score + 0.4 * transfer_entropy + 0.2 * uniformity
    return np.clip(complexity, 0.0, 1.0)

def compute_complexity_sensitivity(trajectory: np.ndarray) -> Dict[str, float]:
    """
    计算complexity对尺度的敏感性
    
    Returns:
        Dict with sensitivity metrics
    """
    multi_scale_results = compute_complexity_multi_scale(trajectory)
    
    # 提取不同k和w的值
    k_values = [3, 5, 8, 13]
    w_values = [1000, 5000, 20000]
    
    # 计算跨k的稳定性（固定w）
    stability_by_w = {}
    for w in w_values:
        cpx_values = [multi_scale_results.get(f'cpx_k{k}_w{w}', 0.0) for k in k_values]
        if len([v for v in cpx_values if v > 0]) >= 2:
            valid_values = [v for v in cpx_values if v > 0]
            stability_by_w[f'stability_w{w}'] = 1.0 - min(np.std(valid_values) / (np.mean(valid_values) + 1e-8), 1.0)
        else:
            stability_by_w[f'stability_w{w}'] = 0.0
    
    # 计算跨w的稳定性（固定k）
    stability_by_k = {}
    for k in k_values:
        cpx_values = [multi_scale_results.get(f'cpx_k{k}_w{w}', 0.0) for w in w_values]
        if len([v for v in cpx_values if v > 0]) >= 2:
            valid_values = [v for v in cpx_values if v > 0]
            stability_by_k[f'stability_k{k}'] = 1.0 - min(np.std(valid_values) / (np.mean(valid_values) + 1e-8), 1.0)
        else:
            stability_by_k[f'stability_k{k}'] = 0.0
    
    # 综合敏感性（越低越好，说明跨尺度稳定）
    all_cpx = [v for v in multi_scale_results.values() if v > 0]
    if len(all_cpx) >= 3:
        overall_sensitivity = np.std(all_cpx) / (np.mean(all_cpx) + 1e-8)
        overall_stability = 1.0 - min(overall_sensitivity, 1.0)
    else:
        overall_stability = 0.0
    
    return {
        **multi_scale_results,
        **stability_by_w,
        **stability_by_k,
        'overall_stability': overall_stability
    }
