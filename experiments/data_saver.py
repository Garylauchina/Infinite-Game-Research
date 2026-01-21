# data_saver.py - 实验数据保存模块
"""
按照规范保存实验数据到 outputs/runs/<date>/run_<run_id>/
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import platform
import sys

def get_run_id() -> str:
    """生成唯一的run_id（基于时间戳）"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_run_directory(base_dir: str = "outputs/runs") -> Path:
    """创建运行目录结构"""
    date_str = datetime.now().strftime("%Y%m%d")
    run_id = get_run_id()
    
    run_dir = Path(base_dir) / date_str / f"run_{run_id}"
    
    # 创建子目录
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    return run_dir

def save_metadata(run_dir: Path, config: Dict[str, Any], seed: int):
    """保存元数据"""
    meta_dir = run_dir / "meta"
    
    # 1. git_commit.txt
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        with open(meta_dir / "git_commit.txt", 'w') as f:
            f.write(f"commit: {git_commit}\n")
            f.write(f"branch: {git_branch}\n")
    except:
        with open(meta_dir / "git_commit.txt", 'w') as f:
            f.write("git: not available\n")
    
    # 2. pip_freeze.txt
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL
        ).decode()
        with open(meta_dir / "pip_freeze.txt", 'w') as f:
            f.write(pip_freeze)
    except:
        with open(meta_dir / "pip_freeze.txt", 'w') as f:
            f.write("pip freeze: not available\n")
    
    # 3. machine.json
    machine_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    
    # 尝试获取BLAS信息
    try:
        import numpy.distutils.system_info as sysinfo
        blas_info = sysinfo.get_info('blas_opt')
        if blas_info:
            machine_info["blas"] = blas_info.get('libraries', ['unknown'])
    except:
        machine_info["blas"] = "unknown"
    
    with open(meta_dir / "machine.json", 'w') as f:
        json.dump(machine_info, f, indent=2)
    
    # 4. config_resolved.yaml
    import yaml
    with open(meta_dir / "config_resolved.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 5. seeds.txt
    with open(meta_dir / "seeds.txt", 'w') as f:
        f.write(f"seed: {seed}\n")
        f.write(f"numpy_seed: {config.get('random_seed', {}).get('numpy', 42)}\n")
        f.write(f"random_seed: {config.get('random_seed', {}).get('random', 42)}\n")
        f.write(f"sklearn_seed: {config.get('random_seed', {}).get('sklearn', 42)}\n")

def save_trajectory(
    run_dir: Path,
    trajectory: np.ndarray,
    player_history: List[int],
    experience_history: List[float],
    complexity_history: List[float],
    cluster_assignments: Optional[np.ndarray] = None,
    save_interval: int = 1
):
    """
    保存轨迹数据
    
    Args:
        trajectory: shape (T, 4) - [price_norm, volatility, liquidity, imbalance]
        player_history: 玩家数量历史
        experience_history: 平均体验历史
        complexity_history: 复杂度历史
        cluster_assignments: 聚类分配（可选）
        save_interval: 保存间隔（1=全部保存）
    """
    raw_dir = run_dir / "raw"
    
    # 构建完整轨迹DataFrame
    T = len(trajectory)
    
    # 对齐时间序列（player_history和experience_history可能较短）
    # 假设它们在adjust_interval时记录
    data = {
        't': np.arange(T),
        'price_norm': trajectory[:, 0],
        'volatility': trajectory[:, 1],
        'liquidity': trajectory[:, 2],
        'imbalance': trajectory[:, 3],
    }
    
    # 如果有复杂度历史，对齐
    if len(complexity_history) == T:
        data['complexity'] = complexity_history
    else:
        # 插值或填充
        data['complexity'] = np.interp(
            np.arange(T),
            np.linspace(0, T-1, len(complexity_history)),
            complexity_history
        )
    
    # 玩家数量（需要插值，因为只在adjust_interval时记录）
    if player_history:
        adjust_interval = T // max(len(player_history), 1)
        player_ts = np.arange(0, T, adjust_interval)[:len(player_history)]
        player_interp = np.interp(
            np.arange(T),
            player_ts,
            player_history[:len(player_ts)]
        )
        data['N'] = player_interp.astype(int)
    else:
        data['N'] = np.zeros(T, dtype=int)
    
    # 平均体验（类似处理）
    if experience_history:
        adjust_interval = T // max(len(experience_history), 1)
        exp_ts = np.arange(0, T, adjust_interval)[:len(experience_history)]
        exp_interp = np.interp(
            np.arange(T),
            exp_ts,
            experience_history[:len(exp_ts)]
        )
        data['avg_exp'] = exp_interp
    else:
        data['avg_exp'] = np.zeros(T)
    
    # 聚类分配（如果有）
    if cluster_assignments is not None and len(cluster_assignments) == T:
        data['cluster'] = cluster_assignments
    
    df = pd.DataFrame(data)
    
    # 按save_interval采样
    if save_interval > 1:
        df = df.iloc[::save_interval].reset_index(drop=True)
    
    # 保存为parquet（高效压缩）
    df.to_parquet(raw_dir / "trajectory.parquet", compression='snappy', index=False)
    
    # 同时保存CSV（便于查看）
    df.to_csv(raw_dir / "trajectory.csv", index=False)
    
    # 分别保存历史数据
    if player_history:
        pd.DataFrame({'N': player_history}).to_csv(
            raw_dir / "player_history.csv", index=False
        )
    if experience_history:
        pd.DataFrame({'avg_exp': experience_history}).to_csv(
            raw_dir / "experience_history.csv", index=False
        )
    if complexity_history:
        pd.DataFrame({'complexity': complexity_history}).to_csv(
            raw_dir / "complexity_history.csv", index=False
        )

def save_metrics(run_dir: Path, metrics: Dict[str, Any]):
    """保存指标"""
    metrics_dir = run_dir / "metrics"
    
    # metrics.json
    with open(metrics_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
