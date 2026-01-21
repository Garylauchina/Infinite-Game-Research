#!/usr/bin/env python3
# run_single.py - 运行单seed/单config实验
"""
用法:
    python experiments/run_single.py --config experiments/configs/quick_test.yaml --seed 42
    python experiments/run_single.py --config experiments/configs/default.yaml --seed 0
"""

import argparse
import sys
import os
import numpy as np
import random
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core_system import V5MarketSimulator
from experiments.config_loader import load_config, save_resolved_config
from experiments.data_saver import (
    create_run_directory, save_metadata, save_trajectory, save_metrics
)
import warnings
warnings.filterwarnings('ignore')

def run_single_experiment(config_path: str = None, seed: int = 42, output_dir: str = "outputs/runs", config_dict: dict = None):
    """运行单个实验"""
    # 加载配置
    if config_dict is not None:
        config = config_dict
    else:
        config = load_config(config_path)
    
    # 创建运行目录
    run_dir = create_run_directory(output_dir)
    print(f"📁 运行目录: {run_dir}")
    
    # 保存元数据
    save_metadata(run_dir, config, seed)
    save_resolved_config(config, str(run_dir / "meta" / "config_resolved.yaml"))
    
    # 设置随机种子
    sim_config = config['simulation']
    seed_config = config['random_seed']
    
    np.random.seed(seed_config['numpy'] + seed)
    random.seed(seed_config['random'] + seed)
    
    # 创建模拟器
    sim = V5MarketSimulator(
        ticks=sim_config['ticks'],
        adjust_interval=sim_config['adjust_interval'],
        MAX_N=sim_config.get('max_n')
    )
    
    # 配置模拟器参数
    sim.ADD_PLAYER_THRESHOLD = sim_config.get('add_player_threshold', 0.35)
    sim.REMOVE_PLAYER_THRESHOLD = sim_config.get('remove_player_threshold', 0.15)
    
    # 运行仿真
    print(f"🚀 开始运行 seed={seed}, ticks={sim_config['ticks']}")
    metrics = sim.run_simulation()
    
    # 准备数据
    trajectory = np.array(sim.state_trajectory)
    player_history = sim.player_history
    experience_history = sim.experience_history
    complexity_history = sim.complexity_history
    
    # 获取聚类分配（如果有）
    cluster_assignments = None
    if config['output'].get('save_cluster_assignments', True):
        if hasattr(sim.structure_metrics, 'cluster_assignments'):
            cluster_assignments = np.array(list(sim.structure_metrics.cluster_assignments))
    
    # 保存数据
    print("💾 保存数据...")
    save_interval = config['output'].get('save_interval', 1)
    save_trajectory(
        run_dir,
        trajectory,
        player_history,
        experience_history,
        complexity_history,
        cluster_assignments,
        save_interval
    )
    
    save_metrics(run_dir, metrics)
    
    print(f"✅ 实验完成！结果保存在: {run_dir}")
    return run_dir, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行单seed实验")
    parser.add_argument("--config", type=str, default="experiments/configs/default.yaml",
                       help="配置文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="outputs/runs",
                       help="输出目录")
    
    args = parser.parse_args()
    
    run_dir, metrics = run_single_experiment(
        args.config, args.seed, args.output_dir
    )
    
    print("\n📊 最终指标:")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k:25s}: {v}")
