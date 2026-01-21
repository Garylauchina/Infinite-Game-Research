#!/usr/bin/env python3
# summarize.py - 汇总run内指标
"""
用法:
    python experiments/analysis/summarize.py outputs/runs/20250115/run_20250115_120000
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys

def summarize_run(run_dir: str):
    """汇总单个run的指标"""
    run_path = Path(run_dir)
    
    # 读取指标
    metrics_file = run_path / "metrics" / "metrics.json"
    if not metrics_file.exists():
        print(f"❌ 指标文件不存在: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # 读取轨迹数据
    traj_file = run_path / "raw" / "trajectory.parquet"
    if traj_file.exists():
        df = pd.read_parquet(traj_file)
        traj_stats = {
            'trajectory_length': len(df),
            'price_norm_mean': df['price_norm'].mean(),
            'price_norm_std': df['price_norm'].std(),
            'liquidity_mean': df['liquidity'].mean(),
            'liquidity_std': df['liquidity'].std(),
            'complexity_mean': df['complexity'].mean() if 'complexity' in df.columns else None,
            'N_mean': df['N'].mean() if 'N' in df.columns else None,
            'N_max': df['N'].max() if 'N' in df.columns else None,
        }
    else:
        traj_stats = {}
    
    # 打印汇总
    print(f"📊 Run汇总: {run_path.name}")
    print("=" * 60)
    
    print("\n🎯 核心指标:")
    for key in ['final_player_count', 'final_complexity', 'avg_liquidity', 'final_avg_experience']:
        if key in metrics:
            print(f"  {key:25s}: {metrics[key]:.4f}")
    
    print("\n📈 轨迹统计:")
    for key, value in traj_stats.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key:25s}: {value:.4f}")
            else:
                print(f"  {key:25s}: {value}")
    
    print("\n📁 文件位置:")
    print(f"  {run_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="汇总run指标")
    parser.add_argument("run_dir", type=str, help="run目录路径")
    args = parser.parse_args()
    
    summarize_run(args.run_dir)
