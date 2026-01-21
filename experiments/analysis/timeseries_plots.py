#!/usr/bin/env python3
# timeseries_plots.py - 生成时间序列图
"""
用法:
    python experiments/analysis/timeseries_plots.py outputs/runs/20250115/run_20250115_120000
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def plot_timeseries(run_dir: str):
    """绘制时间序列图"""
    run_path = Path(run_dir)
    
    traj_file = run_path / "raw" / "trajectory.parquet"
    if not traj_file.exists():
        print(f"❌ 轨迹文件不存在: {traj_file}")
        return
    
    df = pd.read_parquet(traj_file)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # 1. 价格轨迹
    axes[0, 0].plot(df['t'], df['price_norm'], alpha=0.7)
    axes[0, 0].set_xlabel('Tick')
    axes[0, 0].set_ylabel('Price Norm')
    axes[0, 0].set_title('Price Trajectory')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 流动性
    axes[0, 1].plot(df['t'], df['liquidity'], alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Tick')
    axes[0, 1].set_ylabel('Liquidity')
    axes[0, 1].set_title('Liquidity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 复杂度
    if 'complexity' in df.columns:
        axes[1, 0].plot(df['t'], df['complexity'], alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('Tick')
        axes[1, 0].set_ylabel('Complexity')
        axes[1, 0].set_title('Complexity')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 玩家数量
    if 'N' in df.columns:
        axes[1, 1].plot(df['t'], df['N'], alpha=0.7, color='orange', marker='o', markersize=2)
        axes[1, 1].set_xlabel('Tick')
        axes[1, 1].set_ylabel('Player Count (N)')
        axes[1, 1].set_title('Player Count')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 平均体验
    if 'avg_exp' in df.columns:
        axes[2, 0].plot(df['t'], df['avg_exp'], alpha=0.7, color='red')
        axes[2, 0].set_xlabel('Tick')
        axes[2, 0].set_ylabel('Avg Experience')
        axes[2, 0].set_title('Average Experience')
        axes[2, 0].grid(True, alpha=0.3)
    
    # 6. 波动率
    axes[2, 1].plot(df['t'], df['volatility'], alpha=0.7, color='brown')
    axes[2, 1].set_xlabel('Tick')
    axes[2, 1].set_ylabel('Volatility')
    axes[2, 1].set_title('Volatility')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = run_path / "figs" / "timeseries.png"
    plt.savefig(output_file, dpi=150)
    print(f"💾 时间序列图已保存: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制时间序列图")
    parser.add_argument("run_dir", type=str, help="run目录路径")
    args = parser.parse_args()
    
    plot_timeseries(args.run_dir)
