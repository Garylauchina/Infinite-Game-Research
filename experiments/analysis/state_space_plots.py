#!/usr/bin/env python3
# state_space_plots.py - 状态空间可视化
"""
用法:
    python experiments/analysis/state_space_plots.py outputs/runs/20250115/run_20250115_120000
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_state_space(run_dir: str):
    """绘制状态空间图"""
    run_path = Path(run_dir)
    
    traj_file = run_path / "raw" / "trajectory.parquet"
    if not traj_file.exists():
        print(f"❌ 轨迹文件不存在: {traj_file}")
        return
    
    df = pd.read_parquet(traj_file)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Price vs Liquidity
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(df['price_norm'], df['liquidity'], 
                         c=df['t'] if 't' in df.columns else None,
                         alpha=0.5, s=1, cmap='viridis')
    ax1.set_xlabel('Price Norm')
    ax1.set_ylabel('Liquidity')
    ax1.set_title('Price vs Liquidity')
    if 't' in df.columns:
        plt.colorbar(scatter, ax=ax1, label='Tick')
    
    # 2. Price vs Complexity
    if 'complexity' in df.columns:
        ax2 = plt.subplot(2, 3, 2)
        scatter = ax2.scatter(df['price_norm'], df['complexity'],
                             c=df['t'] if 't' in df.columns else None,
                             alpha=0.5, s=1, cmap='viridis')
        ax2.set_xlabel('Price Norm')
        ax2.set_ylabel('Complexity')
        ax2.set_title('Price vs Complexity')
        if 't' in df.columns:
            plt.colorbar(scatter, ax=ax2, label='Tick')
    
    # 3. Liquidity vs Complexity
    if 'complexity' in df.columns:
        ax3 = plt.subplot(2, 3, 3)
        scatter = ax3.scatter(df['liquidity'], df['complexity'],
                             c=df['t'] if 't' in df.columns else None,
                             alpha=0.5, s=1, cmap='viridis')
        ax3.set_xlabel('Liquidity')
        ax3.set_ylabel('Complexity')
        ax3.set_title('Liquidity vs Complexity')
        if 't' in df.columns:
            plt.colorbar(scatter, ax=ax3, label='Tick')
    
    # 4. 3D投影（Price, Liquidity, Complexity）
    if 'complexity' in df.columns:
        ax4 = plt.subplot(2, 3, 4, projection='3d')
        ax4.scatter(df['price_norm'], df['liquidity'], df['complexity'],
                   c=df['t'] if 't' in df.columns else None,
                   alpha=0.5, s=1, cmap='viridis')
        ax4.set_xlabel('Price Norm')
        ax4.set_ylabel('Liquidity')
        ax4.set_zlabel('Complexity')
        ax4.set_title('3D State Space')
    
    # 5. 聚类分配（如果有）
    if 'cluster' in df.columns:
        ax5 = plt.subplot(2, 3, 5)
        scatter = ax5.scatter(df['price_norm'], df['liquidity'],
                             c=df['cluster'], alpha=0.5, s=1, cmap='tab10')
        ax5.set_xlabel('Price Norm')
        ax5.set_ylabel('Liquidity')
        ax5.set_title('Cluster Assignments')
        plt.colorbar(scatter, ax=ax5, label='Cluster')
    
    # 6. N vs Complexity
    if 'N' in df.columns and 'complexity' in df.columns:
        ax6 = plt.subplot(2, 3, 6)
        scatter = ax6.scatter(df['N'], df['complexity'],
                             c=df['t'] if 't' in df.columns else None,
                             alpha=0.5, s=1, cmap='viridis')
        ax6.set_xlabel('Player Count (N)')
        ax6.set_ylabel('Complexity')
        ax6.set_title('N vs Complexity')
        if 't' in df.columns:
            plt.colorbar(scatter, ax=ax6, label='Tick')
    
    plt.tight_layout()
    
    output_file = run_path / "figs" / "state_space.png"
    plt.savefig(output_file, dpi=150)
    print(f"💾 状态空间图已保存: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制状态空间图")
    parser.add_argument("run_dir", type=str, help="run目录路径")
    args = parser.parse_args()
    
    plot_state_space(args.run_dir)
