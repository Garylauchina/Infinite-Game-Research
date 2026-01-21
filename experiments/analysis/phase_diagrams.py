#!/usr/bin/env python3
# phase_diagrams.py - 扫参相位图（输出表+图）
"""
用法:
    python experiments/analysis/phase_diagrams.py outputs/runs/sweep_simulation.max_n.csv
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_phase_diagram(sweep_csv: str, output_dir: str = None):
    """绘制参数扫描相位图"""
    df = pd.read_csv(sweep_csv)
    
    if 'error' in df.columns and not df['error'].isna().all():
        print("⚠️  存在错误记录，将跳过")
        df = df[df['error'].isna()]
    
    if len(df) == 0:
        print("❌ 没有有效数据")
        return
    
    # 获取参数名和值
    param_col = [c for c in df.columns if c.startswith('param_')][0]
    param_name = df[param_col].iloc[0] if param_col == 'param_path' else 'parameter'
    param_values = df['param_value'].unique()
    
    # 选择指标
    metrics = ['final_player_count', 'final_complexity', 'avg_liquidity']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("❌ 没有可用的指标")
        return
    
    # 计算每个参数值的统计
    summary = df.groupby('param_value')[available_metrics].agg(['mean', 'std'])
    
    # 绘制
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        means = summary[metric]['mean']
        stds = summary[metric]['std']
        
        ax.errorbar(param_values, means, yerr=stds, marker='o', capsize=5)
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Parameter')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    if output_dir is None:
        output_dir = Path(sweep_csv).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"phase_diagram_{Path(sweep_csv).stem}.png"
    plt.savefig(output_file, dpi=150)
    print(f"💾 相位图已保存: {output_file}")
    plt.close()
    
    # 打印汇总表
    print("\n📊 参数扫描汇总表:")
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制参数扫描相位图")
    parser.add_argument("sweep_csv", type=str, help="参数扫描CSV文件")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()
    
    plot_phase_diagram(args.sweep_csv, args.output_dir)
