#!/usr/bin/env python3
# compare_runs.py - 对比两个run_id
"""
用法:
    python experiments/analysis/compare_runs.py run1_dir run2_dir
"""

import argparse
import json
import pandas as pd
from pathlib import Path

def compare_runs(run1_dir: str, run2_dir: str):
    """对比两个run"""
    run1_path = Path(run1_dir)
    run2_path = Path(run2_dir)
    
    # 读取指标
    def load_metrics(run_path):
        metrics_file = run_path / "metrics" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    metrics1 = load_metrics(run1_path)
    metrics2 = load_metrics(run2_path)
    
    # 读取轨迹
    def load_trajectory(run_path):
        traj_file = run_path / "raw" / "trajectory.parquet"
        if traj_file.exists():
            return pd.read_parquet(traj_file)
        return None
    
    df1 = load_trajectory(run1_path)
    df2 = load_trajectory(run2_path)
    
    print(f"🔬 对比: {run1_path.name} vs {run2_path.name}")
    print("=" * 80)
    
    # 对比指标
    print("\n📊 指标对比:")
    all_keys = set(metrics1.keys()) | set(metrics2.keys())
    for key in sorted(all_keys):
        v1 = metrics1.get(key, 'N/A')
        v2 = metrics2.get(key, 'N/A')
        
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = v2 - v1
            pct = (diff / v1 * 100) if v1 != 0 else 0
            print(f"  {key:30s}: {v1:10.4f} → {v2:10.4f} ({diff:+.4f}, {pct:+.2f}%)")
        else:
            print(f"  {key:30s}: {v1} → {v2}")
    
    # 对比轨迹统计
    if df1 is not None and df2 is not None:
        print("\n📈 轨迹统计对比:")
        for col in ['price_norm', 'liquidity', 'complexity', 'N']:
            if col in df1.columns and col in df2.columns:
                m1 = df1[col].mean()
                m2 = df2[col].mean()
                diff = m2 - m1
                pct = (diff / m1 * 100) if m1 != 0 else 0
                print(f"  {col:30s}: {m1:10.4f} → {m2:10.4f} ({diff:+.4f}, {pct:+.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比两个run")
    parser.add_argument("run1_dir", type=str, help="第一个run目录")
    parser.add_argument("run2_dir", type=str, help="第二个run目录")
    args = parser.parse_args()
    
    compare_runs(args.run1_dir, args.run2_dir)
