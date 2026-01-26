#!/usr/bin/env python3
"""
V5.2 Adaptive-K 对照实验

实验组：
- G0: V5.2 fixed K=1（基线）
- G1: V5.2 fixed K=3（你们已有）
- G2: V5.2 adaptive K（本改动）

Seeds: {0, 1, 2, 42}
Ticks: 500,000（与 P3 longrun 对齐）
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# 添加 core_system 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_system.main import V5MarketSimulator
from core_system.chaos_rules import compute_collapse_proxy_debug
from core_system.metrics import StructureMetrics


class AdaptiveKExperiment:
    """Adaptive-K 对照实验管理器"""
    
    def __init__(self, output_dir: str = 'experiments/output/adaptive_k'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 实验配置
        self.seeds = [0, 1, 2, 42]
        self.ticks = 500000
        self.adjust_interval = 2000
        
        # Adaptive-K 配置
        self.adaptive_k_config = {
            'alpha': 1.6,
            'K_HARD': 32,
            'panel_size': 8,
            'eps': 0.01,
            'sat_patience': 2,
            'delta_k': 2
        }
        
        # 结果存储
        self.results = {
            'G0': {},  # K=1
            'G1': {},  # K=3
            'G2': {}   # Adaptive-K
        }
    
    def run_single(self, group: str, seed: int, ticks: int = None) -> Dict:
        """运行单次实验"""
        ticks = ticks or self.ticks
        
        print(f"\n{'='*60}")
        print(f"Running {group} seed={seed} ticks={ticks}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        if group == 'G0':
            # Fixed K=1
            sim = V5MarketSimulator(
                ticks=ticks,
                adjust_interval=self.adjust_interval,
                probe_count=1,
                adaptive_k=False,
                seed=seed
            )
        elif group == 'G1':
            # Fixed K=3
            sim = V5MarketSimulator(
                ticks=ticks,
                adjust_interval=self.adjust_interval,
                probe_count=3,
                adaptive_k=False,
                seed=seed
            )
        elif group == 'G2':
            # Adaptive-K
            sim = V5MarketSimulator(
                ticks=ticks,
                adjust_interval=self.adjust_interval,
                probe_count=1,  # 基础值，会被 adaptive 覆盖
                adaptive_k=True,
                adaptive_k_config=self.adaptive_k_config,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown group: {group}")
        
        # 运行仿真
        metrics = sim.run_simulation()
        
        elapsed = time.time() - start_time
        
        # 收集额外统计
        result = {
            'group': group,
            'seed': seed,
            'ticks': ticks,
            'duration_seconds': elapsed,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Adaptive-K 特有统计
        if group == 'G2' and sim.k_max_history:
            k_max_arr = np.array(sim.k_max_history)
            k_dist_arr = np.array(sim.k_distribution) if sim.k_distribution else np.array([1])
            
            result['adaptive_k_stats'] = {
                'k_max_mean': float(np.mean(k_max_arr)),
                'k_max_std': float(np.std(k_max_arr)),
                'k_max_min': int(np.min(k_max_arr)),
                'k_max_max': int(np.max(k_max_arr)),
                'k_max_p50': float(np.median(k_max_arr)),
                'k_max_p95': float(np.percentile(k_max_arr, 95)),
                'k_sampled_mean': float(np.mean(k_dist_arr)),
                'k_sampled_std': float(np.std(k_dist_arr)),
                'k_sampled_p50': float(np.median(k_dist_arr)),
                'k_sampled_p95': float(np.percentile(k_dist_arr, 95)),
                'config': self.adaptive_k_config
            }
        
        # 保存单次结果
        output_path = os.path.join(
            self.output_dir, 
            f'{group}_seed{seed}_ticks{ticks}.json'
        )
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\nSaved: {output_path}")
        
        return result
    
    def run_all(self, ticks: int = None, seeds: List[int] = None):
        """运行所有实验"""
        ticks = ticks or self.ticks
        seeds = seeds or self.seeds
        
        total_start = time.time()
        
        for group in ['G0', 'G1', 'G2']:
            for seed in seeds:
                result = self.run_single(group, seed, ticks)
                self.results[group][seed] = result
        
        total_elapsed = time.time() - total_start
        
        # 生成汇总报告
        self._generate_summary(ticks, seeds, total_elapsed)
        
        return self.results
    
    def _generate_summary(self, ticks: int, seeds: List[int], total_elapsed: float):
        """生成汇总报告"""
        summary = {
            'experiment': 'V5.2 Adaptive-K Comparison',
            'created_at': datetime.now().isoformat(),
            'config': {
                'ticks': ticks,
                'seeds': seeds,
                'adjust_interval': self.adjust_interval,
                'adaptive_k_config': self.adaptive_k_config
            },
            'total_duration_seconds': total_elapsed,
            'groups': {}
        }
        
        for group, group_results in self.results.items():
            if not group_results:
                continue
            
            metrics_list = [r['metrics'] for r in group_results.values()]
            
            # 计算组统计
            group_summary = {
                'n_seeds': len(metrics_list),
                'avg_final_player_count': np.mean([m['final_player_count'] for m in metrics_list]),
                'avg_final_complexity': np.mean([m['final_complexity'] for m in metrics_list]),
                'avg_avg_experience': np.mean([m['final_avg_experience'] for m in metrics_list]),
                'avg_active_protocols': np.mean([m['active_protocols'] for m in metrics_list])
            }
            
            # Adaptive-K 特有统计
            if group == 'G2':
                adaptive_stats = [r.get('adaptive_k_stats', {}) for r in group_results.values()]
                if adaptive_stats and adaptive_stats[0]:
                    group_summary['k_max_mean'] = np.mean([s.get('k_max_mean', 0) for s in adaptive_stats])
                    group_summary['k_max_std'] = np.mean([s.get('k_max_std', 0) for s in adaptive_stats])
                    group_summary['k_sampled_mean'] = np.mean([s.get('k_sampled_mean', 0) for s in adaptive_stats])
                    group_summary['k_sampled_p95'] = np.mean([s.get('k_sampled_p95', 0) for s in adaptive_stats])
            
            summary['groups'][group] = group_summary
        
        # 保存汇总
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total duration: {total_elapsed/60:.1f} minutes")
        print(f"\nGroup Comparison:")
        print(f"{'Group':<10} {'Final N':<12} {'Complexity':<12} {'Experience':<12}")
        print("-" * 50)
        for group, stats in summary['groups'].items():
            print(f"{group:<10} {stats['avg_final_player_count']:<12.1f} "
                  f"{stats['avg_final_complexity']:<12.3f} "
                  f"{stats['avg_avg_experience']:<12.3f}")
        
        if 'G2' in summary['groups'] and 'k_max_mean' in summary['groups']['G2']:
            g2 = summary['groups']['G2']
            print(f"\nAdaptive-K Statistics:")
            print(f"  K_max mean: {g2['k_max_mean']:.2f} ± {g2['k_max_std']:.2f}")
            print(f"  K_sampled mean: {g2['k_sampled_mean']:.2f}, p95: {g2['k_sampled_p95']:.1f}")
        
        print(f"\nSummary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='V5.2 Adaptive-K Experiment')
    parser.add_argument('--ticks', type=int, default=50000, help='Simulation ticks (default 50k for quick test)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Random seeds')
    parser.add_argument('--group', type=str, default=None, help='Run single group (G0/G1/G2)')
    parser.add_argument('--seed', type=int, default=None, help='Run single seed')
    parser.add_argument('--out', type=str, default='experiments/output/adaptive_k', help='Output directory')
    parser.add_argument('--full', action='store_true', help='Run full 500k ticks experiment')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # 完整实验模式
    if args.full:
        args.ticks = 500000
        args.seeds = [0, 1, 2, 42]
    
    exp = AdaptiveKExperiment(output_dir=args.out)
    
    if args.group and args.seed is not None:
        # 单次运行
        exp.run_single(args.group, args.seed, args.ticks)
    elif args.group:
        # 运行单个组的所有 seeds
        for seed in args.seeds:
            exp.run_single(args.group, seed, args.ticks)
    else:
        # 运行所有
        exp.run_all(ticks=args.ticks, seeds=args.seeds)


if __name__ == "__main__":
    main()
