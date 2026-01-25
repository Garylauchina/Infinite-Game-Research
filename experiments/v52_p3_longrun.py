#!/usr/bin/env python3
"""
V5.2 P3 长跑补充验证：collapse_proxy 语义正确性验证

目标：
1. collapse_proxy 高 → 对应结构塌缩（单点吸引子 or 均匀噪声）
2. collapse_proxy 低 → 对应健康结构（存在骨架 / 多结构常驻）
"""

import os
import sys
import json
import zstandard as zstd
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import hashlib

# 设置线程为1（复现性）
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core_system.state_engine import StateEngine, MarketState, Action
from core_system.trading_rules import SIMPLEST_RULES
from core_system.chaos_rules import (
    compute_match_prob, ChaosFactor, 
    compute_collapse_proxy, compute_collapse_proxy_debug
)
from core_system.random_player import RandomExperiencePlayer, ProbeResult
from core_system.metrics import StructureMetrics


@dataclass
class SamplePoint:
    """单个采样点"""
    t: int
    N: int
    avg_exp: float
    complexity: float
    H_transfer_norm: float
    active_protocols_ratio: float
    match_count: int
    H_price_norm: float
    collapse_proxy: float
    # 可选状态量
    price_norm: float
    volatility: float
    liquidity: float
    imbalance: float


class P3LongRunExperiment:
    """P3 长跑验证实验"""
    
    # 固定参数
    FIXED_PARAMS = {
        'n_clusters': 5,
        'window_size': 5000,
        'cluster_update_interval': 500,
        'adjust_interval': 2000,
        'initial_players': 3,
        'add_threshold': 0.35,
        'remove_threshold': 0.15,
        'price_min': SIMPLEST_RULES['price_min'],
        'price_max': SIMPLEST_RULES['price_max'],
    }
    
    def __init__(self, 
                 seed: int,
                 K: int,
                 ticks: int,
                 downsample: int = 10,
                 output_dir: str = 'experiments/output/v52_p3_longrun'):
        self.seed = seed
        self.K = K
        self.ticks = ticks
        self.downsample = downsample
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成参数哈希
        self.params_hash = self._compute_params_hash()
        
        # 输出文件
        self.data_file = self.output_dir / f'v52_p3_longrun_seed{seed}_K{K}_ticks{ticks}.jsonl.zst'
        self.summary_file = self.output_dir / f'v52_p3_longrun_seed{seed}_K{K}_ticks{ticks}_summary.json'
        
        # 采样数据缓存
        self.samples: List[SamplePoint] = []
        self.collapse_history: List[float] = []
        
        # 停止判据状态
        self.stable_checks = 0
        self.collapse_event_detected = False
        self.aborted = False
        
    def _compute_params_hash(self) -> str:
        """计算参数哈希"""
        params_str = json.dumps({
            **self.FIXED_PARAMS,
            'seed': self.seed,
            'K': self.K,
            'ticks': self.ticks,
        }, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    def _create_simulator(self):
        """创建模拟器组件"""
        np.random.seed(self.seed)
        
        # 创建玩家
        players = [
            RandomExperiencePlayer(i, probe_count=self.K)
            for i in range(self.FIXED_PARAMS['initial_players'])
        ]
        
        # 创建状态引擎
        engine = StateEngine()
        
        # 创建结构度量
        structure_metrics = StructureMetrics(
            n_clusters=self.FIXED_PARAMS['n_clusters'],
            window_size=self.FIXED_PARAMS['window_size']
        )
        
        # 创建混沌因子
        chaos_factor = ChaosFactor()
        
        return {
            'players': players,
            'engine': engine,
            'structure_metrics': structure_metrics,
            'chaos_factor': chaos_factor,
        }
    
    def _sample_probes(self, sim, all_probes, s_t):
        """执行探针采样并收集成交信息"""
        players = sim['players']
        player_count = len(players)
        flat_actions = [p for probes in all_probes for p in probes]
        avg_exp = np.mean([p.experience_score for p in players])
        
        all_match_prices = []
        probe_results = []
        
        for probes in all_probes:
            n_success = 0
            match_prices = []
            
            for probe in probes:
                prob = compute_match_prob(
                    probe.price, s_t, flat_actions, player_count,
                    chaos_factor_manager=sim['chaos_factor'],
                    avg_exp=avg_exp
                )
                if np.random.random() < prob:
                    n_success += 1
                    match_prices.append(probe.price)
                    all_match_prices.append(probe.price)
            
            k = len(probes)
            n_failure = k - n_success
            success_rate = n_success / k if k > 0 else 0.0
            
            if len(match_prices) >= 2:
                price_range = self.FIXED_PARAMS['price_max'] - self.FIXED_PARAMS['price_min']
                match_dispersion = np.std(match_prices) / price_range
            else:
                match_dispersion = 0.0
            
            probe_result = ProbeResult(
                probes=probes,
                n_success=n_success,
                n_failure=n_failure,
                match_prices=match_prices,
                success_rate=success_rate,
                match_dispersion=match_dispersion
            )
            probe_results.append(probe_result)
        
        return probe_results, all_match_prices
    
    def _adjust_participation(self, sim, avg_exp):
        """调整玩家数量"""
        players = sim['players']
        
        if len(players) <= 2 and avg_exp > 0.25:
            new_player = RandomExperiencePlayer(len(players), probe_count=self.K)
            players.append(new_player)
            return
        
        if avg_exp > self.FIXED_PARAMS['add_threshold']:
            new_player = RandomExperiencePlayer(len(players), probe_count=self.K)
            players.append(new_player)
        elif avg_exp < self.FIXED_PARAMS['remove_threshold'] and len(players) > 2:
            worst_idx = np.argmin([p.experience_score for p in players])
            players.pop(worst_idx)
    
    def _check_stop_conditions(self, t: int) -> Tuple[bool, str]:
        """检查停止条件"""
        # burn-in 后才开始检查
        if t < 200000:
            return False, ""
        
        # Stop-B: 检测塌缩态
        if len(self.collapse_history) >= 500:  # 5000 ticks / downsample=10
            recent = self.collapse_history[-500:]
            if np.mean(recent) >= 0.85:
                self.collapse_event_detected = True
                return True, "collapse_event_detected"
        
        # Stop-A: 分布稳定性检查（每 20k ticks）
        check_interval = 20000 // self.downsample
        samples_for_stability = 50000 // self.downsample
        
        if len(self.collapse_history) >= samples_for_stability:
            if len(self.collapse_history) % check_interval == 0:
                window = self.collapse_history[-samples_for_stability:]
                mean_val = np.mean(window)
                p95_val = np.percentile(window, 95)
                
                # 检查与上次的变化
                if hasattr(self, '_last_mean') and hasattr(self, '_last_p95'):
                    mean_change = abs(mean_val - self._last_mean)
                    p95_change = abs(p95_val - self._last_p95)
                    
                    if mean_change < 0.02 and p95_change < 0.03:
                        self.stable_checks += 1
                        if self.stable_checks >= 3:
                            return True, "distribution_stable"
                    else:
                        self.stable_checks = 0
                
                self._last_mean = mean_val
                self._last_p95 = p95_val
        
        return False, ""
    
    def run(self) -> Dict:
        """运行实验"""
        print(f"\n{'='*60}")
        print(f"P3 长跑验证: seed={self.seed}, K={self.K}, ticks={self.ticks}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # 创建模拟器
        sim = self._create_simulator()
        
        # 初始状态
        s_t = MarketState(0.5, 0.3, 0.5, 0.5, 0.5)
        
        # 数据收集
        match_prices_history = []
        no_trade_count = 0
        
        # 打开输出文件
        cctx = zstd.ZstdCompressor(level=3)
        with open(self.data_file, 'wb') as f:
            compressor = cctx.stream_writer(f)
            lines_written = 0
            
            for t in range(self.ticks):
                # 更新 chaos factor
                sim['chaos_factor'].update_tick(t)
                
                if match_prices_history:
                    recent_prices = [p for prices in match_prices_history[-10:] for p in prices]
                    sim['chaos_factor'].set_match_prices(recent_prices)
                
                # 获取 active_protocols_ratio（从 metrics）
                _, mdbg = sim['structure_metrics'].compute_complexity_debug()
                active_ratio_struct = mdbg['active_ratio_struct']
                sim['chaos_factor'].set_active_protocols_ratio(active_ratio_struct)
                
                # 生成探针
                all_probes = [p.generate_probes(s_t) for p in sim['players']]
                
                # 成交判定
                probe_results, all_match_prices = self._sample_probes(sim, all_probes, s_t)
                match_prices_history.append(all_match_prices)
                if len(match_prices_history) > 100:
                    match_prices_history.pop(0)
                
                # 收集所有成交
                all_actions = [probe for probes in all_probes for probe in probes]
                all_matches_actions = [Action(p, 'buy') for p in all_match_prices]
                
                # 更新体验
                for player, probe_result in zip(sim['players'], probe_results):
                    player.set_probe_result(probe_result)
                    player.update_experience(True, s_t)
                
                # 状态演化
                s_t = sim['engine'].update(s_t, all_actions, all_matches_actions)
                
                # 更新结构密度
                state_features = np.array([
                    s_t.price_norm, s_t.volatility,
                    s_t.liquidity, s_t.imbalance
                ])
                sim['structure_metrics'].update_trajectory(state_features)
                
                # 计算 complexity（使用 debug 版本）
                if t % self.FIXED_PARAMS['cluster_update_interval'] == 0 and t >= 200:
                    complexity, mdbg = sim['structure_metrics'].compute_complexity_debug()
                    s_t.complexity = complexity
                else:
                    complexity = s_t.complexity
                
                # 计算 collapse_proxy（使用 debug 版本）
                if all_match_prices:
                    collapse, cdbg = compute_collapse_proxy_debug(all_match_prices, active_ratio_struct)
                    H_price_norm = cdbg['H_price_norm']
                    match_count = cdbg['match_count']
                else:
                    collapse = 1.0  # 无成交视为塌缩
                    H_price_norm = 0.0
                    match_count = 0
                    no_trade_count += 1
                
                # 记录 collapse 历史
                self.collapse_history.append(collapse)
                
                # 采样
                if t % self.downsample == 0:
                    N = len(sim['players'])
                    avg_exp = np.mean([p.experience_score for p in sim['players']])
                    
                    sample = SamplePoint(
                        t=t,
                        N=N,
                        avg_exp=avg_exp,
                        complexity=complexity,
                        H_transfer_norm=mdbg['transfer_entropy_norm'],
                        active_protocols_ratio=active_ratio_struct,
                        match_count=match_count,
                        H_price_norm=H_price_norm,
                        collapse_proxy=collapse,
                        price_norm=s_t.price_norm,
                        volatility=s_t.volatility,
                        liquidity=s_t.liquidity,
                        imbalance=s_t.imbalance,
                    )
                    self.samples.append(sample)
                    
                    # 写入 JSONL
                    line = json.dumps(asdict(sample)) + '\n'
                    compressor.write(line.encode())
                    lines_written += 1
                    
                    # 每 10k 行 flush
                    if lines_written % 10000 == 0:
                        compressor.flush()
                
                # 参与调整
                if t % self.FIXED_PARAMS['adjust_interval'] == 0 and t > 0:
                    avg_exp = np.mean([p.experience_score for p in sim['players']])
                    self._adjust_participation(sim, avg_exp)
                
                # 进度输出
                if t % 50000 == 0:
                    N = len(sim['players'])
                    avg_exp = np.mean([p.experience_score for p in sim['players']])
                    print(f"  t={t:6d} N={N:3d} exp={avg_exp:.3f} cpx={complexity:.3f} "
                          f"collapse={collapse:.3f} H_price={H_price_norm:.3f}")
                
                # 检查停止条件
                should_stop, stop_reason = self._check_stop_conditions(t)
                if should_stop:
                    print(f"\n  ⚠️ 提前停止: {stop_reason} at t={t}")
                    break
            
            compressor.close()
        
        end_time = datetime.now()
        
        # 生成 summary
        summary = self._generate_summary(start_time, end_time, no_trade_count, lines_written)
        
        # 保存 summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✅ 实验完成")
        print(f"  数据文件: {self.data_file}")
        print(f"  摘要文件: {self.summary_file}")
        
        return summary
    
    def _generate_summary(self, start_time, end_time, no_trade_count, lines_written) -> Dict:
        """生成实验摘要"""
        # 获取 burn-in 后的数据
        burn_in_samples = 200000 // self.downsample
        if len(self.samples) > burn_in_samples:
            post_burnin = self.samples[burn_in_samples:]
        else:
            post_burnin = self.samples
        
        collapse_values = [s.collapse_proxy for s in post_burnin]
        H_price_values = [s.H_price_norm for s in post_burnin]
        H_transfer_values = [s.H_transfer_norm for s in post_burnin]
        active_ratio_values = [s.active_protocols_ratio for s in post_burnin]
        
        # collapse_proxy 分布统计
        collapse_stats = {
            'mean': float(np.mean(collapse_values)),
            'std': float(np.std(collapse_values)),
            'p50': float(np.percentile(collapse_values, 50)),
            'p95': float(np.percentile(collapse_values, 95)),
            'max': float(np.max(collapse_values)),
        }
        
        # 相关性分析
        correlations = {}
        if len(collapse_values) > 10:
            correlations['collapse_vs_H_price'] = float(np.corrcoef(collapse_values, H_price_values)[0, 1])
            correlations['collapse_vs_H_transfer'] = float(np.corrcoef(collapse_values, H_transfer_values)[0, 1])
            correlations['collapse_vs_active_ratio'] = float(np.corrcoef(collapse_values, active_ratio_values)[0, 1])
        
        # 高 collapse 时的分析
        high_collapse_samples = [s for s in post_burnin if s.collapse_proxy >= 0.8]
        high_collapse_analysis = {}
        if high_collapse_samples:
            high_H_price = [s.H_price_norm for s in high_collapse_samples]
            high_active = [s.active_protocols_ratio for s in high_collapse_samples]
            high_collapse_analysis = {
                'count': len(high_collapse_samples),
                'H_price_mean': float(np.mean(high_H_price)),
                'H_price_near_0': sum(1 for h in high_H_price if h < 0.2) / len(high_H_price),
                'H_price_near_1': sum(1 for h in high_H_price if h > 0.8) / len(high_H_price),
                'active_ratio_mean': float(np.mean(high_active)),
            }
        
        # 获取 git commit
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=project_root
            ).decode().strip()[:8]
        except:
            git_commit = 'unknown'
        
        return {
            'experiment': {
                'seed': self.seed,
                'K': self.K,
                'ticks': self.ticks,
                'downsample': self.downsample,
                'actual_samples': len(self.samples),
                'lines_written': lines_written,
            },
            'params': self.FIXED_PARAMS,
            'params_hash': self.params_hash,
            'git_commit': git_commit,
            'timing': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
            },
            'no_trade_ratio': no_trade_count / self.ticks,
            'collapse_stats': collapse_stats,
            'correlations': correlations,
            'high_collapse_analysis': high_collapse_analysis,
            'stop_conditions': {
                'stable_checks': self.stable_checks,
                'collapse_event_detected': self.collapse_event_detected,
                'aborted': self.aborted,
            },
        }


def main():
    parser = argparse.ArgumentParser(description='V5.2 P3 长跑验证')
    parser.add_argument('--ticks', type=int, default=500000, help='运行 ticks 数')
    parser.add_argument('--downsample', type=int, default=10, help='采样间隔')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='随机种子列表')
    parser.add_argument('--K_list', type=int, nargs='+', default=[1, 3], help='K 值列表')
    parser.add_argument('--out', type=str, default='experiments/output/v52_p3_longrun', help='输出目录')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"V5.2 P3 长跑验证")
    print(f"ticks={args.ticks}, downsample={args.downsample}")
    print(f"seeds={args.seeds}, K_list={args.K_list}")
    print(f"{'#'*60}")
    
    all_summaries = []
    
    for seed in args.seeds:
        for K in args.K_list:
            exp = P3LongRunExperiment(
                seed=seed,
                K=K,
                ticks=args.ticks,
                downsample=args.downsample,
                output_dir=args.out,
            )
            summary = exp.run()
            all_summaries.append(summary)
    
    # 保存 manifest
    manifest = {
        'experiment_type': 'v52_p3_longrun',
        'created_at': datetime.now().isoformat(),
        'total_runs': len(all_summaries),
        'params': {
            'ticks': args.ticks,
            'downsample': args.downsample,
            'seeds': args.seeds,
            'K_list': args.K_list,
        },
        'runs': [
            {
                'seed': s['experiment']['seed'],
                'K': s['experiment']['K'],
                'file': f"v52_p3_longrun_seed{s['experiment']['seed']}_K{s['experiment']['K']}_ticks{args.ticks}.jsonl.zst",
                'collapse_mean': s['collapse_stats']['mean'],
                'collapse_p95': s['collapse_stats']['p95'],
            }
            for s in all_summaries
        ],
    }
    
    manifest_file = Path(args.out) / 'manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # 保存 summary_all.csv
    import csv
    summary_csv = Path(args.out) / 'summary_all.csv'
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seed', 'K', 'ticks', 'actual_samples', 'no_trade_ratio',
            'collapse_mean', 'collapse_std', 'collapse_p50', 'collapse_p95', 'collapse_max',
            'corr_collapse_H_price', 'corr_collapse_H_transfer',
            'high_collapse_count', 'duration_seconds'
        ])
        for s in all_summaries:
            writer.writerow([
                s['experiment']['seed'],
                s['experiment']['K'],
                s['experiment']['ticks'],
                s['experiment']['actual_samples'],
                f"{s['no_trade_ratio']:.4f}",
                f"{s['collapse_stats']['mean']:.4f}",
                f"{s['collapse_stats']['std']:.4f}",
                f"{s['collapse_stats']['p50']:.4f}",
                f"{s['collapse_stats']['p95']:.4f}",
                f"{s['collapse_stats']['max']:.4f}",
                f"{s['correlations'].get('collapse_vs_H_price', 0):.4f}",
                f"{s['correlations'].get('collapse_vs_H_transfer', 0):.4f}",
                s['high_collapse_analysis'].get('count', 0),
                f"{s['timing']['duration_seconds']:.1f}",
            ])
    
    print(f"\n{'='*60}")
    print(f"所有实验完成！")
    print(f"manifest: {manifest_file}")
    print(f"summary: {summary_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
