#!/usr/bin/env python3
"""
V5-Lab 阶段分析执行脚本
按优先级执行：P0 → P1 → P2 → P3
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import json
import numpy as np

# 设置线程为1（复现性）
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 添加当前目录到路径（experiments目录）
experiments_dir = Path(__file__).parent
sys.path.insert(0, str(experiments_dir))
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation_v2_fixed import ComplexSystemValidatorV2Fixed
from multi_scale_complexity import compute_complexity_sensitivity
from chaos_ablation import run_chaos_ablation_experiment

def generate_n_list(min_N=2, max_N=50, method='exponential'):
    """
    生成N sweep的N值列表
    
    Args:
        min_N: 最小N值
        max_N: 最大N值
        method: 'exponential' 或 'linear'
    
    Returns:
        N值列表
    """
    if method == 'exponential':
        # 指数间隔：2, 3, 5, 8, 13, 21, 34, ...
        N_list = [2, 3]
        while N_list[-1] < max_N:
            next_N = N_list[-1] + N_list[-2] if len(N_list) > 1 else N_list[-1] * 2
            if next_N <= max_N:
                N_list.append(next_N)
            else:
                break
        # 确保包含max_N
        if max_N not in N_list and max_N > N_list[-1]:
            N_list.append(max_N)
    else:
        # 线性间隔
        n_points = min(10, max_N - min_N + 1)
        N_list = np.linspace(min_N, max_N, n_points, dtype=int).tolist()
        N_list = sorted(set(N_list))
    
    return [int(N) for N in N_list if N >= min_N and N <= max_N]

def run_p0_nonlinearity(validator, seed, output_dir):
    """
    P0: 非线性测试 - N sweep
    """
    print(f"\n{'='*80}")
    print(f"P0: 非线性测试 (Seed {seed})")
    print(f"{'='*80}")
    
    # 先运行正常模拟以确定N范围
    print("  步骤1: 运行正常模拟确定N范围...")
    traj, N_hist, exp_hist, complexity_hist, metrics = validator.run_simulation_wrapper(seed=seed)
    
    if len(N_hist) > 0:
        min_N = max(2, int(np.min(N_hist)))
        max_N = min(200, int(np.max(N_hist)) + 10)
    else:
        min_N, max_N = 2, 50
    
    # 生成N列表（指数间隔）
    N_list = generate_n_list(min_N, max_N, method='exponential')
    print(f"  N范围: {min_N} → {max_N}, N列表: {N_list}")
    
    # 运行N sweep
    print("  步骤2: 执行N sweep...")
    sweep_ticks = min(100000, validator.ticks // 5)
    if sweep_ticks < 50000:
        sweep_ticks = 50000
    
    results_by_N = validator.run_n_sweep(seed, N_list, sweep_ticks=sweep_ticks)
    
    # 测试非线性
    print("  步骤3: 计算非线性得分...")
    nonlinearity_score = validator.test_nonlinearity_response_curve(
        results_by_N, seed, output_dir
    )
    
    # 检查输出文件
    response_file = output_dir / f'response_curve_{seed}.csv'
    diagnosis_file = output_dir / f'nonlinearity_diagnosis_{seed}.png'
    
    if response_file.exists() and diagnosis_file.exists():
        print(f"  ✅ P0完成: nonlinearity={nonlinearity_score:.4f}")
        print(f"     - {response_file}")
        print(f"     - {diagnosis_file}")
        return True, nonlinearity_score
    else:
        print(f"  ❌ P0失败: 输出文件缺失")
        return False, 0.0

def run_p1_reflexivity(validator, seed, output_dir):
    """
    P1: 反身性测试 - 结构分析
    """
    print(f"\n{'='*80}")
    print(f"P1: 反身性测试 (Seed {seed})")
    print(f"{'='*80}")
    
    # 运行正常模拟
    print("  运行正常模拟...")
    traj, N_hist, exp_hist, complexity_hist, metrics = validator.run_simulation_wrapper(seed=seed)
    
    # 测试反身性
    print("  计算反身性得分...")
    reflexivity_score, calibration_data = validator.test_reflexivity_population(
        N_hist, exp_hist, adjust_interval=2000
    )
    
    # 保存校准数据（转换numpy类型为Python原生类型）
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    calibration_data_clean = convert_numpy(calibration_data)
    calibration_file = output_dir / f'reflexivity_calibration_{seed}.json'
    with open(calibration_file, 'w') as f:
        json.dump(calibration_data_clean, f, indent=2)
    
    # 分析结果
    lag_scores = {
        'lag_1': calibration_data.get('lag_score', 0.0),
        'monotonicity': calibration_data.get('monotonicity', 0.0),
        'base_score': calibration_data.get('base_score', 0.0)
    }
    
    print(f"  ✅ P1完成: reflexivity={reflexivity_score:.4f}")
    print(f"     - lag分析: {lag_scores}")
    print(f"     - 单调性: {calibration_data.get('monotonicity', 0.0):.4f}")
    print(f"     - {calibration_file}")
    
    return True, reflexivity_score, calibration_data

def run_p2_multiscale(validator, seed, output_dir):
    """
    P2: 多尺度结构密度测试
    """
    print(f"\n{'='*80}")
    print(f"P2: 多尺度结构密度测试 (Seed {seed})")
    print(f"{'='*80}")
    
    # 运行正常模拟
    print("  运行正常模拟...")
    traj, N_hist, exp_hist, complexity_hist, metrics = validator.run_simulation_wrapper(seed=seed)
    
    # 计算多尺度complexity
    print("  计算多尺度complexity...")
    sensitivity_results = compute_complexity_sensitivity(traj)
    
    # 保存结果
    sensitivity_file = output_dir / f'complexity_multiscale_{seed}.json'
    with open(sensitivity_file, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                  for k, v in sensitivity_results.items()}, f, indent=2)
    
    overall_stability = sensitivity_results.get('overall_stability', 0.0)
    
    print(f"  ✅ P2完成: overall_stability={overall_stability:.4f}")
    print(f"     - {sensitivity_file}")
    
    return True, sensitivity_results

def run_p3_chaos_ablation(validator, seed, output_dir):
    """
    P3: 混沌因子消融测试
    """
    print(f"\n{'='*80}")
    print(f"P3: 混沌因子消融测试 (Seed {seed})")
    print(f"{'='*80}")
    
    ablation_ticks = min(50000, validator.ticks // 10)
    
    results = []
    
    # 强度扫描
    print("  强度扫描...")
    for strength in [0, 0.5, 1.0, 2.0]:
        print(f"    chaos_strength={strength}...", end=' ', flush=True)
        try:
            result = run_chaos_ablation_experiment(
                seed=seed,
                ticks=ablation_ticks,
                chaos_strength=strength,
                enable_dispersion=True,
                enable_entropy=True,
                enable_overload=True
            )
            results.append(result)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            continue
    
    # 分量消融
    print("  分量消融...")
    for config in [
        {'dispersion': True, 'entropy': False, 'overload': False},
        {'dispersion': False, 'entropy': True, 'overload': False},
        {'dispersion': False, 'entropy': False, 'overload': True},
    ]:
        config_name = '_'.join([k for k, v in config.items() if v])
        print(f"    {config_name}...", end=' ', flush=True)
        try:
            result = run_chaos_ablation_experiment(
                seed=seed,
                ticks=ablation_ticks,
                chaos_strength=1.0,
                enable_dispersion=config['dispersion'],
                enable_entropy=config['entropy'],
                enable_overload=config['overload']
            )
            result['config'] = config_name
            results.append(result)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            continue
    
    # 保存结果
    ablation_file = output_dir / f'chaos_ablation_{seed}.json'
    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_clean = [convert_numpy(r) for r in results]
    with open(ablation_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"  ✅ P3完成: {len(results)}个配置")
    print(f"     - {ablation_file}")
    
    return True, results

def main():
    parser = argparse.ArgumentParser(description='V5-Lab 阶段分析')
    parser.add_argument('--seeds', type=int, default=5, help='种子数量')
    parser.add_argument('--ticks', type=int, default=500000, help='每个种子的ticks数')
    parser.add_argument('--output', type=str, default='phase_analysis_output', help='输出目录')
    parser.add_argument('--phase', type=str, default='all', 
                       choices=['p0', 'p1', 'p2', 'p3', 'all'],
                       help='执行阶段')
    parser.add_argument('--seed-start', type=int, default=0, help='起始seed')
    
    args = parser.parse_args()
    
    # V5.2 审计：打印环境变量配置
    probe_count = int(os.environ.get('IG_PROBE_COUNT', '1'))
    debug_struct = os.environ.get('IG_DEBUG_STRUCT', '0') == '1'
    print(f"\n{'='*80}")
    print(f"V5.2 实验配置:")
    print(f"  IG_PROBE_COUNT (K) = {probe_count}")
    print(f"  IG_DEBUG_STRUCT = {debug_struct}")
    print(f"  seeds = {args.seed_start} to {args.seed_start + args.seeds - 1}")
    print(f"  ticks = {args.ticks}")
    print(f"  output = {args.output}")
    print(f"{'='*80}\n")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validator = ComplexSystemValidatorV2Fixed(args.seeds, args.ticks, str(output_dir))
    
    all_results = []
    
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        print(f"\n{'#'*80}")
        print(f"处理 Seed {seed}")
        print(f"{'#'*80}")
        
        seed_results = {'seed': seed}
        
        # P0: 非线性
        if args.phase in ['p0', 'all']:
            success, nonlinearity = run_p0_nonlinearity(validator, seed, output_dir)
            if not success:
                print(f"  ❌ P0失败，跳过后续阶段")
                continue
            seed_results['nonlinearity'] = nonlinearity
        
        # P1: 反身性
        if args.phase in ['p1', 'all']:
            success, reflexivity, calibration = run_p1_reflexivity(validator, seed, output_dir)
            if not success:
                print(f"  ❌ P1失败，跳过后续阶段")
                continue
            seed_results['reflexivity'] = reflexivity
            seed_results['reflexivity_monotonicity'] = calibration.get('monotonicity', 0.0)
        
        # P2: 多尺度
        if args.phase in ['p2', 'all']:
            success, sensitivity = run_p2_multiscale(validator, seed, output_dir)
            if not success:
                print(f"  ❌ P2失败，跳过后续阶段")
                continue
            seed_results['overall_stability'] = sensitivity.get('overall_stability', 0.0)
        
        # P3: 混沌因子消融
        if args.phase in ['p3', 'all']:
            success, ablation_results = run_p3_chaos_ablation(validator, seed, output_dir)
            if not success:
                print(f"  ❌ P3失败")
            seed_results['chaos_ablation_count'] = len(ablation_results) if success else 0
        
        all_results.append(seed_results)
    
    # 保存汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_file = output_dir / 'phase_analysis_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✅ 汇总结果已保存: {summary_file}")
        print("\n汇总统计:")
        print(summary_df.describe())

if __name__ == "__main__":
    main()
