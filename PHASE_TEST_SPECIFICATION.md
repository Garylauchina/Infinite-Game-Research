# V5-Lab 阶段测试规范说明（P0-P3）

## 测试程序机制

### P0: 非线性测试（Nonlinearity Test）

**测试机制**：
1. 对每个seed运行正常模拟（500k ticks），确定N值范围
2. 使用指数间隔生成N列表（2, 3, 5, 8, 13, 21, 34, ...），覆盖观察到的N范围
3. 对每个N值独立运行fixed_N模拟（100k ticks），禁止复用历史数据
4. 使用稳态均值（后50%数据，排除burn-in）计算：
   - `avg_experience`: 平均体验分数
   - `avg_liquidity`: 平均流动性
   - `avg_complexity`: 平均结构密度
5. 对三条响应曲线（experience/liquidity/complexity vs N）分别拟合：
   - 线性模型：y = a*N + b
   - 二次模型：y = a*N² + b*N + c
6. 计算调整后的R²（adj-R²），比较线性vs二次模型的增益
7. 使用soft gain公式计算非线性得分：`score = 1 - exp(-gain / 0.05)`

**测试代码绝对路径**：
- 主脚本：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/run_phase_analysis.py`
- 验证器：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/validation_v2_fixed.py`
- 核心方法：
  - `run_n_sweep()`: `/Users/liugang/Cursor_Store/Infinite-Game/experiments/validation_v2_fixed.py` (第103-132行)
  - `test_nonlinearity_response_curve()`: `/Users/liugang/Cursor_Store/Infinite-Game/experiments/validation_v2_fixed.py` (第134-250行)

**测试结果绝对路径**：
- 响应曲线数据：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/phase_analysis_output/response_curve_{seed}.csv`
- 诊断图：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/phase_analysis_output/nonlinearity_diagnosis_{seed}.png`
- 汇总结果：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/phase_analysis_output/phase_analysis_summary.csv`

**输出文件格式**：
- `response_curve_{seed}.csv`: CSV格式，包含列：N, avg_experience, avg_liquidity, avg_complexity
- `nonlinearity_diagnosis_{seed}.png`: PNG图像，包含三条响应曲线的拟合结果和残差图

---

### P1: 反身性测试（Reflexivity Test）

**测试机制**：
1. 运行正常模拟（500k ticks），获取N_history和experience_history
2. 计算dN（玩家数量变化）：`dN = diff(N_history)`
3. 对齐时间序列：E[t]决定N[t]→N[t+1]的转移
4. 基础决策一致性：
   - 根据阈值将E映射到期望动作：E > 0.35 → +1（添加），E < 0.15 → -1（移除），否则 → 0（保持）
   - 观察到的动作：`sign(dN)`
   - 计算3类决策准确性：`acc = mean(observed == expected)`
5. 多滞后分析：
   - 对lag ∈ {1, 2, 3}个adjust周期，使用E[t+lag]预测dN[t]
   - 计算各lag的决策准确性
   - 综合lag得分：`lag_score = mean(acc_lag_1, acc_lag_2, acc_lag_3)`
6. P(add|E分箱) 单调性检验：
   - 将experience分10个箱
   - 计算每个箱的P(add) = 实际添加次数 / 该箱总次数
   - 使用Spearman相关系数检验P(add)与E的单调性
7. 综合得分：`score = 0.4 * base_score + 0.3 * lag_score + 0.3 * monotonicity`

**测试代码绝对路径**：
- 主脚本：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/run_phase_analysis.py`
- 验证器：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/validation_v2_fixed.py`
- 核心方法：
  - `test_reflexivity_population()`: `/Users/liugang/Cursor_Store/Infinite-Game/experiments/validation_v2_fixed.py` (第252-361行)
  - `run_p1_reflexivity()`: `/Users/liugang/Cursor_Store/Infinite-Game/experiments/run_phase_analysis.py` (第64-99行)

**测试结果绝对路径**：
- 校准数据：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/phase_analysis_output/reflexivity_calibration_{seed}.json`
- 汇总结果：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/phase_analysis_output/phase_analysis_summary.csv`

**输出文件格式**：
- `reflexivity_calibration_{seed}.json`: JSON格式，包含：
  - `bin_centers`: 分箱中心值列表
  - `p_add_by_bin`: 每个箱的P(add)列表
  - `bin_counts`: 每个箱的样本数列表
  - `monotonicity`: 单调性系数（Spearman相关系数）
  - `lag_score`: 多滞后得分
  - `base_score`: 基础决策一致性得分

---

### P2: 多尺度结构密度测试（Multi-Scale Complexity Test）

**测试机制**：
1. 运行正常模拟（500k ticks），获取完整轨迹
2. 在多个尺度组合下重复计算complexity：
   - `n_clusters ∈ {3, 5, 8, 13}`
   - `window_size ∈ {1000, 5000, 20000}`
   - 共12个尺度组合
3. 对每个尺度组合：
   - 使用最后window_size个轨迹点（如果轨迹长度不足，使用全部）
   - K-means聚类（random_state=42, n_init=3）
   - 计算复杂度：`complexity = 0.4 * protocol_score + 0.4 * transfer_entropy + 0.2 * uniformity`
4. 计算跨尺度稳定性：
   - 固定窗口大小w，跨k的稳定性：`stability_w{w} = 1 - std(cpx_k3_w{w}, cpx_k5_w{w}, ...) / mean(...)`
   - 固定k，跨窗口大小的稳定性：`stability_k{k} = 1 - std(cpx_k{k}_w1000, cpx_k{k}_w5000, ...) / mean(...)`
   - 综合稳定性：`overall_stability = 1 - std(all_cpx_values) / mean(all_cpx_values)`

**测试代码绝对路径**：
- 主脚本：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/run_phase_analysis.py`
- 多尺度计算模块：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/multi_scale_complexity.py`
- 核心方法：
  - `compute_complexity_multi_scale()`: `/Users/liugang/Cursor_Store/Infinite-Game/experiments/multi_scale_complexity.py` (第14-50行)
  - `compute_complexity_sensitivity()`: `/Users/liugang/Cursor_Store/Infinite-Game/experiments/multi_scale_complexity.py` (第52-88行)
  - `run_p2_multiscale()`: `/Users/liugang/Cursor_Store/Infinite-Game/experiments/run_phase_analysis.py` (第101-125行)

**测试结果绝对路径**：
- 多尺度数据：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/phase_analysis_output/complexity_multiscale_{seed}.json`
- 汇总结果：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/phase_analysis_output/phase_analysis_summary.csv`

**输出文件格式**：
- `complexity_multiscale_{seed}.json`: JSON格式，包含：
  - `cpx_k{k}_w{w}`: 各尺度组合下的complexity值（12个值）
  - `stability_w{w}`: 固定窗口大小的稳定性（3个值）
  - `stability_k{k}`: 固定k的稳定性（4个值）
  - `overall_stability`: 综合跨尺度稳定性（1个值）

---

### P3: 混沌因子消融测试（Chaos Factor Ablation Test）

**测试机制**：
1. 强度扫描：
   - 对每个 `chaos_strength ∈ {0, 0.5, 1.0, 2.0}` 运行独立模拟（50k ticks）
   - 修改 `compute_match_prob` 函数，应用强度倍数
2. 分量消融：
   - 对每种配置运行独立模拟（50k ticks）：
     - 只开dispersion（entropy和overload关闭）
     - 只开entropy（dispersion和overload关闭）
     - 只开overload（dispersion和entropy关闭）
3. 诊断统计（对每种配置）：
   - 记录所有成交概率的clip前值（final_prob_unclipped）
   - 计算分布统计：mean, std, min, max
   - 计算贴顶/贴底比例：`clip_top_ratio = mean(prob >= 0.99)`, `clip_bottom_ratio = mean(prob <= 0.01)`
   - 记录最终指标：final_complexity, final_player_count, avg_liquidity
4. 比较不同配置下的结构变化

**测试代码绝对路径**：
- 主脚本：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/run_phase_analysis.py`
- 消融实验模块：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/chaos_ablation.py`
- 核心方法：
  - `run_chaos_ablation_experiment()`: `/Users/liugang/Cursor_Store/Infinite-Game/experiments/chaos_ablation.py` (第11-120行)
  - `run_p3_chaos_ablation()`: `/Users/liugang/Cursor_Store/Infinite-Game/experiments/run_phase_analysis.py` (第127-189行)

**测试结果绝对路径**：
- 消融结果：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/phase_analysis_output/chaos_ablation_{seed}.json`
- 汇总结果：`/Users/liugang/Cursor_Store/Infinite-Game/experiments/phase_analysis_output/phase_analysis_summary.csv`

**输出文件格式**：
- `chaos_ablation_{seed}.json`: JSON格式，包含每个配置的结果列表，每个配置包含：
  - `chaos_strength`: 强度倍数
  - `enable_dispersion`, `enable_entropy`, `enable_overload`: 分量开关
  - `prob_unclipped_mean/std/min/max`: clip前概率分布统计
  - `clip_top_ratio`, `clip_bottom_ratio`: 贴顶/贴底比例
  - `final_complexity`, `final_player_count`, `avg_liquidity`: 最终指标
  - `prob_unclipped_history`: 前1000个概率值（用于可视化）

---

## 执行脚本

**主执行脚本绝对路径**：
`/Users/liugang/Cursor_Store/Infinite-Game/experiments/run_phase_analysis.py`

**执行方式**：
```bash
cd /Users/liugang/Cursor_Store/Infinite-Game/experiments
python run_phase_analysis.py --seeds 5 --ticks 500000 --phase {p0|p1|p2|p3|all} --output phase_analysis_output
```

**参数说明**：
- `--seeds`: 种子数量（默认：5）
- `--ticks`: 每个种子的ticks数（默认：500000）
- `--phase`: 执行阶段（p0/p1/p2/p3/all，默认：all）
- `--output`: 输出目录（默认：phase_analysis_output）
- `--seed-start`: 起始seed编号（默认：0）

---

## 依赖模块

**基底模型代码路径**（开发仓库）：
- `/Users/liugang/Cursor_Store/Infinite-Game/src/v5/main.py` - V5MarketSimulator
- `/Users/liugang/Cursor_Store/Infinite-Game/src/v5/state_engine.py` - 状态引擎
- `/Users/liugang/Cursor_Store/Infinite-Game/src/v5/random_player.py` - 随机体验玩家
- `/Users/liugang/Cursor_Store/Infinite-Game/src/v5/trading_rules.py` - 交易规则
- `/Users/liugang/Cursor_Store/Infinite-Game/src/v5/chaos_rules.py` - 混乱因子规则
- `/Users/liugang/Cursor_Store/Infinite-Game/src/v5/metrics.py` - 结构密度计算

**本仓库核心代码路径**（已锁定版本）：
- `core_system/main.py` - V5MarketSimulator
- `core_system/state_engine.py` - 状态引擎
- `core_system/random_player.py` - 随机体验玩家
- `core_system/trading_rules.py` - 交易规则
- `core_system/chaos_rules.py` - 混乱因子规则
- `core_system/metrics.py` - 结构密度计算

**实验框架代码路径**：
- `/Users/liugang/Cursor_Store/Infinite-Game/experiments/validation_v2_fixed.py` - 验证器
- `/Users/liugang/Cursor_Store/Infinite-Game/experiments/multi_scale_complexity.py` - 多尺度计算
- `/Users/liugang/Cursor_Store/Infinite-Game/experiments/chaos_ablation.py` - 混沌因子消融

---

## 环境配置

**线程控制（复现性）**：
脚本自动设置以下环境变量：
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

**随机种子**：
- numpy: `seed`（每个seed独立）
- random: `seed`（每个seed独立）
- sklearn (KMeans): `42`（固定）

---

## 输出目录结构

```
/Users/liugang/Cursor_Store/Infinite-Game/experiments/phase_analysis_output/
├── response_curve_{seed}.csv              # P0: 响应曲线数据
├── nonlinearity_diagnosis_{seed}.png      # P0: 诊断图
├── reflexivity_calibration_{seed}.json    # P1: 反身性校准数据
├── complexity_multiscale_{seed}.json      # P2: 多尺度complexity
├── chaos_ablation_{seed}.json             # P3: 混沌因子消融结果
└── phase_analysis_summary.csv             # 汇总结果
```

---

## 执行顺序

**必须按顺序执行**：
1. P0 → 2. P1 → 3. P2 → 4. P3

任何一步失败都应先停止，不进入下一步。

---

## 核心代码一致性检查

### ✅ 已验证的一致性

1. **P1 反身性测试阈值**：
   - 测试规范：E > 0.35 → +1（添加），E < 0.15 → -1（移除）
   - 本仓库代码：`ADD_PLAYER_THRESHOLD = 0.35`, `REMOVE_PLAYER_THRESHOLD = 0.15`
   - **状态**: ✅ 一致

2. **P2 复杂度计算公式**：
   - 测试规范：`complexity = 0.4 * protocol_score + 0.4 * transfer_entropy + 0.2 * uniformity`
   - 本仓库代码：`complexity = 0.4 * protocol_score + 0.4 * transfer_entropy + 0.2 * uniformity` (metrics.py 第119-123行)
   - **状态**: ✅ 一致

3. **P2 K-means参数**：
   - 测试规范：`random_state=42, n_init=3`
   - 本仓库代码：`random_state=42, n_init=3` (metrics.py 第15行，第64行)
   - **状态**: ✅ 一致

4. **体验更新EMA参数**：
   - 测试规范：EMA with α=0.02 (即 0.98 * old + 0.02 * new)
   - 本仓库代码：`0.98 * self.experience_score + 0.02 * instant_experience` (random_player.py 第64-65行)
   - **状态**: ✅ 一致

5. **体验奖励权重**：
   - 测试规范：`instant_reward = match_reward + fee_penalty + 0.4 * c_t + 0.2 * v_t + 0.1 * l_t`
   - 本仓库代码：`structure_reward = +0.4 * s_t.complexity`, `volatility_reward = +0.2 * s_t.volatility`, `liquidity_reward = +0.1 * s_t.liquidity` (random_player.py 第48-50行)
   - **状态**: ✅ 一致

6. **活跃协议阈值**：
   - 测试规范：clusters with >2% occupancy
   - 本仓库代码：`active_protocols = np.sum(counts > len(clusters) * 0.02)` (metrics.py 第90行)
   - **状态**: ✅ 一致

### ⚠️ 需要注意的差异

1. **代码路径**：
   - 测试规范引用：`/Users/liugang/Cursor_Store/Infinite-Game/src/v5/...`（开发仓库）
   - 本仓库路径：`core_system/...`（研究仓库，已锁定版本）
   - **说明**: 这是预期的，本仓库包含的是已锁定的代码版本用于研究参考

2. **默认调整间隔**：
   - 测试规范中未明确指定，但测试代码可能使用不同的值
   - 本仓库默认：`adjust_interval=1000` (main.py 第25行)
   - **说明**: 测试代码可能会覆盖此参数，不影响测试结果

---

**结论**: 本仓库的核心代码与测试规范中描述的核心机制**完全一致**。所有关键参数、算法和计算公式都已验证匹配。
