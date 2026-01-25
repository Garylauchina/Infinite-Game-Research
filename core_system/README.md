# 核心系统代码说明

本目录包含 Infinite Game V5.2 的核心实现文件。

**当前版本**：V5.2（开发中）

**V5.2 新特性**：
- 并行探针机制（K probes per tick）
- 体验更新：失败不直接惩罚，通过 success_rate 和 dispersion 间接影响
- Chaos Factor：惩罚结构塌缩而非失败率

## 架构说明

**重要**：V5.0 采用统一模拟器架构，**不再使用 Exchange/Player 的严格分离**。

- **V0-V1**：Exchange/Player 严格分离，有订单簿、撮合引擎
- **V5.0**：统一模拟器（V5MarketSimulator），函数式规则，宏观聚合，无订单簿

## 文件说明

### main.py
主模拟器，包含 `V5MarketSimulator` 类：
- **统一管理**：统一管理所有逻辑（规则执行、状态更新、玩家管理）
- **V5.2 新参数**：`probe_count`（K值）、`probe_count_random`（是否随机化K）
- 初始3个玩家（RandomExperiencePlayer）
- **V5.2 新方法**：`sample_probes()` - 多探针成交判定
- 使用 `chaos_rules.compute_match_prob` 计算成交概率
- 混乱因子管理器 `ChaosFactor`（V5.2支持collapse_proxy）
- 参与调整阈值：ADD_PLAYER_THRESHOLD = 0.35, REMOVE_PLAYER_THRESHOLD = 0.15
- 结构密度计算器 `StructureMetrics`
- 复杂度计算频率：每500 ticks

### random_player.py
`RandomExperiencePlayer` 类（市场先生的分身）：
- **V5.2 新参数**：`probe_count`（K值）、`probe_count_random`
- **V5.2 新方法**：`generate_probes()` - 生成K个并行探针
- **V5.2 新数据类**：`ProbeResult` - 探针结果聚合
- 纯随机报价（价格范围：40000-60000）
- **V5.2 体验更新**：`instant_reward = w₁*success_rate + w₂*match_dispersion + w₃*complexity`
  - 失败不直接惩罚！通过 success_rate 间接影响
  - 权重：w₁=0.4, w₂=0.2, w₃=0.4
- EMA平滑：0.98 * old + 0.02 * new（保持不变）
- 体验历史记录：最近50次行动和体验

### state_engine.py
`StateEngine` 类和 `MarketState` 数据类：
- 5维状态向量：price_norm, volatility, liquidity, imbalance, complexity
- 价格更新：new_price_abs = price_abs * (1 + 0.0005 * pressure)
- 波动率：20-tick窗口，标准差年化近似
- 流动性：成交率（len(matches) / len(actions)）
- 不平衡度：买卖压力归一化到 [0,1]

### trading_rules.py
最简规则集：
- 价格边界：40000-60000
- 固定手续费：0.05%
- 基础成交概率计算（但实际使用的是 chaos_rules.py 中的版本）

### chaos_rules.py
混乱因子规则集（**实际使用的成交概率计算**）：
- **V5.2 新增**：`compute_collapse_proxy()` - 计算结构塌缩代理
  - 公式：`collapse = |2 * H_norm - 1|`
  - 高熵区分：H_norm > 0.8 时，用 `active_protocols_ratio` 区分噪声与健康
- `compute_chaos_factor`：计算报价分散度、方向熵、规模过载
  - 报价分散度：相对分散度（2%为健康基准）
  - 方向熵：买卖方向分布的熵
  - 规模过载：行动过载和玩家过载的加权
- `ChaosFactor` 类：动态混乱因子管理器
  - 基础混乱因子：0.08
  - **V5.2 新增**：`collapse_sensitivity` 参数（γ，塌缩敏感度）
  - **V5.2 新增**：`set_match_prices()`、`set_active_protocols_ratio()` 方法
  - **V5.2 公式**：`chaos = base_chaos * (1 + γ * collapse_proxy) * multiplier`
  - 动态调整：根据玩家数量、平均体验、时间衰减、**结构塌缩**调整
- `compute_match_prob`：完整统计版成交概率
  - 价格优先（40%权重）：指数衰减，exp(-4 * price_dist)
  - 混乱惩罚（动态调整）：最大40%惩罚
  - 状态修正：流动性提升（+30%）、波动率调整（中等波动最佳）

### metrics.py
结构密度计算模块：
- `StructureMetrics` 类
- K-means聚类（窗口大小5000，聚类更新间隔500，n_init=3）
- 优化：使用warm start（缓存上次聚类中心），降低计算频率
- 复杂度计算：0.4 * protocol_score + 0.4 * transfer_entropy + 0.2 * uniformity
  - protocol_score：活跃协议数 / 总聚类数（>2%活跃）
  - transfer_entropy：状态转移矩阵的熵（归一化）
  - uniformity：驻留均匀度（1 - 最大驻留比例）

## 关键实现细节

### 成交概率计算流程
实际使用的是 `chaos_rules.compute_match_prob`，包含：
1. **价格优先**（40%权重）：`price_priority = 0.98 * exp(-4 * price_dist)`
2. **混乱惩罚**（动态调整）：
   - 基础混乱因子：`base_chaos = compute_chaos_factor(actions, player_count)`
   - 动态调整：`adjusted_chaos = base_chaos * (0.08/0.15) * chaos_multiplier`
   - 惩罚：`chaos_penalty = 1.0 - 0.4 * adjusted_chaos`（最大40%惩罚）
3. **状态修正**：
   - 流动性提升：`+0.3 * liquidity`
   - 波动率调整：`-0.1 * (volatility - 0.5)^2`（中等波动最佳）
4. **最终概率**：`final_prob = price_priority * chaos_penalty + liquidity_boost + vol_adjust`，限制在 [0.01, 0.99]

### 混乱因子计算
`compute_chaos_factor` 包含三个分量：
1. **报价分散度**（40%权重）：`std_price / mean_price / 0.02`，限制在 [0.2, 3.0]
2. **方向熵**（35%权重）：买卖方向分布的熵，归一化到 [0,1]
3. **规模过载**（25%权重）：`0.6 * action_overload + 0.4 * player_overload`
   - action_overload = tick_actions / 8.0
   - player_overload = log1p(player_count / 10)

### 动态混乱因子调整
`ChaosFactor.get_chaos_multiplier` 根据以下因素调整：
- **时间衰减**：`decay = 0.95^(tick//10000)`
- **拥挤惩罚**：玩家数>15时×1.2，<5时×0.8
- **体验调整**：平均体验<0.3时×0.7
- **返回乘数**：限制在 [0.5, 1.5]

### 参与调整机制
`adjust_participation` 根据平均体验调整玩家数量：
- **增加玩家**：`avg_exp > 0.35` 且 `player_count < MAX_N`
- **减少玩家**：`avg_exp < 0.15` 且 `player_count > 2`
- **最小玩家保护**：`player_count <= 2` 且 `avg_exp > 0.25` 时强制加人

### 体验更新公式（V5.2）
`RandomExperiencePlayer.update_experience` 计算即时体验：

**V5.2 公式**（使用 ProbeResult）：
```python
# 失败不直接惩罚！
instant_reward = (
    0.4 * success_rate +      # 成功率 [0,1]
    0.2 * match_dispersion +  # 成交分布离散度 [0,1]
    0.4 * complexity          # 世界结构密度 [0,1]
)
```

**V5.0 兼容公式**（单探针模式）：
```python
instant_experience = (
    +1.0 if matched else -0.3  # 成交快感 / 未成交挫败
    -10.0 * fee_rate            # 费用痛苦（费率惩罚）
    +0.4 * complexity          # 结构奖励
    +0.2 * volatility          # 波动奖励
    +0.1 * liquidity           # 流动性奖励
)
```
然后通过EMA平滑：`experience_score = 0.98 * old + 0.02 * new`

### 结构密度计算流程
1. **轨迹更新**：每tick更新状态特征 `[price_norm, volatility, liquidity, imbalance]`
2. **聚类更新**：每500 ticks（且数据>=200）执行K-means聚类
   - 使用warm start（缓存上次聚类中心）
   - n_init=3（首次）或1（warm start）
3. **复杂度计算**：
   - 协议分数：活跃协议数（>2%驻留） / 总聚类数
   - 转移熵：状态转移矩阵的熵，归一化到 [0,1]
   - 均匀度：1 - 最大驻留比例
   - 综合：`0.4 * protocol_score + 0.4 * transfer_entropy + 0.2 * uniformity`

## 数据流

### V5.2 数据流
```
每个Tick：
1. 更新 ChaosFactor（成交价格历史 + 活跃簇占比）
2. 所有玩家生成K个探针 → all_probes
3. 多探针成交判定（每个探针独立判定）→ probe_results
4. 设置 ProbeResult 并更新体验 → experience_score
5. 状态更新（价格、波动率、流动性、不平衡度）→ s_{t+1}
6. 更新结构密度轨迹 → structure_metrics
7. 每500 ticks：计算复杂度 → complexity
8. 每adjust_interval ticks：调整玩家数量
```

### V5.0 数据流（兼容）
```
每个Tick：
1. 所有玩家报价（纯随机）→ actions
2. 计算成交概率（价格优先 + 混乱惩罚 + 状态修正）→ matches
3. 玩家获得反馈（成交/未成交 + 费用 + 结构奖励）→ experience_score
4. 状态更新（价格、波动率、流动性、不平衡度）→ s_{t+1}
5. 更新结构密度轨迹 → structure_metrics
6. 每500 ticks：计算复杂度 → complexity
7. 每adjust_interval ticks：调整玩家数量
```

## 关键参数

- **价格范围**：40000-60000
- **固定手续费**：0.05%
- **初始玩家数**：3
- **参与调整间隔**：1000 ticks（可配置）
- **复杂度计算间隔**：500 ticks
- **结构密度窗口**：5000 ticks
- **聚类数**：5
- **聚类更新间隔**：500 ticks
- **增加玩家阈值**：0.35
- **减少玩家阈值**：0.15
- **基础混乱因子**：0.08
- **混乱惩罚强度**：最大40%

## 性能优化

1. **复杂度计算频率降低**：从每100 ticks改为每500 ticks
2. **聚类更新频率降低**：从每100 ticks改为每500 ticks
3. **K-means优化**：
   - n_init从10降到3（首次）或1（warm start）
   - 使用warm start（缓存上次聚类中心）
   - max_iter=100

## V5.2 关键公式

### 并行探针机制
```
每个 Player 每 tick:
  K = probe_count  # K ≥ 1，固定或弱随机
  for i in 1..K:
    probe_i = generate_probe()  # 随机价格、随机方向、size=1
    success_i ~ Bernoulli(p_match(probe_i, state))
```

### 体验更新（V5.2）
```
# 失败不直接惩罚！
instant_reward = w₁ * success_rate + w₂ * match_dispersion + w₃ * complexity

其中：
  success_rate = n_success / K
  match_dispersion = std(match_prices) / price_range
  w₁ = 0.4, w₂ = 0.2, w₃ = 0.4
```

### 结构塌缩代理（V5.2）
```
collapse = |2 * H_norm - 1|

if H_norm > 0.8:
    collapse *= active_protocols_ratio
    # 高活跃 → 均匀噪声（塌缩）
    # 低活跃 → 多结构有骨架（健康）
```

### Chaos Factor（V5.2）
```
chaos = base_chaos * (1 + γ * collapse_proxy) * multiplier
```

## 版本信息

- **当前版本**：V5.2（开发中）
- **基准版本**：V5.0（main分支，已锁定）
- **最后更新**：2026-01-25
