# 核心系统代码说明

本目录包含 Infinite Game V5.0 的核心实现文件。

## 架构说明

**重要**：V5.0 采用统一模拟器架构，**不再使用 Exchange/Player 的严格分离**。

- **V0-V1**：Exchange/Player 严格分离，有订单簿、撮合引擎
- **V5.0**：统一模拟器（V5MarketSimulator），函数式规则，宏观聚合，无订单簿

## 文件说明

### main.py
主模拟器，包含 `V5MarketSimulator` 类：
- **统一管理**：统一管理所有逻辑（规则执行、状态更新、玩家管理）
- 初始3个玩家（RandomExperiencePlayer）
- 使用 `chaos_rules.compute_match_prob` 计算成交概率
- 混乱因子管理器 `ChaosFactor`
- 参与调整阈值：ADD_PLAYER_THRESHOLD = 0.35, REMOVE_PLAYER_THRESHOLD = 0.15
- 结构密度计算器 `StructureMetrics`

### random_player.py
`RandomExperiencePlayer` 类（市场先生的分身）：
- 纯随机报价（价格范围：40000-60000）
- 体验更新：match_reward + fee_penalty + structure_reward + volatility_reward + liquidity_reward
- EMA平滑：0.98 * old + 0.02 * new

### state_engine.py
`StateEngine` 类和 `MarketState` 数据类：
- 5维状态向量：price_norm, volatility, liquidity, imbalance, complexity
- 价格更新：new_price_abs = price_abs * (1 + 0.0005 * pressure)
- 波动率：20-tick窗口
- 流动性：成交率

### trading_rules.py
最简规则集：
- 价格边界：40000-60000
- 固定手续费：0.05%
- 基础成交概率计算（但实际使用的是 chaos_rules.py 中的版本）

### chaos_rules.py
混乱因子规则集：
- `compute_chaos_factor`：计算报价分散度、方向熵、规模过载
- `ChaosFactor` 类：动态混乱因子管理器
- `compute_match_prob`：完整统计版成交概率（实际使用的版本）

### metrics.py
结构密度计算模块：
- `StructureMetrics` 类
- K-means聚类（窗口大小5000，聚类更新间隔500）
- 复杂度计算：0.4 * protocol_score + 0.4 * transfer_entropy + 0.2 * uniformity

## 关键实现细节

### 成交概率计算
实际使用的是 `chaos_rules.compute_match_prob`，包含：
- 价格优先（40%权重）
- 混乱惩罚（动态调整）
- 状态修正（流动性、波动率）

### 混乱因子
- 基础混乱因子：0.08（从0.15降低）
- 动态调整：根据玩家数量、平均体验调整
- 惩罚强度：最大40%（从60%降低）

### 参与调整
- 增加玩家阈值：0.35（从0.7降低）
- 减少玩家阈值：0.15（从0.3降低）
- 最小玩家保护机制：≤2个玩家且体验>0.25时强制加人

### 体验更新
- match_reward: +1.0 (成交) / -0.3 (未成交)
- fee_penalty: -10.0 * fee_rate
- structure_reward: +0.4 * complexity
- volatility_reward: +0.2 * volatility
- liquidity_reward: +0.1 * liquidity

### 结构密度计算
- 窗口大小：5000 ticks
- 聚类数：5
- 聚类更新间隔：500 ticks
- 复杂度权重：协议数40% + 转移熵40% + 均匀度20%
