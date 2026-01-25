# Infinite Game v5.2 开发计划

> **分支**: v5.2  
> **状态**: 开发中  
> **创建日期**: 2026-01-25

---

## 🎯 版本目标

在不引入智能 agent、不引入显式订单簿、不引入回溯机制的前提下，为市场世界增加一个**交互阻抗（Interaction Impedance）**维度，使结构丰度提升，而不破坏 P0-P3 的理论前提。

**核心理念**：v5.2 并没有让 agent 更聪明，而是让世界对"行为强度"的反馈更真实。

---

## 🚧 设计边界（必须遵守）

本次修改必须满足以下约束：

- [ ] ❌ 不引入策略学习
- [ ] ❌ 不引入持久订单（每 tick 自动撤单）
- [ ] ❌ 不引入历史回溯或状态依赖
- [ ] ❌ 不引入外部接口字段
- [ ] ❌ 不改变 Market Mr. 的角色定义
- [ ] ❌ 不破坏现有 complexity / reflexivity 的定义逻辑
- [ ] ❌ 不调整 P0-P3 的定义
- [ ] ❌ 不调整 Complexity 的计算方式（暂时保持）

**设计原则**：只在 f(s') 内部增加一个维度，而不是改变系统阶数。

---

## 🔑 关键设计决策（三点澄清）

### 1. K 的单一语义：行为采样密度

**K 只有一个含义**：行为采样密度（Behavior Sampling Density）

- **是什么**：Player 在单个 tick 内对世界进行试探的次数
- **不是什么**：
  - ❌ 不是订单厚度（order depth）
  - ❌ 不是持仓量（position size）
  - ❌ 不是交易频率（trading frequency，那是跨 tick 的概念）
  - ❌ 不是市场深度（market depth）

**语义解释**：
```
K = 1  → 弱试探：每个 Player 每 tick 只问世界一次"这个价格行不行"
K > 1  → 强试探：每个 Player 每 tick 问世界 K 次"这些价格行不行"
```

**宏观等价**：K 个探针 ≈ "更密集地试探某个价格区间"，但微观上仍是独立的 unit probe。

**参数设计**：
- 默认 K = 1（与 v5.0 行为一致）
- K 可配置为固定值或弱随机（如 K ~ Uniform(1, K_max)）
- K 不依赖任何状态，不引入适应性

### 2. 失败不作为 reward 的直接符号项

**核心原则**：失败（n_failure）**不直接**出现在 reward 公式中作为惩罚项。

**原因**：
- 直接惩罚失败 → 隐含假设"失败=坏" → 破坏"失败也是结构"的设计理念
- 失败的"好坏"取决于失败的**分布形态**，而非失败的**数量**

**正确做法**：失败通过**结构特征**间接影响 reward

```
❌ 错误：instant_reward = α * n_success - β * n_failure    # 失败直接惩罚
✅ 正确：instant_reward = f(success_rate, match_dispersion, world_complexity)
```

**具体公式**：
```
instant_reward = w₁ * success_rate           # 成功率（0-1）
               + w₂ * match_dispersion       # 成交分布的离散度
               + w₃ * complexity             # 当前世界结构密度

其中：
  success_rate = n_success / (n_success + n_failure)
  match_dispersion = 成交价格的标准差 / 价格范围（归一化）
```

**设计含义**：
- 如果失败集中在某个价格区间（硬区间），dispersion 低但 pattern 清晰 → 体验不应下降
- 如果失败均匀分布（无结构噪声），dispersion 高但无 pattern → 体验可能下降
- 失败的"结构性"通过 dispersion 和 complexity 间接体现

### 3. 结构塌缩的最小可计算 proxy

**定义**：结构塌缩（Structure Collapse）= 世界状态从"有结构"退化为"均匀噪声"或"单点吸引子"

**最小可计算 proxy**：

```
collapse_proxy = |2 * H_norm - 1|

其中：
  H = -Σ pᵢ * log(pᵢ)           # 成交价格分布的熵
  H_max = log(n_bins)            # 均匀分布时的最大熵
  H_norm = H / H_max             # 归一化熵（0-1）
```

**直觉解释**：
| H_norm | 含义 | collapse_proxy |
|--------|------|----------------|
| ≈ 0 | 所有成交集中在一个价格（单点吸引子） | ≈ 1（高塌缩） |
| ≈ 0.5 | 成交分布在有限的几个区间（有结构） | ≈ 0（低塌缩） |
| ≈ 1 | 成交均匀分布（候选塌缩，需进一步区分） | ≈ 1（待第二判定） |

**补充判定：区分均匀噪声与健康多结构**

当 H_norm 高时（≈ 1），用 `active_protocols_ratio` 区分：

```
active_protocols_ratio = n_active_clusters / K_total

其中：
  n_active_clusters = 驻留比例 > 2% 的簇数
  K_total = 总簇数
```

**语义**：
- **active_protocols_ratio 高** → 很多簇都活跃 → **均匀噪声**（到处跳，无骨架）→ **塌缩**
- **active_protocols_ratio 低/中** → 只有部分簇活跃 → **多结构有骨架** → **健康**

**完整判定表**：
| H_norm | active_protocols_ratio | 结论 |
|--------|------------------------|------|
| ≈ 0 | 任意 | 塌缩（单点吸引子） |
| ≈ 0.5 | 任意 | 健康（有结构） |
| ≈ 1 | 高 | 塌缩（均匀噪声） |
| ≈ 1 | 低/中 | 健康（多结构/多骨架） |

**完整塌缩判定公式**：
```
collapse = |2 * H_norm - 1|

if H_norm > 0.8:
    # active_protocols_ratio 高 → 均匀噪声塌缩 → collapse 保持高
    # active_protocols_ratio 低 → 多结构非噪声 → collapse 被压低
    collapse *= active_protocols_ratio
```

**实现细节**：
```python
def compute_collapse_proxy(match_prices: List[float], 
                           active_protocols_ratio: float,
                           n_bins: int = 10) -> float:
    """
    计算结构塌缩代理指标（含高熵区分）
    
    参数：
      active_protocols_ratio: 活跃簇占比（驻留>2%的簇数 / 总簇数）
                              高 → 均匀噪声（塌缩）
                              低 → 多结构有骨架（健康）
    """
    if len(match_prices) < 2:
        return 1.0  # 无数据时视为塌缩
    
    # 计算价格分布直方图
    hist, _ = np.histogram(match_prices, bins=n_bins, density=True)
    hist = hist + 1e-10  # 避免 log(0)
    hist = hist / hist.sum()  # 归一化为概率
    
    # 计算归一化熵
    H = -np.sum(hist * np.log(hist))
    H_norm = H / np.log(n_bins)
    
    # 基础塌缩 proxy
    collapse = abs(2 * H_norm - 1)
    
    # 高熵时：用 active_protocols_ratio 区分噪声与健康
    if H_norm > 0.8:
        collapse *= active_protocols_ratio  # 高→保持塌缩，低→压成健康
    
    return collapse
```

---

## 📋 开发任务清单

### Phase 1: 核心概念实现

#### 1.1 交互阻抗（Interaction Impedance）概念引入

- [ ] 在 `core_system/` 中设计交互阻抗的内部数据结构
- [ ] 定义交互阻抗的计算逻辑：世界对"行为强度"的响应曲线
- [ ] 注意：这不是滑点、不是成交失败、不是订单簿

#### 1.2 Player 行为改造：并行探针机制

- [ ] 修改 `random_player.py`：保持"零策略"基础
  - [ ] 保持：随机 buy/sell/none
  - [ ] 保持：随机报价位置分布
  - [ ] 保持：单位 size = 1
- [ ] 新增：每个 Player 每 tick 生成 K 次"并行报价探针"
  - [ ] K ≥ 1，固定或弱随机
  - [ ] K 是"行为强度的采样次数"，不是订单厚度
  - [ ] 多个探针在同一 tick 内独立评估
- [ ] 设计 K 参数的配置方式

### Phase 2: 成交与体验机制改造

#### 2.1 成交判定：从"是否成交"到"反馈分布"

- [ ] 修改 `trading_rules.py`：
  - [ ] 原逻辑：probe → Bernoulli(success)
  - [ ] 新逻辑：probe × K → K 次独立 Bernoulli
- [ ] 每次 probe 的成交概率仍由原有聚合概率函数给出
- [ ] 确保：不累计、不回溯、不记忆
- [ ] 成交结果只影响本 tick 的：
  - [ ] 成交数量
  - [ ] 即时 experience 更新

#### 2.2 体验更新：允许"失败也是结构"

- [ ] 修改 `random_player.py` 中的 `update_experience()` 方法
- [ ] **关键原则**：失败（n_failure）**不直接**出现在 reward 公式中
- [ ] 新的体验更新公式：
  ```
  instant_reward = w₁ * success_rate      # 成功率，不是成功数
                 + w₂ * match_dispersion  # 成交分布离散度
                 + w₃ * complexity        # 世界结构密度
  ```
- [ ] 失败通过以下方式**间接**影响体验：
  - [ ] success_rate 下降（但不是直接惩罚）
  - [ ] match_dispersion 变化（失败的分布形态）
  - [ ] complexity 变化（世界整体结构）
- [ ] 设计含义：
  - [ ] 大量失败 ≠ 不好玩（如果失败有结构）
  - [ ] 失败集中在硬区间 → dispersion 低但 pattern 清晰 → 体验不下降
  - [ ] 失败均匀分布（无结构噪声）→ 体验可能下降
- [ ] 不改变体验 EMA 形式（0.98/0.02），只改变 instant_reward 的构成项

### Phase 3: 混沌因子（Chaos Factor）修订

#### 3.1 Chaos Factor 逻辑调整

- [ ] 修改 `chaos_rules.py`：
  - [ ] 原假设（已失效）：玩家数 ↑ → 撮合失败 ↑ → 体验下降 → 玩家减少
  - [ ] 新现实：玩家数 ↑ → 探针密度 ↑ → 世界阻抗显影
- [ ] v5.2 中 Chaos Factor 的新角色：
  - [ ] 不再惩罚"失败率"
  - [ ] 只惩罚**结构塌缩**：
    - [ ] 成交全集中在一点（单点吸引子，H_norm ≈ 0）
    - [ ] 成交完全均匀分布（无结构噪声，H_norm ≈ 1）
- [ ] **结构塌缩 proxy 实现**：
  ```
  collapse = |2 * H_norm - 1|
  
  H_norm ≈ 0   → collapse ≈ 1（塌缩：单点吸引子）
  H_norm ≈ 0.5 → collapse ≈ 0（健康：有结构）
  H_norm ≈ 1   → collapse ≈ 1（候选塌缩，需第二判定）
  ```
- [ ] **高熵区分**：当 H_norm > 0.8 时，用 `active_protocols_ratio` 区分
  ```
  if H_norm > 0.8:
      collapse *= active_protocols_ratio
  # active_protocols_ratio 高 → 均匀噪声 → collapse 保持高（塌缩）
  # active_protocols_ratio 低 → 多结构有骨架 → collapse 压低（健康）
  ```
- [ ] Chaos Factor 公式：`chaos = base_chaos * (1 + γ * collapse_proxy)`
- [ ] 实现要求：
  - [ ] 仍然是无记忆、无历史
  - [ ] 仍然是 tick-local
  - [ ] 新增输入项：
    - [ ] 探针总数（用于计算 probe_density）
    - [ ] 成交价格分布（用于计算 collapse_proxy）

### Phase 4: 集成与验证

#### 4.1 代码集成

- [ ] 修改 `main.py`（V5MarketSimulator）集成所有改动
- [ ] 修改 `state_engine.py` 适配新的状态更新逻辑
- [ ] 确保 `metrics.py` 的 complexity 计算保持不变

#### 4.2 单元测试

- [ ] 为新的探针机制编写测试
- [ ] 为新的体验更新逻辑编写测试
- [ ] 为修订后的 Chaos Factor 编写测试

#### 4.3 回归测试

- [ ] 确保 P0 (非线性测试) 仍然通过
- [ ] 确保 P1 (反身性测试) 仍然通过
- [ ] 确保 P2 (多尺度结构密度测试) 仍然通过
- [ ] 确保 P3 (混沌因子消融测试) 行为符合预期（可能需要调整基准）

### Phase 5: 文档更新

- [ ] 更新 `core_system/README.md`
- [ ] 更新 `TECHNICAL_DOCUMENTATION.md`
- [ ] 更新 `InfiniteGame_V5_TechnicalNote.md` 或创建 V5.2 版本
- [ ] 更新 `README.md` 版本信息

---

## 📐 关键公式（待实现）

### 并行探针机制

```
每个 Player 每 tick:
  K = sample_K()  # K ≥ 1，固定或弱随机，不依赖状态
  for i in 1..K:
    probe_i = generate_probe()  # 随机价格、随机方向、size=1
    success_i ~ Bernoulli(p_match(probe_i, state))
  
  # 聚合本 tick 的探针结果
  n_success = Σ success_i
  n_failure = K - n_success
  match_prices = [probe_i.price for i where success_i = 1]
```

**K 的语义**：行为采样密度，不是订单厚度，不是持仓量。

### 体验更新（修订版）

```
# 失败不直接作为惩罚项！
instant_reward = w₁ * success_rate 
               + w₂ * match_dispersion 
               + w₃ * complexity

其中：
  success_rate    = n_success / K                           # 成功率（0-1）
  match_dispersion = std(match_prices) / price_range        # 归一化离散度
  complexity      = current_structure_density               # 世界结构密度

建议权重：w₁ = 0.4, w₂ = 0.2, w₃ = 0.4

# EMA 平滑（保持不变）
experience = 0.98 * experience + 0.02 * instant_reward
```

**设计要点**：失败通过 success_rate 和 match_dispersion 间接影响，不直接惩罚。

### Chaos Factor（修订版）

```
# 不再惩罚失败率，只惩罚结构塌缩
chaos_factor = base_chaos * (1 + γ * collapse_proxy)

其中：
  base_chaos = f(dispersion, entropy, overload)  # 保持 v5.0 原有计算
  collapse_proxy = 1 - |H_norm - 0.5| * 2        # 结构塌缩代理（见上文）
  γ = collapse_sensitivity                        # 塌缩敏感度参数

collapse_proxy 的含义：
  ≈ 0  → 健康结构（中等熵）  → chaos 不额外增加
  ≈ 1  → 塌缩状态（极端熵）  → chaos 增加，抑制过度行为
```

### 结构塌缩 Proxy 计算

```python
def compute_collapse_proxy(match_prices: List[float], 
                           active_protocols_ratio: float,
                           n_bins: int = 10) -> float:
    """
    计算结构塌缩代理指标（含高熵区分）
    
    参数：
      active_protocols_ratio: 活跃簇占比（驻留>2%的簇数 / 总簇数）
    
    返回值：
      0 → 健康结构
      1 → 塌缩状态（单点吸引子或均匀噪声）
    """
    if len(match_prices) < 2:
        return 1.0
    
    hist, _ = np.histogram(match_prices, bins=n_bins, density=True)
    hist = hist + 1e-10
    hist = hist / hist.sum()
    
    H = -np.sum(hist * np.log(hist))
    H_norm = H / np.log(n_bins)
    
    # 基础塌缩 proxy
    collapse = abs(2 * H_norm - 1)
    
    # 高熵时：用 active_protocols_ratio 区分噪声与健康
    if H_norm > 0.8:
        # 高 → 均匀噪声（塌缩），低 → 多结构有骨架（健康）
        collapse *= active_protocols_ratio
    
    return collapse
```

**判定逻辑总结**：
- `active_protocols_ratio` = 驻留 >2% 的簇数 / 总簇数
- 高熵 + 高活跃占比 → 到处跳、无骨架 → **均匀噪声塌缩**
- 高熵 + 低活跃占比 → 只有部分簇常驻 → **多结构健康**

---

## 🔬 验证标准

1. **结构丰度提升**：v5.2 应该产生比 v5.0 更丰富的市场结构
2. **理论前提保持**：P0-P3 的核心假设和结论不应被破坏
3. **零策略保持**：Player 仍然是"零策略"的，不引入任何学习
4. **Tick-Local**：所有新机制都是 tick-local 的，无跨 tick 依赖

---

## 📝 备注

- 本文件基于 `/Users/liugang/Desktop/5.2内容.txt` 设计文档创建
- v5.0 代码已锁定在 `main` 分支，可作为参考基准
- 开发过程中如有设计变更，请同步更新本文件
- 2026-01-25: 补充三点关键设计决策（K 语义、失败非直接惩罚、塌缩 proxy）
- 2026-01-25: 修正高熵区分公式符号（`collapse *= active_protocols_ratio`，高→塌缩，低→健康）

---

**最后更新**: 2026-01-25
