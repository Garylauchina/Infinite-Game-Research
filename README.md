# Infinite-Game-Research

> A finite game is played for the purpose of winning, an infinite game for the purpose of continuing the play.  
> — James P. Carse, *Finite and Infinite Games*

---

## 项目简介

这是一个开发项目的并行研究仓库，用于进行理论展示和测试数据分析。Infinite Game 是一个基于反身性原理的交易所模拟器，用于研究金融市场中的结构涌现和演化机制。

**核心设计范式**：
1. **唯一参与主体**：一个市场中存在唯一参与主体"市场先生"（Market Entity），代表着所有交易的聚合
2. **结构密度作为奖励函数**：设计了"结构密度"作为奖励函数的核心组成部分（权重0.4），驱动市场先生的参与决策

**开发仓库**: [`Infinite-Game`](../Infinite-Game) (本地路径: `/Users/liugang/Cursor_Store/Infinite-Game`)

---

## 文档导航

### 📚 核心文档

1. **[理论框架](THEORETICAL_FRAMEWORK.md)**  
   完整的理论框架文档，包括：
   - 哲学基础（有限与无限游戏）
   - 核心理论概念（反身性、涌现、协议、相变）
   - 系统架构理论
   - 反身性机制详解
   - 涌现与复杂性分析
   - 范式演进（V0 → V4.x → V5.0）

2. **[研究论文](RESEARCH_PAPER.md)**  
   学术研究论文（草稿），包括：
   - 摘要与关键词
   - 引言与研究问题
   - 文献综述
   - 理论框架
   - 系统设计
   - 实验设计与结果分析
   - 讨论与结论

3. **[技术文档](TECHNICAL_DOCUMENTATION.md)**  
   技术实现文档，包括：
   - 系统概述与架构设计
   - 核心组件详解
   - 数据流与关键算法
   - 实验方法与数据分析
   - 性能优化与扩展性

---

## 项目结构

```
Infinite-Game-Research/
├── README.md                      # 项目导航（本文件）
├── LICENSE                        # MIT 许可证
├── THEORETICAL_FRAMEWORK.md       # 理论框架文档
├── RESEARCH_PAPER.md              # 研究论文（草稿）
├── TECHNICAL_DOCUMENTATION.md     # 技术文档
├── InfiniteGame_V5_TechnicalNote.md  # V5.0 技术笔记
├── PHASE_TEST_SPECIFICATION.md      # P0-P3 阶段测试规范
├── core_system/                   # 核心系统代码（已锁定版本，可直接使用）
│   ├── README.md                  # 代码说明
│   ├── __init__.py                # 包初始化
│   ├── main.py                    # 主模拟器（V5MarketSimulator）
│   ├── random_player.py           # 随机体验玩家
│   ├── state_engine.py            # 状态引擎
│   ├── trading_rules.py           # 交易规则
│   ├── chaos_rules.py             # 混乱因子规则
│   └── metrics.py                 # 结构密度计算
├── experiments/                   # 实验框架
│   ├── README.md                  # 实验框架说明
│   ├── QUICK_START.md             # 快速开始指南
│   ├── requirements.txt           # 依赖列表
│   ├── configs/                   # 配置文件
│   │   ├── default.yaml           # 默认配置
│   │   ├── quick_test.yaml        # 快速测试
│   │   └── full_validation.yaml   # 完整验证
│   ├── analysis/                  # 分析脚本
│   │   ├── summarize.py           # 汇总指标
│   │   ├── timeseries_plots.py    # 时间序列图
│   │   ├── state_space_plots.py   # 状态空间图
│   │   ├── compare_runs.py        # 对比运行
│   │   └── phase_diagrams.py      # 参数扫描图
│   ├── run_single.py              # 单seed运行
│   ├── config_loader.py           # 配置加载
│   └── data_saver.py              # 数据保存
├── data/                          # 测试数据目录
│   ├── experiments/               # 实验运行数据
│   ├── analysis/                  # 分析结果
│   └── README.md                  # 数据说明
└── archive/                       # 归档目录
    ├── README.md                  # 归档说明
    └── [旧资料和工具文件]
```

---

## 核心概念

### 系统定位：市场基底模型（Market Substrate / Zero Model）

**当前系统定位为**：一个仅保留交易市场最小共性规则的**市场基底模型**。

该模型刻意剥离了：
- 个体策略智能与学习
- 信息不对称与预期博弈
- 信用、杠杆、清算与破产
- 制度级规则切换
- 外生冲击与危机触发器

**研究目标**：在极简、共性的交易规则下，市场是否仍然会自发地产生稳定而非平凡的结构？

**核心发现**：市场复杂结构的一个重要来源，是交易规则本身，而非参与者行为或制度细节。

### 市场先生（Market Entity / MarketMr）：唯一参与主体

**核心设计范式**：一个市场中存在**唯一参与主体"市场先生"**，代表着所有交易的聚合。

- **唯一参与主体**：对于一个单一市场，仅存在一个抽象的"市场先生"，它是市场中所有交易的聚合代表
- **分身机制**：市场先生通过多个分身（Avatar/Player）表示其参与强度，分身数量 = 参与强度
- **交易聚合**：所有分身的交易行为聚合为市场的整体交易活动，市场先生代表所有交易的统一主体
- **盈亏抵消**：多个分身的盈利/亏损互相抵消，市场先生实际支付的只是交易费用

### 市场结构密度（Market Structure Density）：奖励函数

**核心设计范式**：设计了"结构密度"作为奖励函数的核心组成部分。

- **定义**：衡量市场状态轨迹在状态空间中的分布复杂程度
- **作为奖励函数**：结构密度直接作为体验奖励的核心组成部分（权重0.4），驱动市场先生的参与决策
- **计算**：通过聚类分析、转移矩阵熵等指标计算（`complexity = 0.4 * protocol_score + 0.4 * transfer_entropy + 0.2 * uniformity`）
- **反馈机制**：市场结构密度的强弱，决定市场先生参与强度的大小（分身数量），形成"结构密度 → 体验奖励 → 参与强度"的正反馈循环

### 混乱因子（Chaos Factor）

- **定义**：当参与强度高时，"混乱因子"会升高，惩罚交易成功概率
- **作用**：防止系统过度拥挤，维持动态平衡，避免单一吸引子主导

### 规则提炼（Rule Extraction）

- **设计原则**：不去模拟单个交易所复杂的规则集，而是将金融市场最根本的共性规则提炼出来
- **核心思想**：简单规则也会产生复杂结构
- **意义**：通过提炼共性规则，我们能够观察到市场结构的本质特征，而不被具体交易所的复杂规则所干扰

### 涌现（Emergence）

宏观层面的复杂结构从微观层面的简单规则和交互中自发产生。

### 设计范式的潜在应用

本模拟器采用的设计范式（单一抽象主体 + 结构密度作为奖励函数）可能具有一定的普适性，适用于需要从系统整体视角理解、通过结构复杂度驱动演化的场景。潜在的应用领域包括：

- **虚拟经济/游戏经济**：通过结构密度自动调节经济系统活力
- **多智能体环境设计**：通过环境复杂度自适应调节训练难度
- **人工社会模拟**：通过社会复杂度驱动社会系统演化
- **模拟城市/生态**：通过系统复杂度维持系统稳定性和可持续性
- **AI训练环境**：通过环境复杂度实现自适应课程学习

**未来工作方向**：本模拟器的长期发展目标是能够持续输出"结构参数集"（包括结构密度计算方法、参与强度调整机制、反馈循环参数等），这些参数集可以作为模块化的组件植入到各种模拟实验环境中，为其他领域的研究提供参考。

---

## 版本信息

- **当前版本**: V5.0 (GameTheoryMarket)
- **核心范式**: 规则驱动 + 随机试玩 + 状态格子行走

---

## 实验结果

**注意**：P0-P3 阶段测试正在进行中，实验结果将在测试完成后更新到文档中。

详细的测试规范请参考 [PHASE_TEST_SPECIFICATION.md](PHASE_TEST_SPECIFICATION.md)。

---

## 快速开始

### 阅读建议

1. **初学者**：从 [理论框架](THEORETICAL_FRAMEWORK.md) 开始，了解核心概念
2. **研究者**：阅读 [研究论文](RESEARCH_PAPER.md)，了解研究方法和发现
3. **开发者**：查看 [技术文档](TECHNICAL_DOCUMENTATION.md)，了解实现细节

### 相关资源

- **核心代码**: 本仓库的 `core_system/` 目录（已锁定版本，可直接使用）
- **实验框架**: 本仓库的 `experiments/` 目录（完整实验框架，可直接运行和复现）
- **测试规范**: [PHASE_TEST_SPECIFICATION.md](PHASE_TEST_SPECIFICATION.md) - P0-P3 阶段测试规范
- **技术笔记**: [InfiniteGame_V5_TechnicalNote.md](InfiniteGame_V5_TechnicalNote.md) - 完整技术文档
- **开发仓库**: [`Infinite-Game`](../Infinite-Game) - 包含最新开发版本和实验脚本

### 代码说明

本仓库的 `core_system/` 目录包含完整的、可直接运行的核心系统代码：

- ✅ **可直接使用**：所有核心代码已包含，无需依赖外部仓库
- ✅ **完全可复现**：固定随机种子，确定性执行，支持完全复现的实验
- ✅ **已验证一致性**：核心代码与测试规范完全一致（见 [PHASE_TEST_SPECIFICATION.md](PHASE_TEST_SPECIFICATION.md)）

**核心组件**：
- `main.py`: V5MarketSimulator - 主模拟器
- `state_engine.py`: StateEngine - 状态更新引擎
- `random_player.py`: RandomExperiencePlayer - 随机体验玩家
- `trading_rules.py`: 交易规则（价格优先、手续费）
- `chaos_rules.py`: 混乱因子规则（动态调整）
- `metrics.py`: StructureMetrics - 结构密度计算

**代码版本**：已锁定版本，确保研究结果的可复现性。如需最新开发版本，请参考开发仓库。

### 实验框架与复现

本仓库的 `experiments/` 目录包含**完整的、可直接复现的实验框架**：

#### 快速开始

1. **安装依赖**：
   ```bash
   pip install -r experiments/requirements.txt
   ```

2. **运行快速测试**：
   ```bash
   python experiments/run_single.py --config experiments/configs/quick_test.yaml --seed 42
   ```

3. **查看结果**：
   - 实验数据保存在 `outputs/runs/` 目录
   - 使用分析脚本生成可视化图表

#### 完整实验复现

**单次运行**：
```bash
python experiments/run_single.py --config experiments/configs/default.yaml --seed 42
```

**批量运行**（多seed）：
```bash
for seed in 42 100 200 300 400; do
    python experiments/run_single.py --config experiments/configs/default.yaml --seed $seed
done
```

**阶段测试**（P0-P3）：
参考 [PHASE_TEST_SPECIFICATION.md](PHASE_TEST_SPECIFICATION.md) 了解详细的测试规范和复现方法。

#### 复现性保证

- ✅ **固定随机种子**：所有随机操作使用固定种子
- ✅ **确定性执行**：相同输入产生相同输出
- ✅ **完整数据记录**：所有实验数据自动保存，包含元数据
- ✅ **环境配置**：自动设置线程控制，确保跨平台一致性

详细说明请参考：
- [experiments/README.md](experiments/README.md) - 实验框架说明
- [experiments/QUICK_START.md](experiments/QUICK_START.md) - 快速开始指南
- [PHASE_TEST_SPECIFICATION.md](PHASE_TEST_SPECIFICATION.md) - 阶段测试规范

---

## 复现性

本仓库提供**完全可复现的研究结果**：

- ✅ **核心代码完整**：`core_system/` 目录包含所有必要的代码
- ✅ **实验框架完整**：`experiments/` 目录包含完整的实验脚本和分析工具
- ✅ **文档完整**：所有技术细节、参数设置、算法描述都已文档化
- ✅ **测试规范**：P0-P3 阶段测试规范已包含，核心代码已验证一致性

**复现步骤**：
1. 克隆本仓库
2. 安装依赖：`pip install -r experiments/requirements.txt`
3. 运行实验：`python experiments/run_single.py --config experiments/configs/default.yaml --seed 42`
4. 查看结果：实验数据自动保存到 `outputs/runs/` 目录

所有实验结果都可以通过本仓库的代码和配置完全复现。

## 贡献

本项目是研究仓库，提供完整的可复现代码和文档。如需贡献代码或实验，请参考开发仓库 [`Infinite-Game`](../Infinite-Game)。

---

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 联系方式

- **作者**: 刘刚
- **项目**: Infinite Game
- **日期**: 2025-01-15

---

**最后更新**: 2025-01-15
