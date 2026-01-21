# V5.0 实验框架

实验框架用于运行、保存和分析 Infinite Game V5.0 的实验。

## 目录结构

```
experiments/
├── configs/              # 配置文件
│   ├── default.yaml      # 默认配置
│   ├── quick_test.yaml   # 快速测试
│   └── full_validation.yaml  # 完整验证
│
├── analysis/             # 分析脚本
│   ├── summarize.py      # 汇总run指标
│   ├── compare_runs.py   # 对比两个run
│   ├── phase_diagrams.py # 参数扫描相位图
│   ├── timeseries_plots.py  # 时间序列图
│   └── state_space_plots.py  # 状态空间图
│
├── run_single.py         # 单seed运行
├── config_loader.py      # 配置加载
├── data_saver.py         # 数据保存
└── README.md             # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r experiments/requirements.txt
```

### 2. 单seed实验

```bash
# 使用默认配置
python experiments/run_single.py --seed 42

# 使用快速测试配置
python experiments/run_single.py --config experiments/configs/quick_test.yaml --seed 0
```

### 3. 分析结果

```bash
# 汇总run指标
python experiments/analysis/summarize.py outputs/runs/20250115/run_20250115_120000

# 生成时间序列图
python experiments/analysis/timeseries_plots.py outputs/runs/20250115/run_20250115_120000

# 生成状态空间图
python experiments/analysis/state_space_plots.py outputs/runs/20250115/run_20250115_120000
```

## 配置文件

配置文件使用YAML格式，支持嵌套结构。所有未指定的参数将使用`default.yaml`中的默认值。

### 配置示例

```yaml
simulation:
  ticks: 50000
  adjust_interval: 1000
  max_n: null  # null表示无上限
  
chaos_rules:
  enabled: true
  base_chaos: 0.08
  
random_seed:
  numpy: 42
  random: 42
  sklearn: 42
```

## 数据格式

### 轨迹数据 (trajectory.parquet)

包含以下列：
- `t`: tick索引
- `price_norm`: 标准化价格 [0,1]
- `volatility`: 波动率 [0,1]
- `liquidity`: 流动性 [0,1]
- `imbalance`: 不平衡度 [0,1]
- `complexity`: 复杂度 [0,1]
- `N`: 玩家数量
- `avg_exp`: 平均体验
- `cluster`: 聚类分配（如果启用）

## 复现性

每个run都会保存：
- Git commit hash
- Python包版本（pip freeze）
- 机器信息（包括BLAS）
- 完整配置（config_resolved.yaml）
- 随机种子

确保复现性：
1. 使用相同的配置文件
2. 使用相同的随机种子
3. 使用相同版本的依赖包

## 注意事项

- 轨迹数据默认保存为parquet格式（高效压缩），同时保存CSV便于查看
- 运行脚本需要从项目根目录执行，确保能正确导入 `core_system` 模块
