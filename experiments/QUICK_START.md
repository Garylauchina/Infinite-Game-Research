# 快速开始指南

本指南帮助您快速运行第一个实验并查看结果。

## 1. 安装依赖

```bash
pip install -r experiments/requirements.txt
```

## 2. 运行快速测试

```bash
# 使用快速测试配置（10000 ticks，约1-2分钟）
python experiments/run_single.py --config experiments/configs/quick_test.yaml --seed 42
```

这将：
- 运行一个短时间的实验（10000 ticks）
- 在 `outputs/runs/` 目录下创建运行目录
- 保存轨迹数据、指标和元数据

## 3. 查看结果

实验完成后，您会看到类似输出：

```
✅ 实验完成！结果保存在: outputs/runs/20250121/run_20250121_193000

📊 最终指标:
  final_player_count        : 5.0000
  final_complexity          : 0.8234
  avg_liquidity             : 0.6123
  final_avg_experience      : 0.7456
```

## 4. 分析结果

### 汇总指标

```bash
python experiments/analysis/summarize.py outputs/runs/20250121/run_20250121_193000
```

### 生成可视化图表

```bash
# 时间序列图
python experiments/analysis/timeseries_plots.py outputs/runs/20250121/run_20250121_193000

# 状态空间图
python experiments/analysis/state_space_plots.py outputs/runs/20250121/run_20250121_193000
```

图表将保存在 `outputs/runs/.../figs/` 目录下。

## 5. 运行完整验证

```bash
# 使用完整验证配置（500000 ticks，约30-60分钟）
python experiments/run_single.py --config experiments/configs/full_validation.yaml --seed 42
```

## 6. 查看数据

实验数据保存在运行目录的 `raw/` 子目录下：

- `trajectory.parquet` - 完整轨迹数据（Parquet格式，高效）
- `trajectory.csv` - 完整轨迹数据（CSV格式，便于查看）
- `player_history.csv` - 玩家数量历史
- `experience_history.csv` - 平均体验历史
- `complexity_history.csv` - 复杂度历史

元数据保存在 `meta/` 子目录下：

- `config_resolved.yaml` - 完整配置（含默认值）
- `git_commit.txt` - Git提交信息
- `pip_freeze.txt` - Python包版本
- `machine.json` - 机器信息
- `seeds.txt` - 随机种子信息

## 7. 自定义配置

您可以创建自己的配置文件，只需继承默认配置并覆盖需要的参数：

```yaml
# my_config.yaml
simulation:
  ticks: 20000
  adjust_interval: 1000

chaos_rules:
  base_chaos: 0.10  # 调整混乱因子
```

然后运行：

```bash
python experiments/run_single.py --config my_config.yaml --seed 42
```

## 常见问题

### Q: 如何确保结果可复现？

A: 每个运行都会保存完整的元数据，包括：
- Git commit hash
- Python包版本
- 随机种子
- 完整配置

使用相同的配置和种子，应该能得到相同的结果。

### Q: 输出目录在哪里？

A: 默认在 `outputs/runs/` 目录下，按日期和运行ID组织。

### Q: 如何批量运行多个seeds？

A: 目前需要手动循环运行，或使用shell脚本：

```bash
for seed in 0 1 2 3 4; do
    python experiments/run_single.py --seed $seed
done
```

### Q: 数据文件太大怎么办？

A: 在配置文件中调整 `save_interval` 参数，例如设置为 10 表示每10个tick保存一次。

## 下一步

- 阅读 [experiments/README.md](README.md) 了解完整功能
- 查看 [核心系统代码](../core_system/README.md) 了解实现细节
- 阅读 [研究文档](../README.md) 了解理论背景
