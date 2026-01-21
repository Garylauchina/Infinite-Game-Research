# 实验框架设置完成

## ✅ 已完成的工作

### 1. 实验框架文件

已从 Infinite-Game 仓库复制完整的实验框架：

#### 核心运行脚本
- ✅ `run_single.py` - 单seed运行脚本
- ✅ `config_loader.py` - 配置加载与解析
- ✅ `data_saver.py` - 数据保存模块

#### 配置文件
- ✅ `configs/default.yaml` - 默认配置
- ✅ `configs/quick_test.yaml` - 快速测试配置
- ✅ `configs/full_validation.yaml` - 完整验证配置

#### 分析脚本
- ✅ `analysis/summarize.py` - 汇总run指标
- ✅ `analysis/timeseries_plots.py` - 时间序列图
- ✅ `analysis/state_space_plots.py` - 状态空间可视化
- ✅ `analysis/compare_runs.py` - 对比两个run
- ✅ `analysis/phase_diagrams.py` - 参数扫描相位图

#### 文档
- ✅ `README.md` - 实验框架说明
- ✅ `QUICK_START.md` - 快速开始指南
- ✅ `requirements.txt` - 依赖列表

### 2. 代码适配

- ✅ 调整了导入路径：从 `src.v5` 改为 `core_system`
- ✅ 确保所有脚本可以正确导入核心系统代码

### 3. 目录结构

```
experiments/
├── __init__.py
├── README.md
├── QUICK_START.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── quick_test.yaml
│   └── full_validation.yaml
├── analysis/
│   ├── __init__.py
│   ├── summarize.py
│   ├── timeseries_plots.py
│   ├── state_space_plots.py
│   ├── compare_runs.py
│   └── phase_diagrams.py
├── run_single.py
├── config_loader.py
└── data_saver.py
```

## 📋 待完成的工作

### 缺失的实验数据

以下数据需要从 Infinite-Game 仓库导入到 `data/` 目录：

1. **示例运行数据**：
   - 从 `Infinite-Game/EXPERIMENTS/outputs/runs/` 复制1-2个示例运行
   - 包含完整的目录结构（meta/, raw/, metrics/, figs/）

2. **示例图表**：
   - 时间序列图（timeseries.png）
   - 状态空间图（state_space.png）
   - 其他分析图表

3. **示例输出文件**：
   - `trajectory.parquet` / `trajectory.csv` - 轨迹数据
   - `metrics.json` - 指标数据
   - `config_resolved.yaml` - 完整配置

### 建议的导入步骤

1. **选择示例运行**：
   ```bash
   # 从 Infinite-Game 仓库选择一个完整的运行目录
   # 例如：EXPERIMENTS/outputs/runs/20260121/run_20260121_165333
   ```

2. **复制到本仓库**：
   ```bash
   # 复制到 data/experiments/sample_run/
   cp -r Infinite-Game/EXPERIMENTS/outputs/runs/20260121/run_20260121_165333 \
        Infinite-Game-Research/data/experiments/sample_run/
   ```

3. **创建示例说明**：
   - 在 `data/experiments/` 创建 `SAMPLE_RUN_README.md`
   - 说明示例数据的来源和用途

## 🚀 快速验证

运行以下命令验证实验框架是否正常工作：

```bash
# 1. 安装依赖
pip install -r experiments/requirements.txt

# 2. 运行快速测试
python experiments/run_single.py --config experiments/configs/quick_test.yaml --seed 42

# 3. 查看结果
ls -la outputs/runs/*/run_*/

# 4. 生成可视化
python experiments/analysis/timeseries_plots.py outputs/runs/.../run_...
```

## 📝 注意事项

1. **输出目录**：实验输出保存在 `outputs/runs/` 目录，已加入 `.gitignore`
2. **路径问题**：所有脚本需要从项目根目录运行
3. **依赖版本**：确保安装的依赖版本与 `requirements.txt` 一致
4. **数据格式**：轨迹数据同时保存为 Parquet（高效）和 CSV（便于查看）

## 🔗 相关文档

- [实验框架说明](README.md)
- [快速开始指南](QUICK_START.md)
- [核心系统代码](../core_system/README.md)
- [项目主README](../README.md)
