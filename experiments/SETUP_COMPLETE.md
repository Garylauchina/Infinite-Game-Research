# 实验框架设置完成

## ✅ 已完成的工作

### 1. 实验框架文件

本仓库已包含完整的实验框架：

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

### 实验数据

实验数据将保存在 `data/` 目录中：

1. **运行数据**：
   - 保存在 `data/experiments/runs/` 目录
   - 包含完整的目录结构（meta/, raw/, metrics/, figs/）

2. **分析结果**：
   - 时间序列图（timeseries.png）
   - 状态空间图（state_space.png）
   - 其他分析图表

3. **输出文件**：
   - `trajectory.parquet` / `trajectory.csv` - 轨迹数据
   - `metrics.json` - 指标数据
   - `config_resolved.yaml` - 完整配置

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
