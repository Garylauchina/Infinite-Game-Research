# 测试数据目录

此目录用于存储从 Infinite-Game 仓库导入的实验数据和测试结果。

## 目录结构

```
data/
├── experiments/     # 实验运行数据
│   ├── runs/       # 单次运行数据
│   └── sweeps/     # 参数扫描数据
├── analysis/       # 分析结果
│   ├── phase_analysis/  # 阶段分析结果
│   └── validation/      # 验证结果
└── README.md       # 本文件
```

## 数据导入说明

从 Infinite-Game 仓库导入数据时，请保持以下结构：

1. **实验运行数据**：从 `Infinite-Game/EXPERIMENTS/outputs/runs/` 复制到 `data/experiments/runs/`
2. **分析结果**：从 `Infinite-Game/EXPERIMENTS/phase_analysis_output/` 复制到 `data/analysis/phase_analysis/`
3. **验证结果**：从 `Infinite-Game/v5.0_phase1/` 相关输出复制到 `data/analysis/validation/`

## 注意事项

- 大文件请使用 Git LFS 或外部存储
- 数据文件应包含元数据（seed、配置、时间戳等）
- 定期清理过时数据，保持仓库大小合理
