# 测试数据目录

此目录用于存储实验数据和测试结果。

## 目录结构

```
data/
├── experiments/     # 实验运行数据
│   ├── runs/       # 单次运行数据
│   └── sweeps/     # 参数扫描数据
├── analysis/       # 分析结果
│   ├── phase_analysis/  # 阶段分析结果（P0-P3）
│   └── validation/      # 验证结果
└── README.md       # 本文件
```

## 数据说明

1. **实验运行数据**：保存在 `data/experiments/runs/` 目录
2. **分析结果**：保存在 `data/analysis/phase_analysis/` 目录（P0-P3 阶段测试结果）
3. **验证结果**：保存在 `data/analysis/validation/` 目录

## 注意事项

- 大文件请使用 Git LFS 或外部存储
- 数据文件应包含元数据（seed、配置、时间戳等）
- 定期清理过时数据，保持仓库大小合理
