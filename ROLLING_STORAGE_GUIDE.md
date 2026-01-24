# 滚动存储方案说明

## 概述

滚动存储方案会自动监控磁盘使用率，当超过阈值时自动清理最旧的段文件，防止磁盘空间耗尽导致服务崩溃。

---

## 配置参数

通过环境变量配置滚动存储行为：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `DISK_USAGE_THRESHOLD` | `80.0` | 磁盘使用率阈值（%），超过此值触发清理 |
| `DISK_CLEANUP_TARGET` | `70.0` | 清理目标使用率（%），清理到此值停止 |
| `KEEP_MIN_SEGMENTS` | `10` | 至少保留的段数量（即使超过阈值也不删除） |
| `KEEP_MIN_DAYS` | `1.0` | 至少保留的天数（最近 N 天的数据不删除） |

### 配置示例

```bash
# 设置更严格的清理策略（80% 触发，清理到 60%）
export DISK_USAGE_THRESHOLD=80.0
export DISK_CLEANUP_TARGET=60.0
export KEEP_MIN_SEGMENTS=20
export KEEP_MIN_DAYS=2.0

# 重启服务使配置生效
```

---

## 工作原理

### 1. 磁盘监控

每次关闭段文件（`close_segment()`）时，系统会：
- 检查当前磁盘使用率
- 如果使用率 < 阈值，不执行任何操作
- 如果使用率 ≥ 阈值，触发自动清理

### 2. 清理策略

清理过程遵循以下规则（按优先级）：

1. **保护当前写入的段**：不会删除正在写入的段
2. **最小保留数量**：至少保留 `KEEP_MIN_SEGMENTS` 个段
3. **最小保留天数**：至少保留最近 `KEEP_MIN_DAYS` 天的数据
4. **从最旧开始删除**：按创建时间排序，从最旧的段开始删除
5. **达到目标停止**：清理到 `DISK_CLEANUP_TARGET` 使用率后停止

### 3. 索引更新

清理完成后，系统会：
- 从 `segments.json` 中移除已删除段的记录
- 原子性更新索引文件（fsync 确保数据持久化）

---

## 监控和查询

### API 端点

#### `GET /disk`

获取磁盘使用情况和清理配置：

```bash
curl https://45.76.97.37/disk
```

**响应示例**：
```json
{
  "disk": {
    "usage_percent": 55.2,
    "total_gb": 47.0,
    "used_gb": 25.9,
    "free_gb": 21.1
  },
  "segments": {
    "count": 15,
    "total_bytes": 461234567,
    "total_gb": 0.43
  },
  "cleanup_config": {
    "threshold_percent": 80.0,
    "target_percent": 70.0,
    "keep_min_segments": 10,
    "keep_min_days": 1.0
  },
  "status": {
    "needs_cleanup": false
  }
}
```

---

## 日志

清理操作会记录详细日志：

```
[INFO] 磁盘使用率 82.3% 超过阈值 80.0%，开始清理旧数据...
[INFO] Deleted segment: segment_20260123_120000_100000.jsonl.zst (45.2 MB)
[INFO] Deleted segment: segment_20260123_115500_80000.jsonl.zst (42.8 MB)
[INFO] 清理完成: 删除 5 个段，释放 215.6 MB，剩余 12 个段，磁盘使用率约 68.5%
```

---

## 最佳实践

### 1. 合理设置阈值

- **生产环境**：建议 `DISK_USAGE_THRESHOLD=75.0`，提前清理
- **测试环境**：可以使用默认值 `80.0`

### 2. 保留策略

- **最小段数**：根据数据回放需求设置（如需要回放最近 10 个段，设置 `KEEP_MIN_SEGMENTS=10`）
- **最小天数**：根据业务需求设置（如需要保留最近 1 天的数据，设置 `KEEP_MIN_DAYS=1.0`）

### 3. 监控告警

建议设置外部监控，当磁盘使用率持续超过阈值时发送告警：

```bash
# 示例监控脚本
#!/bin/bash
USAGE=$(curl -s https://45.76.97.37/disk | jq -r '.disk.usage_percent')
if (( $(echo "$USAGE > 85" | bc -l) )); then
    echo "警告: 磁盘使用率 ${USAGE}%"
    # 发送告警...
fi
```

### 4. 定期检查

定期检查磁盘状态和清理效果：

```bash
# 每天检查一次
curl https://45.76.97.37/disk | jq '.'
```

---

## 故障排查

### 问题 1: 清理未触发

**可能原因**：
- 磁盘使用率未达到阈值
- 段数量少于 `KEEP_MIN_SEGMENTS`
- 所有段都在 `KEEP_MIN_DAYS` 保护期内

**解决方法**：
- 检查 `/disk` API 返回的 `status.needs_cleanup`
- 查看日志确认是否有清理尝试
- 调整 `KEEP_MIN_SEGMENTS` 或 `KEEP_MIN_DAYS` 参数

### 问题 2: 清理后使用率仍然很高

**可能原因**：
- 数据生成速率过快
- `DISK_CLEANUP_TARGET` 设置过高
- 保留策略过于保守

**解决方法**：
- 降低 `DISK_CLEANUP_TARGET`（如从 70% 降到 60%）
- 减少 `KEEP_MIN_SEGMENTS` 或 `KEEP_MIN_DAYS`
- 考虑增加 `DOWNSAMPLE` 降低数据生成速率

### 问题 3: 误删重要数据

**预防措施**：
- 设置合理的 `KEEP_MIN_SEGMENTS` 和 `KEEP_MIN_DAYS`
- 定期备份重要数据到其他存储
- 监控清理日志，确认删除的段符合预期

---

## 与 DOWNSAMPLE 配合使用

滚动存储可以与 `DOWNSAMPLE` 参数配合使用，进一步降低磁盘压力：

```bash
# 降低数据生成速率（每 10 个 tick 记录一次）
export IG_DOWNSAMPLE=10

# 设置清理策略
export DISK_USAGE_THRESHOLD=80.0
export DISK_CLEANUP_TARGET=70.0
export KEEP_MIN_SEGMENTS=10
export KEEP_MIN_DAYS=1.0
```

**效果**：
- 数据生成速率降低 10 倍
- 磁盘使用率增长更慢
- 清理频率降低

---

## 总结

滚动存储方案提供了：
- ✅ **自动监控**：实时监控磁盘使用率
- ✅ **自动清理**：超过阈值时自动删除旧数据
- ✅ **安全保护**：保护当前写入和最近数据
- ✅ **可配置**：通过环境变量灵活调整策略
- ✅ **可观测**：提供 API 和日志监控

**建议配置**（生产环境）：
```bash
export DISK_USAGE_THRESHOLD=75.0
export DISK_CLEANUP_TARGET=65.0
export KEEP_MIN_SEGMENTS=15
export KEEP_MIN_DAYS=1.5
```

---

**最后更新**: 2026-01-23
