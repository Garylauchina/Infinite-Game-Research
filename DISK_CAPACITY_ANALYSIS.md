# 磁盘容量分析报告

## 当前状态

### 磁盘信息
- **总容量**: 46.3 GB
- **已使用**: 23.6 GB (50.9%)
- **可用空间**: 20.7 GB
- **数据目录**: 461 MB (`/data/infinite_game/`)

### 数据生成情况
- **运行时间**: 0.26 小时（约 15 分钟）
- **当前 tick**: 249,142
- **Tick 速率**: ~271 ticks/秒
- **已完成段**: 3 个
- **已完成段总大小**: 440.2 MB

---

## 数据生成速率

### 压缩后磁盘占用
- **数据速率**: 494.9 KB/秒
- **每小时**: 1,739.8 MB (1.7 GB)
- **每天**: 40.78 GB

### 数据特点
- 每个 tick 压缩后约 **2.2 KB**
- 当前玩家数: ~74 个（从实际数据看）
- 每个 tick 包含完整的 agents、actions、matches 数据

---

## 磁盘容量预测

### 按当前速率
- **可用空间 (20.7 GB) 可运行**: **0.5 天 (12.2 小时)**
- **达到 80% 使用率**: 约 0.3 天
- **达到 90% 使用率**: 约 0.4 天
- **达到 95% 使用率**: 约 0.45 天

### ⚠️ 紧急警告
**按当前速率，磁盘将在 12 小时内填满！**

---

## 问题分析

### 数据生成速率过高的原因
1. **DOWNSAMPLE=1**: 每个 tick 都记录，没有降采样
2. **玩家数量多**: 当前约 74 个玩家，每个 tick 数据量大
3. **完整数据记录**: 每个 tick 包含完整的 agents、actions、matches

### 数据量估算
- 每个 tick: ~2.2 KB（压缩后）
- 每秒: ~271 ticks × 2.2 KB = ~596 KB/秒
- 每天: ~40.78 GB

---

## 解决方案

### 方案 1: 增加 DOWNSAMPLE（推荐）⭐⭐⭐⭐⭐

**修改环境变量**:
```bash
export IG_DOWNSAMPLE=10  # 每 10 个 tick 记录一次
```

**效果**:
- 数据速率降低 10 倍: ~4.08 GB/天
- 可运行时间: ~5 天
- **优点**: 简单，立即生效
- **缺点**: 数据粒度变粗

### 方案 2: 定期清理旧数据 ⭐⭐⭐⭐

**实现自动清理脚本**:
- 保留最近 N 天的数据（如 7 天）
- 自动删除超过 N 天的段文件
- 更新索引文件

**效果**:
- 磁盘占用稳定在 ~285 GB（7 天 × 40.78 GB）
- 需要约 285 GB 磁盘空间
- **优点**: 保持数据完整性
- **缺点**: 需要足够磁盘空间

### 方案 3: 增加磁盘容量 ⭐⭐⭐

**扩容方案**:
- 当前: 46.3 GB
- 建议: 至少 200 GB（可运行约 5 天）
- 理想: 500 GB+（可运行 12+ 天）

### 方案 4: 数据归档 ⭐⭐⭐

**实施策略**:
- 定期将旧数据归档到其他存储（如对象存储、NAS）
- 本地只保留最近数据
- 需要时从归档恢复

---

## 推荐方案组合

### 短期（立即实施）
1. **增加 DOWNSAMPLE=10**: 降低数据生成速率 10 倍
2. **监控磁盘使用**: 设置告警（如 80% 使用率）

### 中期（1-2 天内）
1. **实施自动清理**: 保留最近 3-7 天数据
2. **增加磁盘容量**: 扩容到 200 GB+

### 长期（1 周内）
1. **数据归档系统**: 将旧数据迁移到低成本存储
2. **优化数据格式**: 考虑更高效的存储格式

---

## 立即行动建议

### 1. 修改 DOWNSAMPLE（最快见效）
```bash
# 停止服务
kill $(ps aux | grep "python3.*experiments.live.server" | grep -v grep | awk '{print $2}')

# 设置环境变量
export IG_DOWNSAMPLE=10

# 重启服务
cd /root/Infinite-Game-Research && source .venv/bin/activate && \
  nohup python3 -m experiments.live.server > /tmp/infinite-server.log 2>&1 &
```

**预期效果**:
- 数据速率: 40.78 GB/天 → 4.08 GB/天
- 可运行时间: 12 小时 → 5 天

### 2. 设置磁盘监控
```bash
# 创建监控脚本
cat > /root/check_disk.sh << 'EOF'
#!/bin/bash
USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $USAGE -gt 80 ]; then
    echo "警告: 磁盘使用率 ${USAGE}%"
    # 可以发送告警或自动清理
fi
EOF
chmod +x /root/check_disk.sh

# 添加到 crontab（每 10 分钟检查一次）
# */10 * * * * /root/check_disk.sh
```

---

## 数据清理脚本示例

```python
#!/usr/bin/env python3
"""自动清理旧数据，保留最近 N 天"""
import json
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path('/data/infinite_game')
SEGMENTS_DIR = DATA_DIR / 'segments'
INDEX_DIR = DATA_DIR / 'index'
KEEP_DAYS = 7  # 保留最近 7 天

segments_json = INDEX_DIR / 'segments.json'
with open(segments_json, 'r') as f:
    segments = json.load(f)

cutoff_time = datetime.now() - timedelta(days=KEEP_DAYS)
deleted_count = 0
deleted_size = 0

for seg in segments[:]:
    created = datetime.fromisoformat(seg['created_at'].replace('Z', '+00:00'))
    if created < cutoff_time:
        seg_file = SEGMENTS_DIR / Path(seg['path']).name
        if seg_file.exists():
            size = seg_file.stat().st_size
            seg_file.unlink()
            segments.remove(seg)
            deleted_count += 1
            deleted_size += size

# 更新索引
with open(segments_json, 'w') as f:
    json.dump(segments, f, indent=2)

print(f'已删除 {deleted_count} 个旧段，释放 {deleted_size / 1024 / 1024:.1f} MB')
```

---

## 总结

### 当前状况
- ⚠️ **紧急**: 按当前速率，磁盘将在 **12 小时内填满**
- 数据生成速率: **40.78 GB/天**（压缩后）
- 可用空间: **20.7 GB**

### 建议
1. **立即**: 设置 `DOWNSAMPLE=10`，降低数据生成速率
2. **短期**: 实施自动清理策略，保留最近 3-7 天数据
3. **长期**: 考虑扩容或数据归档

### 预期效果（DOWNSAMPLE=10）
- 数据速率: 4.08 GB/天
- 可运行时间: 约 5 天
- 磁盘压力: 大幅降低
