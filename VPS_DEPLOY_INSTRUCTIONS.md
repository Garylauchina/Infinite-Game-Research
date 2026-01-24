# VPS 程序员执行指令（简短版）

## 1. 改 wrapper：优化内存和性能

已完成的优化：
- ✅ **tprint 节流**：使用时间间隔（`IG_PRINT_EVERY_SEC`）而非 tick 间隔
- ✅ **last_pt**：保存最新数据点，WebSocket 不拷贝 buf
- ✅ **traj/cpx 窗口化**：使用 `deque(maxlen=...)` 避免内存无限增长

## 2. 默认环境变量

已在 `infinite-live.service` 中配置：

```bash
Environment=IG_LOG_EVERY=20000          # 日志输出间隔（ticks）
Environment=IG_PRINT_EVERY_SEC=10        # 打印节流（秒）
Environment=IG_TRAJ_WIN=20000            # 轨迹窗口大小
Environment=IG_CPX_WIN=20000             # 复杂度历史窗口大小
Environment=IG_DOWNSAMPLE=10             # 数据降采样
```

## 3. Systemd/journald 日志上限（可选）

当前配置：日志输出到文件（避免 journald 限制）

```ini
StandardOutput=append:/root/Infinite-Game-Research/logs/infinite-live.out.log
StandardError=append:/root/Infinite-Game-Research/logs/infinite-live.err.log
```

如果使用 journald，可以取消注释以下配置：

```ini
# StandardOutput=journal
# StandardError=journal
# LogRateLimitIntervalSec=60
# LogRateLimitBurst=1000
```

## 部署步骤

```bash
# 1. 更新服务文件
sudo cp /root/Infinite-Game-Research/infinite-live.service /etc/systemd/system/
sudo systemctl daemon-reload

# 2. 重启服务
sudo systemctl restart infinite-live.service

# 3. 检查状态
sudo systemctl status infinite-live.service

# 4. 查看日志
tail -f /root/Infinite-Game-Research/logs/infinite-live.out.log
```

## 优化说明

1. **tprint 节流**：按时间间隔（默认10秒）而非 tick 间隔打印，减少日志量
2. **last_pt**：WebSocket 直接使用 `last_pt`，避免每次 `list(buf)[-1]` 拷贝
3. **窗口化**：`traj_window` 和 `cpx_window` 使用 `deque(maxlen=20000)`，自动丢弃旧数据
