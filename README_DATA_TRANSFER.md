# 从 VPS 复制 Infinite Game 数据到 Mac 本机

## 数据位置
- **VPS 路径**: `/data/infinite_game/`
- **段文件目录**: `/data/infinite_game/segments/` (约 3.2 GB)
- **索引文件**: `/data/infinite_game/index/segments.json` (约 9 KB)

---

## 方法 1: 使用 scp 复制（推荐）

### 复制整个数据目录
```bash
# 在 Mac 终端执行
scp -r root@your-vps-ip:/data/infinite_game ~/Downloads/infinite_game_data
```

### 只复制索引文件（先查看有哪些段）
```bash
# 复制索引文件
scp root@your-vps-ip:/data/infinite_game/index/segments.json ~/Downloads/

# 查看索引内容，决定要复制哪些段
cat ~/Downloads/segments.json | python3 -m json.tool
```

### 复制特定段文件
```bash
# 复制单个段文件
scp root@your-vps-ip:/data/infinite_game/segments/segment_20260123_175958_638716.jsonl.zst ~/Downloads/

# 复制多个段文件（使用通配符）
scp root@your-vps-ip:/data/infinite_game/segments/segment_20260123_17*.jsonl.zst ~/Downloads/
```

### 复制最新 N 个段文件
```bash
# 先 SSH 登录，找出最新的段文件
ssh root@your-vps-ip "ls -t /data/infinite_game/segments/*.jsonl.zst | head -5"

# 然后复制（替换为实际文件名）
scp root@your-vps-ip:/data/infinite_game/segments/segment_xxx.jsonl.zst ~/Downloads/
```

---

## 方法 2: 使用 rsync（推荐用于大文件/增量同步）

### 同步整个目录
```bash
# 在 Mac 终端执行
rsync -avz --progress root@your-vps-ip:/data/infinite_game/ ~/Downloads/infinite_game_data/
```

### 只同步已完成的段（排除 pending）
```bash
rsync -avz --progress --exclude='*_pending.jsonl.zst' \
  root@your-vps-ip:/data/infinite_game/segments/ \
  ~/Downloads/infinite_game_segments/
```

### 只同步索引文件
```bash
rsync -avz root@your-vps-ip:/data/infinite_game/index/ ~/Downloads/infinite_game_index/
```

**rsync 优势**:
- 支持断点续传
- 只传输差异部分
- 显示传输进度
- 可以排除特定文件

---

## 方法 3: 使用 SFTP（交互式）

```bash
# 在 Mac 终端执行
sftp root@your-vps-ip

# 进入 SFTP 后执行
cd /data/infinite_game
ls segments/
get segments/segment_20260123_175958_638716.jsonl.zst ~/Downloads/
get index/segments.json ~/Downloads/
exit
```

---

## 方法 4: 通过 HTTP API 下载（如果后端运行中）

### 下载索引
```bash
curl http://your-vps-ip:8000/segments > ~/Downloads/segments.json
```

### 下载段文件（解压后的 JSONL）
```bash
# 下载并保存为 JSONL 文件
curl "http://your-vps-ip:8000/segment/segments/segment_20260123_175958_638716.jsonl.zst?format=jsonl" \
  > ~/Downloads/segment_20260123_175958_638716.jsonl
```

**注意**: 这种方式下载的是解压后的文本，文件会很大（约 10-20 倍压缩前大小）

---

## 方法 5: 使用 SSHFS 挂载（实时访问）

### 安装 sshfs（如果未安装）
```bash
# 使用 Homebrew
brew install macfuse sshfs
```

### 挂载远程目录
```bash
# 创建挂载点
mkdir ~/vps_data

# 挂载
sshfs root@your-vps-ip:/data/infinite_game ~/vps_data

# 现在可以直接访问
ls ~/vps_data/segments/
cp ~/vps_data/segments/segment_xxx.jsonl.zst ~/Downloads/

# 卸载
umount ~/vps_data
```

---

## 方法 6: 压缩后传输（节省带宽）

### 在 VPS 上压缩
```bash
# SSH 登录 VPS
ssh root@your-vps-ip

# 压缩数据目录（排除 pending）
cd /data/infinite_game
tar -czf /tmp/infinite_game_data.tar.gz --exclude='*_pending.jsonl.zst' segments/ index/

# 退出 SSH
exit
```

### 在 Mac 上下载压缩包
```bash
scp root@your-vps-ip:/tmp/infinite_game_data.tar.gz ~/Downloads/
cd ~/Downloads
tar -xzf infinite_game_data.tar.gz
```

---

## 推荐方案

### 场景 1: 只需要查看数据结构和样本
```bash
# 只复制索引和几个样本段
scp root@your-vps-ip:/data/infinite_game/index/segments.json ~/Downloads/
scp root@your-vps-ip:/data/infinite_game/segments/segment_20260123_175958_638716.jsonl.zst ~/Downloads/
```

### 场景 2: 需要完整数据用于分析
```bash
# 使用 rsync，支持断点续传
rsync -avz --progress --exclude='*_pending.jsonl.zst' \
  root@your-vps-ip:/data/infinite_game/ \
  ~/Downloads/infinite_game_data/
```

### 场景 3: 需要实时访问（不复制）
```bash
# 使用 SSHFS 挂载
sshfs root@your-vps-ip:/data/infinite_game ~/vps_data
```

---

## 在 Mac 上读取数据

### 安装依赖
```bash
# 安装 Python 依赖
pip3 install zstandard

# 或使用虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install zstandard
```

### 读取压缩的段文件
```python
import json
import zstandard as zstd
from pathlib import Path
import io

# 读取段文件
seg_path = Path('~/Downloads/segment_xxx.jsonl.zst').expanduser()
dctx = zstd.ZstdDecompressor()

with open(seg_path, 'rb') as f:
    with dctx.stream_reader(f) as reader:
        text_reader = io.TextIOWrapper(reader, encoding='utf-8')
        
        # 读取 header
        header = json.loads(text_reader.readline().strip())
        print(f"Header: {header}")
        
        # 读取前 10 个 tick
        for i in range(10):
            line = text_reader.readline()
            if not line:
                break
            tick = json.loads(line.strip())
            print(f"Tick {i+1}: t={tick['t']}, agents={len(tick.get('agents', []))}")
```

---

## 注意事项

1. **文件大小**: 每个段文件约 100-170 MB（压缩后），完整数据约 3.2 GB
2. **网络带宽**: 如果网络较慢，建议使用 rsync 或压缩传输
3. **磁盘空间**: 确保 Mac 有足够空间（解压后可能更大）
4. **Pending 段**: 建议排除 `*_pending.jsonl.zst`（未完成的段）
5. **SSH 密钥**: 建议配置 SSH 密钥认证，避免每次输入密码

---

## 快速命令参考

```bash
# 替换 your-vps-ip 为实际 IP 地址

# 1. 复制索引
scp root@your-vps-ip:/data/infinite_game/index/segments.json ~/Downloads/

# 2. 复制最新 5 个段
ssh root@your-vps-ip "ls -t /data/infinite_game/segments/*.jsonl.zst | grep -v pending | head -5" | \
  xargs -I {} scp root@your-vps-ip:{} ~/Downloads/

# 3. 使用 rsync 同步（推荐）
rsync -avz --progress --exclude='*_pending.jsonl.zst' \
  root@your-vps-ip:/data/infinite_game/segments/ \
  ~/Downloads/infinite_game_segments/
```
