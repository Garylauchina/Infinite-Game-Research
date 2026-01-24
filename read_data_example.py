#!/usr/bin/env python3
"""
读取 Infinite Game 数据示例
"""
import json
import zstandard as zstd
from pathlib import Path

# 数据目录
DATA_DIR = Path("/data/infinite_game")
SEGMENTS_DIR = DATA_DIR / "segments"
INDEX_DIR = DATA_DIR / "index"

# 1. 读取索引文件
print("=" * 60)
print("1. 读取索引文件 segments.json")
print("=" * 60)
segments_json = INDEX_DIR / "segments.json"
with open(segments_json, 'r') as f:
    segments = json.load(f)

print(f"共有 {len(segments)} 个段")
for i, seg in enumerate(segments[:3]):  # 只显示前3个
    print(f"\n段 {i+1}:")
    print(f"  路径: {seg['path']}")
    print(f"  时间范围: t={seg['start_t']}..{seg['end_t']}")
    print(f"  行数: {seg['lines']}")
    print(f"  压缩大小: {seg['bytes'] / 1024 / 1024:.1f} MB")
    print(f"  创建时间: {seg['created_at']}")

# 2. 读取一个段文件
print("\n" + "=" * 60)
print("2. 读取段文件内容（前3条记录）")
print("=" * 60)
if segments:
    first_seg = segments[0]
    seg_path = DATA_DIR / first_seg['path']
    
    print(f"\n读取: {seg_path}")
    
    dctx = zstd.ZstdDecompressor()
    with open(seg_path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            import io
            text_reader = io.TextIOWrapper(reader, encoding='utf-8')
            
            # 读取 header
            header_line = text_reader.readline()
            if header_line:
                header = json.loads(header_line.strip())
                print(f"\nHeader:")
                print(f"  type: {header.get('type')}")
                print(f"  schema: {header.get('schema')}")
                print(f"  start_t: {header.get('start_t')}")
                print(f"  seed: {header.get('seed')}")
            
            # 读取前3个 tick
            print(f"\n前3个 tick 记录:")
            for i in range(3):
                line = text_reader.readline()
                if not line:
                    break
                tick = json.loads(line.strip())
                print(f"\n  Tick {i+1}:")
                print(f"    t: {tick.get('t')}")
                print(f"    agents: {len(tick.get('agents', []))} 个")
                print(f"    actions: {len(tick.get('actions', []))} 个")
                print(f"    matches: {len(tick.get('matches', []))} 个")
                if tick.get('s'):
                    s = tick['s']
                    print(f"    state: price_norm={s.get('price_norm', 0):.3f}, volatility={s.get('volatility', 0):.3f}")

print("\n" + "=" * 60)
print("3. 读取方式总结")
print("=" * 60)
print("""
数据保存位置：
  - 主目录: /data/infinite_game/ (可通过环境变量 INFINITE_DATA_DIR 配置)
  - 段文件: /data/infinite_game/segments/*.jsonl.zst
  - 索引文件: /data/infinite_game/index/segments.json

数据格式：
  - 每个段文件是 zstd 压缩的 JSONL
  - 第一行: segment_header (包含 schema, seed, start_t 等)
  - 后续行: tick 记录 (包含 t, s, agents, actions, matches)

读取方式：
  1. 通过 HTTP API (推荐):
     - GET /segments -> 获取索引
     - GET /segment/<filename> -> 获取段文件流
     - GET /range?t0=<start>&t1=<end> -> 获取时间范围数据
  
  2. 直接读取文件:
     - 使用 zstandard 库解压 .jsonl.zst 文件
     - 逐行解析 JSONL
     - 第一行是 header，后续是 tick 记录

Python 读取示例:
  import zstandard as zstd
  import json
  
  dctx = zstd.ZstdDecompressor()
  with open('segment_xxx.jsonl.zst', 'rb') as f:
      with dctx.stream_reader(f) as reader:
          header = json.loads(reader.readline().decode('utf-8'))
          for line in reader:
              tick = json.loads(line.decode('utf-8'))
              # 处理 tick 数据
""")
