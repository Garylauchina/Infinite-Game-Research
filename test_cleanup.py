#!/usr/bin/env python3
"""测试滚动存储清理功能"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from experiments.live.server import get_disk_usage, cleanup_old_segments, DATA_DIR, SEGMENTS_DIR, INDEX_DIR
import json
from pathlib import Path

def main():
    print("=" * 60)
    print("滚动存储清理功能测试")
    print("=" * 60)
    print()
    
    # 1. 检查磁盘使用率
    print("1. 检查磁盘使用率...")
    usage_percent, total, used, free = get_disk_usage()
    if usage_percent is None:
        print("❌ 无法获取磁盘使用率")
        return
    
    print(f"   磁盘使用率: {usage_percent:.2f}%")
    print(f"   总容量: {total / (1024**3):.2f} GB")
    print(f"   已使用: {used / (1024**3):.2f} GB")
    print(f"   可用空间: {free / (1024**3):.2f} GB")
    print()
    
    # 2. 检查段文件
    print("2. 检查段文件...")
    segments_json = INDEX_DIR / "segments.json"
    if not segments_json.exists():
        print("❌ segments.json 不存在")
        return
    
    with open(segments_json, 'r') as f:
        segments = json.load(f)
    
    if not isinstance(segments, list):
        print("❌ segments.json 格式错误")
        return
    
    print(f"   段文件数量: {len(segments)}")
    
    total_bytes = sum(s.get("bytes", 0) for s in segments)
    print(f"   总大小: {total_bytes / (1024**3):.2f} GB")
    print()
    
    # 3. 显示最旧的段
    if len(segments) > 0:
        segments_sorted = sorted(segments, key=lambda x: x.get("created_at", ""))
        print("3. 最旧的 5 个段:")
        for i, seg in enumerate(segments_sorted[:5]):
            print(f"   {i+1}. {seg.get('path', 'unknown')}")
            print(f"      创建时间: {seg.get('created_at', 'unknown')}")
            print(f"      大小: {seg.get('bytes', 0) / (1024**2):.2f} MB")
            print(f"      Tick 范围: {seg.get('start_t', 0)}..{seg.get('end_t', 0)}")
        print()
    
    # 4. 检查是否需要清理
    from experiments.live.server import DISK_USAGE_THRESHOLD
    print(f"4. 清理阈值: {DISK_USAGE_THRESHOLD}%")
    if usage_percent >= DISK_USAGE_THRESHOLD:
        print(f"   ⚠️  磁盘使用率 {usage_percent:.2f}% >= 阈值 {DISK_USAGE_THRESHOLD}%，需要清理")
    else:
        print(f"   ✅ 磁盘使用率 {usage_percent:.2f}% < 阈值 {DISK_USAGE_THRESHOLD}%，无需清理")
    print()
    
    # 5. 询问是否执行清理
    if usage_percent >= DISK_USAGE_THRESHOLD:
        response = input("是否执行清理? (y/N): ").strip().lower()
        if response == 'y':
            print()
            print("5. 执行清理...")
            success, msg = cleanup_old_segments()
            if success:
                print(f"   ✅ {msg}")
            else:
                print(f"   ❌ {msg}")
        else:
            print("   跳过清理")
    else:
        print("5. 无需清理")
    
    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
