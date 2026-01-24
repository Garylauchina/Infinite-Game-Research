#!/usr/bin/env python3
"""清理旧数据，保留最新的连续运行数据"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = Path(os.environ.get("INFINITE_DATA_DIR", "/data/infinite_game"))
SEGMENTS_DIR = DATA_DIR / "segments"
INDEX_DIR = DATA_DIR / "index"
SEGMENTS_JSON = INDEX_DIR / "segments.json"

def find_latest_continuous_run(segments):
    """找到最新的连续运行（基于创建时间和 tick 连续性）"""
    if not segments:
        return []
    
    # 按创建时间排序（最新的在前）
    segments_sorted = sorted(segments, key=lambda x: x.get("created_at", ""), reverse=True)
    
    # 找到最新的连续段序列
    latest_run = []
    if segments_sorted:
        # 从最新的段开始，向前查找连续的段
        latest_seg = segments_sorted[0]
        latest_run.append(latest_seg)
        
        # 查找连续的段（end_t + 1 == next_start_t）
        current_end_t = latest_seg.get("end_t", -1)
        
        for seg in segments_sorted[1:]:
            seg_start_t = seg.get("start_t", -1)
            seg_end_t = seg.get("end_t", -1)
            
            # 检查是否连续
            if seg_end_t + 1 == current_end_t or seg_start_t <= current_end_t:
                latest_run.append(seg)
                current_end_t = min(current_end_t, seg_start_t)
            else:
                # 不连续，停止查找
                break
    
    return latest_run

def cleanup_old_data():
    """清理旧数据，只保留最新的连续运行"""
    print("=" * 60)
    print("清理旧数据")
    print("=" * 60)
    print()
    
    if not SEGMENTS_JSON.exists():
        print("❌ segments.json 不存在，无需清理")
        return
    
    # 读取 segments.json
    try:
        with open(SEGMENTS_JSON, 'r') as f:
            segments = json.load(f)
    except Exception as e:
        print(f"❌ 读取 segments.json 失败: {e}")
        return
    
    if not isinstance(segments, list):
        print("❌ segments.json 格式错误")
        return
    
    if len(segments) == 0:
        print("✅ 没有数据需要清理")
        return
    
    print(f"当前共有 {len(segments)} 个段")
    
    # 找到最新的连续运行
    latest_run = find_latest_continuous_run(segments)
    
    if not latest_run:
        print("❌ 未找到有效的连续运行")
        return
    
    print(f"最新连续运行包含 {len(latest_run)} 个段")
    if latest_run:
        latest_seg = latest_run[0]
        oldest_seg = latest_run[-1]
        print(f"  Tick 范围: {oldest_seg.get('start_t', 0)}..{latest_seg.get('end_t', 0)}")
        print(f"  最新段创建时间: {latest_seg.get('created_at', 'unknown')}")
    
    # 计算要删除的段
    segments_to_keep = {seg.get("path") for seg in latest_run}
    segments_to_delete = [seg for seg in segments if seg.get("path") not in segments_to_keep]
    
    if not segments_to_delete:
        print("✅ 没有需要删除的旧数据")
        return
    
    print(f"\n将删除 {len(segments_to_delete)} 个旧段，保留 {len(latest_run)} 个段")
    
    # 确认删除
    response = input("\n确认删除? (y/N): ").strip().lower()
    if response != 'y':
        print("取消删除")
        return
    
    # 删除文件
    deleted_count = 0
    deleted_bytes = 0
    
    for seg in segments_to_delete:
        seg_path = DATA_DIR / seg.get("path", "")
        if seg_path.exists():
            try:
                file_size = seg_path.stat().st_size
                seg_path.unlink()
                deleted_count += 1
                deleted_bytes += file_size
                print(f"  删除: {seg_path.name} ({file_size/1024/1024:.2f} MB)")
            except Exception as e:
                print(f"  ❌ 删除失败 {seg_path.name}: {e}")
        else:
            print(f"  ⚠️  文件不存在: {seg_path.name}")
            deleted_count += 1
    
    # 更新 segments.json
    try:
        with open(SEGMENTS_JSON, 'w') as f:
            json.dump(latest_run, f, indent=2)
        print(f"\n✅ 更新 segments.json: 保留 {len(latest_run)} 个段")
    except Exception as e:
        print(f"\n❌ 更新 segments.json 失败: {e}")
        return
    
    print(f"\n✅ 清理完成:")
    print(f"  删除段数: {deleted_count}")
    print(f"  释放空间: {deleted_bytes/1024/1024:.2f} MB ({deleted_bytes/1024/1024/1024:.2f} GB)")
    print(f"  保留段数: {len(latest_run)}")

if __name__ == "__main__":
    cleanup_old_data()
