#!/usr/bin/env python3
"""测试 WebSocket 连接并监控 tick 消息"""
import asyncio
import websockets
import ssl
import json
import time
from datetime import datetime

ssl_context = ssl.SSLContext()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def test_websocket():
    uri = "wss://45.76.97.37/ws/stream"
    
    print("=" * 60)
    print("WebSocket Tick 广播测试")
    print("=" * 60)
    print(f"连接地址: {uri}")
    print()
    
    try:
        async with websockets.connect(uri, ssl=ssl_context) as ws:
            print("✅ WebSocket 连接成功")
            print()
            
            tick_count = 0
            heartbeat_count = 0
            last_tick_t = None
            start_time = time.time()
            
            print("开始接收消息（30 秒）...")
            print("-" * 60)
            
            try:
                while True:
                    timeout = 35.0  # 稍微长于心跳间隔
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        data = json.loads(msg)
                        msg_type = data.get('type', 'unknown')
                        
                        if msg_type == 'tick':
                            tick_count += 1
                            tick_t = data.get('t', 'unknown')
                            
                            if last_tick_t is not None:
                                gap = tick_t - last_tick_t if isinstance(tick_t, int) and isinstance(last_tick_t, int) else None
                                if gap and gap > 1:
                                    print(f"⚠️  Tick 间隔: {gap} (可能丢失了 {gap-1} 个 tick)")
                            
                            last_tick_t = tick_t
                            
                            elapsed = time.time() - start_time
                            print(f"[{elapsed:6.1f}s] Tick {tick_count}: t={tick_t}, agents={len(data.get('agents', []))}")
                            
                        elif msg_type == 'heartbeat':
                            heartbeat_count += 1
                            tick_t = data.get('t', 'unknown')
                            elapsed = time.time() - start_time
                            
                            if heartbeat_count == 1:
                                print(f"[{elapsed:6.1f}s] 第一个心跳: t={tick_t}")
                            elif heartbeat_count % 5 == 0:
                                print(f"[{elapsed:6.1f}s] 心跳 {heartbeat_count}: t={tick_t}")
                            
                            # 检查 tick 是否在增长
                            if last_tick_t is not None and isinstance(tick_t, int) and isinstance(last_tick_t, int):
                                if tick_t > last_tick_t:
                                    missing = tick_t - last_tick_t - 1
                                    if missing > 0:
                                        print(f"⚠️  心跳显示 tick 从 {last_tick_t} 增长到 {tick_t}，但未收到 {missing} 个 tick 消息")
                            
                        elif msg_type == 'pong':
                            print(f"[{time.time()-start_time:6.1f}s] 收到 pong")
                        else:
                            print(f"[{time.time()-start_time:6.1f}s] 未知消息类型: {msg_type}")
                        
                        # 30 秒后停止
                        if time.time() - start_time >= 30:
                            break
                            
                    except asyncio.TimeoutError:
                        print(f"⚠️  30 秒内未收到任何消息")
                        break
                        
            except Exception as e:
                print(f"❌ 接收消息错误: {e}")
            
            elapsed = time.time() - start_time
            print()
            print("-" * 60)
            print("测试结果:")
            print(f"  运行时间: {elapsed:.1f} 秒")
            print(f"  收到 Tick 消息: {tick_count}")
            print(f"  收到心跳消息: {heartbeat_count}")
            print(f"  最后 Tick: t={last_tick_t}")
            
            if tick_count == 0:
                print()
                print("❌ 问题: 未收到任何 tick 消息")
                print("   可能原因:")
                print("   1. write_tick_record 未被调用")
                print("   2. 广播逻辑未执行")
                print("   3. main_event_loop 未正确设置")
            elif tick_count < 10:
                print()
                print("⚠️  警告: tick 消息过少")
                print(f"   30 秒内只收到 {tick_count} 个 tick，预期应该更多")
            else:
                print()
                print("✅ Tick 消息正常接收")
                
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket())
