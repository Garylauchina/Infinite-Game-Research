#!/usr/bin/env python3
"""从 WSS 接口获取连续 500 个 tick 数据"""
import asyncio
import websockets
import ssl
import json
import sys
from datetime import datetime

ssl_context = ssl.SSLContext()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def fetch_ticks(output_file: str, count: int = 500):
    """获取连续的 tick 数据"""
    uri = "wss://45.76.97.37/ws/stream"
    
    print("=" * 60)
    print(f"从 WSS 接口获取 {count} 个连续 tick 数据")
    print("=" * 60)
    print(f"连接地址: {uri}")
    print(f"输出文件: {output_file}")
    print()
    
    ticks = []
    start_t = None
    
    try:
        async with websockets.connect(uri, ssl=ssl_context) as ws:
            print("✅ WebSocket 连接成功")
            print("开始接收数据...")
            print("-" * 60)
            
            while len(ticks) < count:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    data = json.loads(msg)
                    
                    if data.get('type') == 'tick':
                        if start_t is None:
                            start_t = data.get('t')
                            print(f"开始 tick: t={start_t}")
                        
                        ticks.append(data)
                        
                        if len(ticks) % 50 == 0:
                            print(f"已接收: {len(ticks)}/{count} ticks (当前 t={data.get('t')})")
                        
                        if len(ticks) >= count:
                            break
                    elif data.get('type') == 'heartbeat':
                        # 忽略心跳
                        pass
                    else:
                        print(f"收到非 tick 消息: {data.get('type')}")
                        
                except asyncio.TimeoutError:
                    print("⚠️  60 秒内未收到消息，停止接收")
                    break
                except Exception as e:
                    print(f"❌ 接收消息错误: {e}")
                    break
            
            print()
            print("-" * 60)
            print(f"接收完成: {len(ticks)} 个 tick")
            
            if ticks:
                end_t = ticks[-1].get('t')
                print(f"Tick 范围: {start_t}..{end_t}")
                print(f"时间跨度: {ticks[-1].get('ts_wall', 0) - ticks[0].get('ts_wall', 0):.2f} 秒")
            
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 保存数据
    if not ticks:
        print("❌ 未收到任何 tick 数据")
        return False
    
    try:
        # 保存为 JSON 数组格式
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ticks, f, indent=2, ensure_ascii=False)
        
        file_size = len(json.dumps(ticks, ensure_ascii=False).encode('utf-8'))
        print(f"✅ 数据已保存到: {output_file}")
        print(f"   文件大小: {file_size / 1024:.2f} KB ({file_size / 1024 / 1024:.2f} MB)")
        print(f"   Tick 数量: {len(ticks)}")
        
        # 显示第一个和最后一个 tick 的摘要
        if len(ticks) > 0:
            first = ticks[0]
            last = ticks[-1]
            print()
            print("数据摘要:")
            print(f"  第一个 tick: t={first.get('t')}, ts_wall={first.get('ts_wall')}")
            if first.get('frame'):
                print(f"    depth: bids={len(first['frame'].get('depth', {}).get('bids', []))}, asks={len(first['frame'].get('depth', {}).get('asks', []))}")
                print(f"    trades: {len(first['frame'].get('trades', []))}")
            print(f"  最后一个 tick: t={last.get('t')}, ts_wall={last.get('ts_wall')}")
            if last.get('frame'):
                print(f"    depth: bids={len(last['frame'].get('depth', {}).get('bids', []))}, asks={len(last['frame'].get('depth', {}).get('asks', []))}")
                print(f"    trades: {len(last['frame'].get('trades', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    output_file = sys.argv[1] if len(sys.argv) > 1 else "wss_ticks_500.json"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    success = asyncio.run(fetch_ticks(output_file, count))
    sys.exit(0 if success else 1)
