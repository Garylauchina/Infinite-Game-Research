#!/usr/bin/env python3
"""
测试 graph_snapshot 数据输出
"""
import json
import time
import websocket
import threading

received_snapshots = []
received_updates = []

def on_message(ws, message):
    try:
        data = json.loads(message)
        if data.get('last'):
            last = data['last']
            t = last.get('t', 0)
            
            # 检查是否有 graph_snapshot
            if 'graph_snapshot' in last:
                snapshot = last['graph_snapshot']
                received_snapshots.append({
                    't': t,
                    'snapshot': snapshot
                })
                print(f"\n[✓] 收到 graph_snapshot @ t={t}")
                print(f"    - n_clusters: {snapshot.get('n_clusters')}")
                print(f"    - active_protocols: {snapshot.get('active_protocols')}")
                print(f"    - protocol_score: {snapshot.get('protocol_score', 0):.3f}")
                print(f"    - transfer_entropy: {snapshot.get('transfer_entropy', 0):.3f}")
                print(f"    - uniformity: {snapshot.get('uniformity', 0):.3f}")
                print(f"    - counts: {snapshot.get('counts', [])}")
                if len(snapshot.get('transition_matrix', [])) > 0:
                    print(f"    - transition_matrix: {len(snapshot['transition_matrix'])}x{len(snapshot['transition_matrix'][0])}")
            else:
                # 普通更新
                cluster_id = last.get('cluster_id')
                if cluster_id is not None:
                    received_updates.append({
                        't': t,
                        'cluster_id': cluster_id,
                        'matches': last.get('matches', 0)
                    })
                    if len(received_updates) % 10 == 0:
                        print(f"[.] 更新 @ t={t}, cluster_id={cluster_id}, matches={last.get('matches', 0)}", end='', flush=True)
    except Exception as e:
        print(f"\n[ERROR] {e}")

def on_error(ws, error):
    print(f"\n[ERROR] WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("\n[INFO] WebSocket closed")

def on_open(ws):
    print("[INFO] WebSocket connected, waiting for data...")

if __name__ == "__main__":
    print("=" * 60)
    print("测试 graph_snapshot 数据输出")
    print("=" * 60)
    
    ws_url = "ws://localhost:8000/ws"
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    
    # 运行30秒
    def run_ws():
        ws.run_forever()
    
    thread = threading.Thread(target=run_ws, daemon=True)
    thread.start()
    
    try:
        time.sleep(30)  # 等待30秒收集数据
        ws.close()
        
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        print(f"收到 graph_snapshot 数量: {len(received_snapshots)}")
        print(f"收到普通更新数量: {len(received_updates)}")
        
        if received_snapshots:
            print("\n第一个 graph_snapshot 详情:")
            first = received_snapshots[0]['snapshot']
            print(json.dumps(first, indent=2, ensure_ascii=False))
            
            if len(received_snapshots) > 1:
                print("\n最后一个 graph_snapshot 详情:")
                last = received_snapshots[-1]['snapshot']
                print(json.dumps(last, indent=2, ensure_ascii=False))
        else:
            print("\n⚠️  未收到任何 graph_snapshot！")
            print("   可能原因：")
            print("   1. 数据还未达到200个tick（需要初始化）")
            print("   2. 聚类更新间隔未到（每500 tick）")
            print("   3. 后端代码未正确生成 graph_snapshot")
        
    except KeyboardInterrupt:
        ws.close()
        print("\n[INFO] 测试中断")
