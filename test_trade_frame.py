#!/usr/bin/env python3
"""测试交易展示帧数据结构"""
import asyncio
import websockets
import ssl
import json
import time

ssl_context = ssl.SSLContext()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def validate_tick(tick):
    """验证 tick 消息结构"""
    errors = []
    
    # 检查必需字段
    required_fields = ["type", "schema", "t", "ts_wall"]
    for field in required_fields:
        if field not in tick:
            errors.append(f"缺少必需字段: {field}")
    
    # 检查 schema
    if "schema" in tick:
        schema = tick["schema"]
        if schema.get("name") != "ig_wss":
            errors.append(f"Schema name 错误: {schema.get('name')}")
        if schema.get("version") != 1:
            errors.append(f"Schema version 错误: {schema.get('version')}")
    
    # 检查 frame
    if "frame" not in tick:
        errors.append("缺少 frame 字段")
    else:
        frame = tick["frame"]
        
        # 检查 depth
        if "depth" not in frame:
            errors.append("frame 缺少 depth 字段")
        else:
            depth = frame["depth"]
            if "bids" not in depth or "asks" not in depth:
                errors.append("depth 缺少 bids 或 asks")
            else:
                # 验证 bids 降序
                bids = depth["bids"]
                if len(bids) > 1:
                    for i in range(len(bids) - 1):
                        if bids[i][0] < bids[i+1][0]:
                            errors.append(f"bids 未按降序排列: {bids[i][0]} < {bids[i+1][0]}")
                
                # 验证 asks 升序
                asks = depth["asks"]
                if len(asks) > 1:
                    for i in range(len(asks) - 1):
                        if asks[i][0] > asks[i+1][0]:
                            errors.append(f"asks 未按升序排列: {asks[i][0]} > {asks[i+1][0]}")
                
                # 验证 mid/spread
                if depth.get("best_bid") and depth.get("best_ask"):
                    expected_mid = (depth["best_bid"] + depth["best_ask"]) / 2.0
                    expected_spread = depth["best_ask"] - depth["best_bid"]
                    if abs(depth.get("mid", 0) - expected_mid) > 0.01:
                        errors.append(f"mid 计算错误: {depth.get('mid')} != {expected_mid}")
                    if abs(depth.get("spread", 0) - expected_spread) > 0.01:
                        errors.append(f"spread 计算错误: {depth.get('spread')} != {expected_spread}")
        
        # 检查 trades
        if "trades" in frame:
            trades = frame["trades"]
            for i, trade in enumerate(trades):
                required_trade_fields = ["trade_id", "t", "ts_wall", "side", "price", "size", "a", "mode"]
                for field in required_trade_fields:
                    if field not in trade:
                        errors.append(f"trade[{i}] 缺少字段: {field}")
                
                # 验证 trade_id 格式
                if "trade_id" in trade:
                    trade_id = trade["trade_id"]
                    if not trade_id.startswith(f"t:{trade['t']}-"):
                        errors.append(f"trade[{i}] trade_id 格式错误: {trade_id}")
        
        # 检查 ohlcv
        if "ohlcv" in frame:
            ohlcv = frame["ohlcv"]
            required_ohlcv_fields = ["tf", "open", "high", "low", "close", "volume", "trades", "vwap"]
            for field in required_ohlcv_fields:
                if field not in ohlcv:
                    errors.append(f"ohlcv 缺少字段: {field}")
            
            # 验证 OHLC 逻辑
            if "high" in ohlcv and "low" in ohlcv and "open" in ohlcv and "close" in ohlcv:
                if ohlcv["high"] < ohlcv["low"]:
                    errors.append("ohlcv high < low")
                if ohlcv["high"] < ohlcv["open"] or ohlcv["high"] < ohlcv["close"]:
                    errors.append("ohlcv high 小于 open 或 close")
                if ohlcv["low"] > ohlcv["open"] or ohlcv["low"] > ohlcv["close"]:
                    errors.append("ohlcv low 大于 open 或 close")
    
    return errors

async def test_trade_frame():
    uri = "wss://45.76.97.37/ws/stream"
    
    print("=" * 60)
    print("交易展示帧数据结构测试")
    print("=" * 60)
    print(f"连接地址: {uri}")
    print()
    
    try:
        async with websockets.connect(uri, ssl=ssl_context) as ws:
            print("✅ WebSocket 连接成功")
            print()
            
            tick_count = 0
            depth_count = 0
            trades_count = 0
            ohlcv_count = 0
            error_count = 0
            start_time = time.time()
            
            print("开始接收消息（30 秒）...")
            print("-" * 60)
            
            try:
                while True:
                    timeout = 35.0
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        data = json.loads(msg)
                        
                        if data.get('type') == 'tick':
                            tick_count += 1
                            
                            # 验证结构
                            errors = validate_tick(data)
                            if errors:
                                error_count += 1
                                print(f"❌ Tick {tick_count} 验证失败:")
                                for err in errors[:5]:  # 只显示前5个错误
                                    print(f"   - {err}")
                                if len(errors) > 5:
                                    print(f"   ... 还有 {len(errors) - 5} 个错误")
                            else:
                                # 统计
                                if data.get("frame", {}).get("depth"):
                                    depth_count += 1
                                if data.get("frame", {}).get("trades"):
                                    trades_count += len(data["frame"]["trades"])
                                if data.get("frame", {}).get("ohlcv"):
                                    ohlcv_count += 1
                                
                                # 每 100 个 tick 显示一次
                                if tick_count % 100 == 0:
                                    elapsed = time.time() - start_time
                                    print(f"[{elapsed:6.1f}s] Tick {tick_count}: depth={depth_count}, trades={trades_count}, ohlcv={ohlcv_count}, errors={error_count}")
                        
                        elif data.get('type') == 'heartbeat':
                            # 心跳消息，忽略
                            pass
                        
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
            print(f"  有 depth 的 tick: {depth_count}")
            print(f"  总成交笔数: {trades_count}")
            print(f"  有 ohlcv 的 tick: {ohlcv_count}")
            print(f"  验证错误数: {error_count}")
            
            if error_count == 0 and tick_count > 0:
                print()
                print("✅ 所有 tick 验证通过！")
            elif error_count > 0:
                print()
                print(f"⚠️  发现 {error_count} 个验证错误")
            else:
                print()
                print("⚠️  未收到任何 tick 消息")
                
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_trade_frame())
