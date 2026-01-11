#!/usr/bin/env python3
"""
快速测试历史regime API接口
"""
import sys
import requests
import json
from datetime import datetime, timedelta

def test_history_api():
    """测试历史regime API"""
    base_url = "http://localhost:5858"
    
    print("="*80)
    print("测试历史regime API接口")
    print("="*80)
    
    # 测试1: 按回看小时数
    print("\n测试1: 按回看小时数查询")
    print(f"请求: GET {base_url}/api/history/BTCUSDT?timeframe=15m&lookback_hours=24")
    
    try:
        response = requests.get(
            f"{base_url}/api/history/BTCUSDT",
            params={
                "timeframe": "15m",
                "lookback_hours": 24
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功!")
            print(f"  - 交易对: {data.get('symbol')}")
            print(f"  - 时间框架: {data.get('timeframe')}")
            print(f"  - 回看小时数: {data.get('lookback_hours')}")
            print(f"  - 记录数量: {data.get('count')}")
            
            if data.get('count', 0) > 0:
                first = data['history'][0]
                last = data['history'][-1]
                print(f"  - 第一条: {first.get('timestamp')} -> {first.get('regime_name')}")
                print(f"  - 最后一条: {last.get('timestamp')} -> {last.get('regime_name')}")
        else:
            print(f"❌ 失败: HTTP {response.status_code}")
            print(f"  错误: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败: 请确保API服务器正在运行")
        print("   启动服务器: python run_server.py")
        return False
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    # 测试2: 按日期范围
    print("\n测试2: 按日期范围查询")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"请求: GET {base_url}/api/history/BTCUSDT?timeframe=15m&start_date={start_str}&end_date={end_str}")
    
    try:
        response = requests.get(
            f"{base_url}/api/history/BTCUSDT",
            params={
                "timeframe": "15m",
                "start_date": start_str,
                "end_date": end_str
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 成功!")
            print(f"  - 交易对: {data.get('symbol')}")
            print(f"  - 时间框架: {data.get('timeframe')}")
            print(f"  - 开始日期: {data.get('start_date')}")
            print(f"  - 结束日期: {data.get('end_date')}")
            print(f"  - 记录数量: {data.get('count')}")
            
            if data.get('count', 0) > 0:
                first = data['history'][0]
                last = data['history'][-1]
                print(f"  - 第一条: {first.get('timestamp')} -> {first.get('regime_name')}")
                print(f"  - 最后一条: {last.get('timestamp')} -> {last.get('regime_name')}")
        else:
            print(f"❌ 失败: HTTP {response.status_code}")
            print(f"  错误: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ 所有测试通过!")
    print("="*80)
    return True


if __name__ == "__main__":
    success = test_history_api()
    sys.exit(0 if success else 1)
