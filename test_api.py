"""
API 测试脚本 - 验证多步预测 API 工作正常
"""
import sys
import logging
from datetime import datetime, timedelta
from model_api import ModelAPI
from config import TrainingConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_predict_regimes():
    """测试 predict_regimes() API"""
    print("\n" + "="*80)
    print("测试 1: predict_regimes() - 多步预测 API")
    print("="*80)
    
    try:
        api = ModelAPI()
        
        # 检查可用模型
        available = api.list_available_models()
        if not available:
            print("❌ 没有可用的模型，请先训练模型")
            return False
        
        symbol = available[0]
        print(f"\n使用交易对: {symbol}")
        
        # 测试多步预测
        result = api.predict_regimes(
            symbol=symbol,
            primary_timeframe="15m",
            include_history=True,
            history_bars=16
        )
        
        # 验证返回结构
        assert 'symbol' in result, "缺少 'symbol' 字段"
        assert 'timeframe' in result, "缺少 'timeframe' 字段"
        assert 'timestamp' in result, "缺少 'timestamp' 字段"
        assert 'predictions' in result, "缺少 'predictions' 字段"
        assert 'is_multistep' in result, "缺少 'is_multistep' 字段"
        assert result['is_multistep'] == True, "is_multistep 应该为 True"
        
        # 验证多步预测
        predictions = result['predictions']
        assert 't+1' in predictions, "缺少 t+1 预测"
        assert 't+2' in predictions, "缺少 t+2 预测"
        assert 't+3' in predictions, "缺少 t+3 预测"
        assert 't+4' in predictions, "缺少 t+4 预测"
        
        # 验证每个预测的结构
        for horizon in ['t+1', 't+2', 't+3', 't+4']:
            pred = predictions[horizon]
            assert 'probabilities' in pred, f"{horizon} 缺少 'probabilities'"
            assert 'most_likely' in pred, f"{horizon} 缺少 'most_likely'"
            assert 'confidence' in pred, f"{horizon} 缺少 'confidence'"
            assert 'is_uncertain' in pred, f"{horizon} 缺少 'is_uncertain'"
            
            # 验证概率和为1
            prob_sum = sum(pred['probabilities'].values())
            assert abs(prob_sum - 1.0) < 0.01, f"{horizon} 概率和不为1: {prob_sum}"
        
        # 验证历史序列
        if 'historical_regimes' in result:
            hist = result['historical_regimes']
            assert 'sequence' in hist, "历史序列缺少 'sequence'"
            assert 'lookback_hours' in hist, "历史序列缺少 'lookback_hours'"
        
        print("\n✅ predict_regimes() 测试通过!")
        print(f"  - 交易对: {result['symbol']}")
        print(f"  - 时间框架: {result['timeframe']}")
        print(f"  - 多步预测: {result['is_multistep']}")
        print(f"  - t+1 预测: {predictions['t+1']['most_likely']} ({predictions['t+1']['confidence']:.2%})")
        print(f"  - t+2 预测: {predictions['t+2']['most_likely']} ({predictions['t+2']['confidence']:.2%})")
        print(f"  - t+3 预测: {predictions['t+3']['most_likely']} ({predictions['t+3']['confidence']:.2%})")
        print(f"  - t+4 预测: {predictions['t+4']['most_likely']} ({predictions['t+4']['confidence']:.2%})")
        
        if 'historical_regimes' in result:
            hist = result['historical_regimes']
            print(f"  - 历史序列: {len(hist.get('sequence', []))} 根K线")
        
        return True
        
    except Exception as e:
        print(f"\n❌ predict_regimes() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predict_next_regime():
    """测试 predict_next_regime() API（向后兼容）"""
    print("\n" + "="*80)
    print("测试 2: predict_next_regime() - 向后兼容 API")
    print("="*80)
    
    try:
        api = ModelAPI()
        
        available = api.list_available_models()
        if not available:
            print("❌ 没有可用的模型")
            return False
        
        symbol = available[0]
        
        result = api.predict_next_regime(
            symbol=symbol,
            primary_timeframe="15m"
        )
        
        # 验证返回结构
        assert 'symbol' in result, "缺少 'symbol' 字段"
        assert 'timeframe' in result, "缺少 'timeframe' 字段"
        assert 'regime_probabilities' in result, "缺少 'regime_probabilities' 字段"
        assert 'most_likely_regime' in result, "缺少 'most_likely_regime' 字段"
        assert 'confidence' in result, "缺少 'confidence' 字段"
        
        # 验证概率和为1
        prob_sum = sum(result['regime_probabilities'].values())
        assert abs(prob_sum - 1.0) < 0.01, f"概率和不为1: {prob_sum}"
        
        print("\n✅ predict_next_regime() 测试通过!")
        print(f"  - 交易对: {result['symbol']}")
        print(f"  - 最可能状态: {result['most_likely_regime']['name']}")
        print(f"  - 概率: {result['most_likely_regime']['probability']:.2%}")
        print(f"  - 置信度: {result['confidence']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ predict_next_regime() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predict_multi_timeframe_regimes():
    """测试 predict_multi_timeframe_regimes() API"""
    print("\n" + "="*80)
    print("测试 3: predict_multi_timeframe_regimes() - 多时间框架多步预测")
    print("="*80)
    
    try:
        api = ModelAPI()
        
        available = api.list_available_models()
        if not available:
            print("❌ 没有可用的模型")
            return False
        
        symbol = available[0]
        
        # 测试多时间框架预测
        result = api.predict_multi_timeframe_regimes(
            symbol=symbol,
            timeframes=["15m"],  # 只测试一个时间框架
            include_history=True
        )
        
        assert 'symbol' in result, "缺少 'symbol' 字段"
        assert 'regimes' in result, "缺少 'regimes' 字段"
        
        for tf, regime_result in result['regimes'].items():
            if 'error' in regime_result:
                print(f"  ⚠️ {tf} 时间框架: {regime_result['error']}")
                continue
            
            assert 'predictions' in regime_result, f"{tf} 缺少 'predictions'"
            assert 't+1' in regime_result['predictions'], f"{tf} 缺少 t+1 预测"
        
        print("\n✅ predict_multi_timeframe_regimes() 测试通过!")
        print(f"  - 交易对: {result['symbol']}")
        print(f"  - 时间框架数量: {len(result['regimes'])}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ predict_multi_timeframe_regimes() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_model_metadata():
    """测试 get_model_metadata() API"""
    print("\n" + "="*80)
    print("测试 4: get_model_metadata() - 模型元数据")
    print("="*80)
    
    try:
        api = ModelAPI()
        
        available = api.list_available_models()
        if not available:
            print("❌ 没有可用的模型")
            return False
        
        symbol = available[0]
        
        metadata = api.get_model_metadata(symbol, primary_timeframe="15m")
        
        assert 'symbol' in metadata, "缺少 'symbol' 字段"
        assert 'n_states' in metadata, "缺少 'n_states' 字段"
        assert 'regime_mapping' in metadata, "缺少 'regime_mapping' 字段"
        assert 'regime_names' in metadata, "缺少 'regime_names' 字段"
        
        print("\n✅ get_model_metadata() 测试通过!")
        print(f"  - 交易对: {metadata['symbol']}")
        print(f"  - 状态数量: {metadata['n_states']}")
        print(f"  - 状态映射: {metadata['regime_mapping']}")
        print(f"  - 状态名称: {metadata['regime_names']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ get_model_metadata() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_regime_history():
    """测试 get_regime_history() API"""
    print("\n" + "="*80)
    print("测试 5: get_regime_history() - 历史regime序列")
    print("="*80)
    
    try:
        api = ModelAPI()
        
        available = api.list_available_models()
        if not available:
            print("❌ 没有可用的模型")
            return False
        
        symbol = available[0]
        
        # 测试1: 按回看小时数
        print(f"\n测试1: 按回看小时数查询（24小时）")
        result1 = api.get_regime_history(
            symbol=symbol,
            lookback_hours=24,
            primary_timeframe="15m"
        )
        
        assert 'symbol' in result1, "缺少 'symbol' 字段"
        assert 'timeframe' in result1, "缺少 'timeframe' 字段"
        assert 'history' in result1, "缺少 'history' 字段"
        assert 'count' in result1, "缺少 'count' 字段"
        assert isinstance(result1['history'], list), "'history' 应该是列表"
        
        print(f"  ✅ 按回看小时数查询成功")
        print(f"  - 交易对: {result1['symbol']}")
        print(f"  - 时间框架: {result1['timeframe']}")
        print(f"  - 回看小时数: {result1['lookback_hours']}")
        print(f"  - 记录数量: {result1['count']}")
        
        if result1['count'] > 0:
            first_record = result1['history'][0]
            assert 'timestamp' in first_record, "历史记录缺少 'timestamp'"
            assert 'regime_name' in first_record, "历史记录缺少 'regime_name'"
            assert 'confidence' in first_record, "历史记录缺少 'confidence'"
            print(f"  - 第一条记录: {first_record['timestamp']} -> {first_record['regime_name']} ({first_record['confidence']:.2%})")
        
        # 测试2: 按日期范围查询
        print(f"\n测试2: 按日期范围查询（最近7天）")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        result2 = api.get_regime_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            primary_timeframe="15m"
        )
        
        assert 'symbol' in result2, "缺少 'symbol' 字段"
        assert 'start_date' in result2, "缺少 'start_date' 字段"
        assert 'end_date' in result2, "缺少 'end_date' 字段"
        assert 'history' in result2, "缺少 'history' 字段"
        
        print(f"  ✅ 按日期范围查询成功")
        print(f"  - 交易对: {result2['symbol']}")
        print(f"  - 时间框架: {result2['timeframe']}")
        print(f"  - 开始日期: {result2['start_date']}")
        print(f"  - 结束日期: {result2['end_date']}")
        print(f"  - 记录数量: {result2['count']}")
        
        print("\n✅ get_regime_history() 测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ get_regime_history() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_predict():
    """测试 batch_predict() API"""
    print("\n" + "="*80)
    print("测试 6: batch_predict() - 批量预测")
    print("="*80)
    
    try:
        api = ModelAPI()
        
        available = api.list_available_models()
        if not available:
            print("❌ 没有可用的模型")
            return False
        
        # 只测试第一个可用的交易对
        symbols = [available[0]]
        
        results = api.batch_predict(
            symbols=symbols,
            primary_timeframe="15m"
        )
        
        assert len(results) == len(symbols), "返回结果数量不匹配"
        
        for symbol, result in results.items():
            if 'error' in result:
                print(f"  ⚠️ {symbol}: {result['error']}")
                continue
            
            assert 'most_likely_regime' in result, f"{symbol} 缺少 'most_likely_regime'"
        
        print("\n✅ batch_predict() 测试通过!")
        print(f"  - 预测交易对数量: {len(results)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ batch_predict() 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*80)
    print("API 多步预测功能测试")
    print("="*80)
    
    results = []
    
    # 运行所有测试
    results.append(("predict_regimes", test_predict_regimes()))
    results.append(("predict_next_regime", test_predict_next_regime()))
    results.append(("predict_multi_timeframe_regimes", test_predict_multi_timeframe_regimes()))
    results.append(("get_model_metadata", test_get_model_metadata()))
    results.append(("get_regime_history", test_get_regime_history()))
    results.append(("batch_predict", test_batch_predict()))
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有 API 测试通过!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
