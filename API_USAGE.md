# Model API 使用指南

这个文档说明如何使用 `model_api.py` 模块，让其他项目可以方便地获取 market regime 预测结果。

## 重要说明

**LSTM模型只能预测下一根K线的market regime**。模型使用过去64根K线的特征序列来预测下一根K线的状态。这是单步预测，不能直接预测多根K线。

## 快速开始

### 基本用法

```python
from model_api import ModelAPI, predict_regime, get_regime_probability

# 方式1: 使用 ModelAPI 类
api = ModelAPI()
result = api.predict_next_regime(
    symbol="BTCUSDT",
    timeframe="15m"
)

# 方式2: 使用便捷函数
result = predict_regime("BTCUSDT", "15m")
```

### 返回结果格式

```python
{
    'symbol': 'BTCUSDT',                    # 交易对
    'timeframe': '15m',                      # 时间框架
    'timestamp': datetime(...),              # 预测时间
    'regime_probabilities': {               # 各状态的概率分布
        'Strong_Trend': 0.35,
        'Weak_Trend': 0.25,
        'Range': 0.20,
        'Choppy_High_Vol': 0.10,
        'Volatility_Spike': 0.05,
        'Squeeze': 0.05
    },
    'most_likely_regime': {                 # 最可能的状态
        'id': 1,
        'name': 'Strong_Trend',
        'probability': 0.35
    },
    'confidence': 0.35,                     # 置信度（最高概率）
    'is_uncertain': False,                  # 是否不确定（置信度过低）
    'model_info': {                         # 模型信息
        'primary_timeframe': '15m',
        'n_states': 6,
        'sequence_length': 64,              # 使用的历史K线数量
        'regime_mapping': {0: 'Choppy_High_Vol', 1: 'Strong_Trend', ...}
    }
}
```

## API 方法详解

### 1. 预测下一根K线的market regime

```python
api = ModelAPI()

# 预测下一根15分钟K线的market regime
result = api.predict_next_regime(
    symbol="BTCUSDT",
    timeframe="15m"      # 必须与训练时的主时间框架一致（默认15m）
)

# 获取最可能的状态
most_likely = result['most_likely_regime']
print(f"最可能的状态: {most_likely['name']}")
print(f"概率: {most_likely['probability']:.2%}")

# 获取所有状态的概率分布
probs = result['regime_probabilities']
for regime_name, prob in probs.items():
    print(f"{regime_name}: {prob:.2%}")
```

### 2. 获取特定状态的概率

```python
# 方式1: 使用 ModelAPI
api = ModelAPI()
prob = api.get_regime_probability(
    symbol="BTCUSDT",
    regime_name="Strong_Trend",
    timeframe="15m"
)

# 方式2: 使用便捷函数
prob = get_regime_probability("BTCUSDT", "Strong_Trend")
print(f"Strong_Trend 概率: {prob:.2%}")
```

### 3. 获取模型元数据

```python
api = ModelAPI()
metadata = api.get_model_metadata("BTCUSDT")

print(f"状态数量: {metadata['n_states']}")
print(f"状态映射: {metadata['regime_mapping']}")
print(f"主时间框架: {metadata['primary_timeframe']}")
print(f"所有状态名称: {metadata['regime_names']}")
```

### 4. 列出可用的模型

```python
api = ModelAPI()
available = api.list_available_models()
print(f"可用的交易对: {available}")
# 输出: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', ...]
```

### 5. 批量预测多个交易对

```python
api = ModelAPI()
results = api.batch_predict(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    timeframe="15m"
)

for symbol, result in results.items():
    if 'error' not in result:
        print(f"{symbol}: {result['most_likely_regime']['name']}")
    else:
        print(f"{symbol}: 错误 - {result['error']}")
```

### 6. 获取历史 Market Regime 序列

**新增功能**：获取历史上的 market regime 序列，支持回测场景。

```python
from datetime import datetime, timedelta

api = ModelAPI()

# 方式1: 按回看小时数查询（从当前时间往前回看）
result = api.get_regime_history(
    symbol="BTCUSDT",
    lookback_hours=24,  # 回看24小时
    primary_timeframe="15m"
)

# 方式2: 按日期范围查询（适合回测）
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # 最近30天

result = api.get_regime_history(
    symbol="BTCUSDT",
    start_date=start_date,
    end_date=end_date,
    primary_timeframe="15m"
)

# 返回结果格式
{
    'symbol': 'BTCUSDT',
    'timeframe': '15m',
    'lookback_hours': 24,  # 或 None（如果使用日期范围）
    'start_date': None,     # 或 ISO 格式日期字符串
    'end_date': None,       # 或 ISO 格式日期字符串
    'timestamp': datetime(...),
    'count': 96,            # 记录数量
    'history': [
        {
            'timestamp': '2024-01-01T11:45:00',
            'regime_id': 0,
            'regime_name': 'Range',
            'confidence': 0.85,
            'is_uncertain': False,
            'original_regime': 'Range'
        },
        ...
    ]
}

# 使用示例：遍历历史regime
for record in result['history']:
    print(f"{record['timestamp']}: {record['regime_name']} ({record['confidence']:.2%})")
```

**参数说明**：
- `lookback_hours`: 回看小时数（最大720小时/30天）
- `start_date`: 开始日期时间（datetime对象）
- `end_date`: 结束日期时间（datetime对象，最大范围365天）
- 如果指定了 `start_date` 和 `end_date`，则使用日期范围查询
- 如果只指定了 `lookback_hours`，则从当前时间往前回看
- 如果都不指定，默认回看24小时

**性能优化**：
- 优先从SQLite缓存读取历史K线数据
- 使用批量预测提高性能（一次性预测所有历史样本）
- 适合回测场景，支持长时间范围查询

## 完整示例

```python
from model_api import ModelAPI
from datetime import datetime

def check_market_regime(symbol: str):
    """检查市场状态并做出决策"""
    api = ModelAPI()
    
    # 预测下一根K线的market regime
    result = api.predict_next_regime(symbol, "15m")
    
    # 获取最可能的状态
    regime = result['most_likely_regime']
    confidence = result['confidence']
    
    print(f"\n{symbol} 市场状态分析 ({datetime.now()})")
    print("=" * 60)
    print(f"预测: 下一根15分钟K线")
    print(f"最可能状态: {regime['name']}")
    print(f"概率: {regime['probability']:.2%}")
    print(f"置信度: {confidence:.2%}")
    
    # 根据状态做出决策
    if regime['name'] == 'Strong_Trend' and confidence > 0.4:
        print("→ 建议: 趋势交易策略")
    elif regime['name'] == 'Range' and confidence > 0.4:
        print("→ 建议: 区间交易策略")
    elif result['is_uncertain']:
        print("→ 建议: 市场状态不确定，谨慎操作")
    else:
        print("→ 建议: 根据具体状态调整策略")
    
    # 显示所有状态概率
    print("\n所有状态概率分布:")
    print("-" * 60)
    for name, prob in sorted(
        result['regime_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = "█" * int(prob * 50)
        print(f"{name:25s} {prob:6.2%} {bar}")

# 使用示例
if __name__ == "__main__":
    check_market_regime("BTCUSDT")
```

## 状态名称说明

系统定义了6种market regime状态：

1. **Strong_Trend** - 强趋势：高ADX，明显的趋势方向
2. **Weak_Trend** - 弱趋势：中等ADX，有一定趋势
3. **Range** - 区间震荡：低ADX，中等波动率
4. **Choppy_High_Vol** - 高波动无方向：低ADX，高波动率
5. **Volatility_Spike** - 波动率突增：极高波动率
6. **Squeeze** - 低波动蓄势：极低波动率，低ADX

注意：状态名称由HMM模型在训练时自动映射，不同训练可能略有差异。

## 注意事项

1. **时间框架一致性**: `timeframe` 参数必须与训练时的主时间框架一致（默认是 `15m`）。如果指定了不同的时间框架，系统会发出警告并使用训练时的主时间框架。

2. **模型必须先训练**: 使用API前，必须先训练对应交易对的模型。可以使用 `training_pipeline.py` 或 `examples.py` 进行训练。

3. **预测说明**: LSTM模型只能预测下一根K线的状态。模型使用过去64根K线的特征序列来预测下一根K线的状态。

4. **置信度阈值**: 如果最高概率低于配置的阈值（默认0.4），`is_uncertain` 会被设置为 `True`，表示预测不确定。

5. **错误处理**: 如果模型文件不存在或加载失败，API会抛出 `ValueError` 异常。建议使用 try-except 进行错误处理。

## 集成到其他项目

### 作为独立服务

可以将 `model_api.py` 复制到其他项目中，只需要确保：
- `config.py` 中的路径配置正确
- 模型文件存在于 `models/{SYMBOL}/` 目录下
- 安装了必要的依赖（见 `requirements.txt`）

### 作为库使用

```python
# 在其他项目中
import sys
sys.path.append('/path/to/regime_trainer')

from model_api import predict_regime

result = predict_regime("BTCUSDT", "15m")
```

### REST API 封装（可选）

如果需要提供HTTP接口，可以使用Flask或FastAPI封装：

```python
from flask import Flask, jsonify
from model_api import ModelAPI

app = Flask(__name__)
api = ModelAPI()

@app.route('/predict/<symbol>')
def predict(symbol):
    result = api.predict_next_regime(symbol, "15m")
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
```

## 常见问题

**Q: 如何知道哪些交易对有可用的模型？**
A: 使用 `api.list_available_models()` 方法。

**Q: 预测结果中的概率分布是什么意思？**
A: 每个概率表示该状态在下一根K线中出现的可能性。所有概率之和为1.0。

**Q: 为什么不能预测多根K线？**
A: LSTM模型训练时只学习预测下一根K线的状态。模型使用过去64根K线的特征序列来预测下一根K线的状态。如果需要预测多根K线，需要实现自回归预测或使用HMM转移矩阵，但预测精度会随着步数增加而下降。

**Q: 可以预测其他时间框架吗？**
A: 目前只支持训练时使用的主时间框架（默认15m）。要支持其他时间框架，需要重新训练模型。

**Q: 如何更新模型？**
A: 使用 `training_pipeline.py` 进行增量训练或完整重训。训练完成后，API会自动使用新的模型。

**Q: 如何获取历史regime数据用于回测？**
A: 使用 `get_regime_history()` 方法，支持按回看小时数或日期范围查询。例如：
```python
# 获取最近30天的历史regime
from datetime import datetime, timedelta
api = ModelAPI()
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
history = api.get_regime_history("BTCUSDT", start_date=start_date, end_date=end_date)
```

**Q: 历史regime数据从哪里获取？**
A: 优先从SQLite缓存数据库读取历史K线数据，如果缓存中没有数据，会从Binance API获取。历史regime是通过LSTM模型对历史K线数据进行滑动窗口预测得到的。
