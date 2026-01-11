# 历史 Market Regime API 使用指南

## 问题说明

在终端中直接输入 `GET /api/history/...` 会被 zsh 解释为命令，而不是 HTTP 请求。需要使用正确的工具来发送 HTTP 请求。

## 正确的使用方法

### 方法1: 使用 curl（推荐）

```bash
# 按回看小时数查询（24小时）
curl "http://localhost:5858/api/history/BTCUSDT?timeframe=15m&lookback_hours=24"

# 按日期范围查询（最近7天）
curl "http://localhost:5858/api/history/BTCUSDT?timeframe=15m&start_date=2024-01-01&end_date=2024-01-31"

# 格式化输出（使用 jq，如果已安装）
curl "http://localhost:5858/api/history/BTCUSDT?timeframe=15m&lookback_hours=24" | jq
```

### 方法2: 使用 Python 测试脚本

```bash
# 运行测试脚本（会自动测试两种查询方式）
python test_history_api.py
```

### 方法3: 使用 httpie（如果已安装）

```bash
# 安装 httpie: pip install httpie
http GET localhost:5858/api/history/BTCUSDT timeframe==15m lookback_hours==24
```

### 方法4: 在浏览器中访问

```
http://localhost:5858/api/history/BTCUSDT?timeframe=15m&lookback_hours=24
```

## API 参数说明

### 查询参数

- `timeframe` (必需): 时间框架，支持 `5m` 或 `15m`
- `lookback_hours` (可选): 回看小时数，从当前时间往前回看
- `start_date` (可选): 开始日期，ISO 8601 格式 (YYYY-MM-DD 或 YYYY-MM-DDTHH:MM:SS)
- `end_date` (可选): 结束日期，ISO 8601 格式

**注意**: 
- 如果指定了 `start_date` 和 `end_date`，则使用日期范围查询
- 如果只指定了 `lookback_hours`，则从当前时间往前回看
- 如果都不指定，默认回看 24 小时

### 限制

- 最大回看小时数: 720 小时（30天）
- 最大日期范围: 365 天（1年）

## 返回格式

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "lookback_hours": 24,
  "start_date": null,
  "end_date": null,
  "timestamp": "2024-01-01T12:00:00",
  "count": 96,
  "history": [
    {
      "timestamp": "2024-01-01T11:45:00",
      "regime_id": 0,
      "regime_name": "Range",
      "confidence": 0.85,
      "is_uncertain": false,
      "original_regime": "Range"
    },
    ...
  ]
}
```

## 启动服务器

```bash
# 启动 API 服务器
python run_server.py

# 服务器将在 http://0.0.0.0:5858 上运行
```

## 常见问题

### Q: 为什么直接输入 GET 命令不行？

A: 在终端中，`GET` 不是一个有效的命令。需要使用 HTTP 客户端工具（如 curl、httpie）或浏览器来发送 HTTP 请求。

### Q: 如何测试 API 是否正常工作？

A: 运行测试脚本：
```bash
python test_history_api.py
```

### Q: 支持多长时间范围的历史数据？

A: 
- 按回看小时数：最多 720 小时（30天）
- 按日期范围：最多 365 天（1年）

### Q: 历史数据从哪里获取？

A: 优先从 SQLite 缓存数据库读取，如果缓存中没有数据，会从 Binance API 获取。
