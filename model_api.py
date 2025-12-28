"""
模型 API 模块 - 为其他项目提供简单的预测接口

这个模块提供了简单的接口，让其他项目可以方便地：
1. 预测未来多根K线的market regime概率分布（t+1 到 t+4）
2. 获取历史regime序列（过去N根K线）
3. 获取模型元数据（状态映射、时间框架等）
4. 查询可用的交易对和模型信息

注意：模型支持多步预测（t+1 到 t+4），同时提供历史regime序列。
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from config import TrainingConfig, setup_logging
from realtime_predictor import RealtimeRegimePredictor, MultiTimeframeRegimePredictor

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelAPI:
    """
    模型 API - 提供简单的预测接口
    
    使用示例:
        api = ModelAPI()
        
        # 预测下一根15分钟K线的market regime
        result = api.predict_next_regime(
            symbol="BTCUSDT",
            timeframe="15m"
        )
        
        # 获取模型元数据
        metadata = api.get_model_metadata("BTCUSDT")
    """
    
    def __init__(self, config: TrainingConfig = None):
        """
        初始化 API
        
        Args:
            config: 训练配置，如果为None则使用默认配置
        """
        self.config = config or TrainingConfig
        self._predictors = {}  # 缓存预测器，避免重复加载模型 {(symbol, timeframe): predictor}
        self._multi_tf_predictors = {}  # 缓存多时间框架预测器 {symbol: predictor}
    
    def _get_predictor(self, symbol: str, primary_timeframe: str = None) -> RealtimeRegimePredictor:
        """
        获取或创建预测器（带缓存）
        
        Args:
            symbol: 交易对
            primary_timeframe: 主时间框架（如 "5m" 或 "15m"），如果为 None 则使用默认配置
            
        Returns:
            预测器实例
        """
        if primary_timeframe is None:
            primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        cache_key = (symbol, primary_timeframe)
        
        if cache_key not in self._predictors:
            try:
                self._predictors[cache_key] = RealtimeRegimePredictor(
                    symbol, self.config, primary_timeframe
                )
            except FileNotFoundError as e:
                logger.error(f"无法加载 {symbol} ({primary_timeframe}) 的模型: {e}")
                raise ValueError(f"模型文件不存在，请先训练 {symbol} 的 {primary_timeframe} 模型")
        
        return self._predictors[cache_key]
    
    def _get_multi_tf_predictor(self, symbol: str, timeframes: list = None) -> MultiTimeframeRegimePredictor:
        """
        获取或创建多时间框架预测器（带缓存）
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表
            
        Returns:
            多时间框架预测器实例
        """
        if timeframes is None:
            timeframes = self.config.ENABLED_MODELS
        
        cache_key = symbol
        
        if cache_key not in self._multi_tf_predictors:
            self._multi_tf_predictors[cache_key] = MultiTimeframeRegimePredictor(
                symbol, self.config, timeframes
            )
        
        return self._multi_tf_predictors[cache_key]
    
    def predict_regimes(
        self,
        symbol: str,
        primary_timeframe: str = None,
        include_history: bool = True,
        history_bars: int = 16
    ) -> Dict:
        """
        预测未来多根 K 线的 market regime 概率分布（支持 t+1 到 t+4）
        
        这是推荐的新 API 方法，同时返回：
        - 历史 regime 序列（过去 N 根 K 线的状态）
        - 未来 4 步预测（t+1 到 t+4 的概率分布）
        
        Args:
            symbol: 交易对（如 "BTCUSDT"）
            primary_timeframe: 主时间框架（如 "5m", "15m"），如果为 None 则使用默认配置
            include_history: 是否包含历史 regime 序列
            history_bars: 历史序列的 K 线数量（默认 16 根，对于 15m 约为 4 小时）
            
        Returns:
            包含预测结果的字典:
            {
                'symbol': str,
                'timeframe': str,
                'timestamp': datetime,
                'historical_regimes': {  # 历史 regime 序列
                    'sequence': ['Range', 'Range', 'Weak_Trend', ...],
                    'timestamps': [...],
                    'confidences': [...],
                    'count': 16,
                    'lookback_hours': 4.0
                },
                'predictions': {  # 多步预测
                    't+1': {
                        'probabilities': {...},
                        'most_likely': str,
                        'confidence': float,
                        'is_uncertain': bool
                    },
                    't+2': {...},
                    't+3': {...},
                    't+4': {...}
                },
                'is_multistep': bool,  # 是否多步模型
                'model_info': {...}
            }
        """
        if primary_timeframe is None:
            primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        # 获取预测器
        predictor = self._get_predictor(symbol, primary_timeframe)
        
        # 获取完整预测结果（包括多步预测和历史序列）
        current_regime = predictor.get_current_regime()
        
        # 获取模型元数据
        model_info = self._get_model_info(predictor)
        model_info['sequence_length'] = predictor.lstm_classifier.sequence_length
        model_info['is_multistep'] = True  # 现在总是多步预测
        model_info['prediction_horizons'] = predictor.lstm_classifier.prediction_horizons
        
        # 构建预测结果
        predictions = {}
        for horizon, pred in current_regime.get('predictions', {}).items():
            predictions[horizon] = {
                'probabilities': pred['probabilities'],
                'most_likely': pred['regime_name'],
                'regime_id': pred['regime_id'],
                'confidence': pred['confidence'],
                'is_uncertain': pred['is_uncertain']
            }
        
        # 构建结果
        result = {
            'symbol': symbol,
            'timeframe': primary_timeframe,
            'timestamp': datetime.now(),
            'predictions': predictions,
            'is_multistep': True,  # 现在总是多步预测
            'model_info': model_info
        }
        
        # 添加历史 regime 序列
        if include_history:
            result['historical_regimes'] = current_regime.get('historical_regimes', {})
        
        # 日志输出
        if predictions.get('t+1'):
            t1 = predictions['t+1']
            logger.info(
                f"{symbol} 多步预测: t+1={t1['most_likely']} ({t1['confidence']:.2%})"
            )
            for h in ['t+2', 't+3', 't+4']:
                if h in predictions:
                    p = predictions[h]
                    logger.debug(f"  {h}: {p['most_likely']} ({p['confidence']:.2%})")
        
        return result
    
    def predict_next_regime(
        self,
        symbol: str,
        timeframe: str = None,
        primary_timeframe: str = None
    ) -> Dict:
        """
        预测下一根K线的market regime概率分布（向后兼容的接口）
        
        注意：此方法保留用于向后兼容。推荐使用 predict_regimes() 方法以获取多步预测。
        
        对于多步模型，此方法只返回 t+1 的预测结果。
        
        Args:
            symbol: 交易对（如 "BTCUSDT"）
            timeframe: [已废弃] 使用 primary_timeframe 代替
            primary_timeframe: 主时间框架（如 "5m", "15m"），如果为 None 则使用默认配置
            
        Returns:
            包含预测结果的字典（只包含 t+1 预测，向后兼容格式）
        """
        # 处理 timeframe 参数（向后兼容）
        if primary_timeframe is None:
            if timeframe is not None and timeframe in self.config.MODEL_CONFIGS:
                primary_timeframe = timeframe
            else:
                primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        # 获取预测器
        predictor = self._get_predictor(symbol, primary_timeframe)
        timeframe = primary_timeframe  # 用于返回结果
        
        # 获取当前市场状态预测
        current_regime = predictor.get_current_regime()
        
        # 提取概率分布（t+1）
        regime_probs = current_regime['probabilities']
        
        # 找到最可能的状态
        most_likely_id = current_regime['regime_id']
        most_likely_name = current_regime['regime_name']
        most_likely_prob = current_regime['confidence']
        
        # 获取模型元数据
        model_info = self._get_model_info(predictor)
        model_info['sequence_length'] = predictor.lstm_classifier.sequence_length
        
        # 构建结果（向后兼容格式）
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            'regime_probabilities': regime_probs,
            'most_likely_regime': {
                'id': int(most_likely_id),
                'name': most_likely_name,
                'probability': float(most_likely_prob)
            },
            'confidence': float(current_regime['confidence']),
            'is_uncertain': current_regime.get('is_uncertain', False),
            'model_info': model_info
        }
        
        logger.info(
            f"{symbol} 下一根{timeframe}K线预测: "
            f"{most_likely_name} (概率: {most_likely_prob:.2%})"
        )
        
        return result
    
    # 保持向后兼容（已废弃，建议使用 predict_next_regime）
    def predict_future_regimes(
        self,
        symbol: str,
        timeframe: str = "15m",
        n_bars: int = 1
    ) -> Dict:
        """
        [已废弃] 预测未来N根K线的market regime概率分布
        
        注意：此方法已废弃。请使用 predict_regimes() 方法以获取多步预测（t+1 到 t+4）。
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            n_bars: 已废弃，将被忽略
            
        Returns:
            预测结果（只返回 t+1 预测）
        """
        if n_bars != 1:
            logger.warning(
                f"predict_future_regimes() 已废弃。"
                f"请使用 predict_regimes() 方法以获取多步预测（t+1 到 t+4）。"
            )
        
        return self.predict_next_regime(symbol, timeframe)
    
    def get_model_metadata(self, symbol: str, primary_timeframe: str = None) -> Dict:
        """
        获取模型元数据
        
        Args:
            symbol: 交易对
            primary_timeframe: 主时间框架（如 "5m" 或 "15m"），如果为 None 则使用默认配置
            
        Returns:
            模型元数据字典:
            {
                'symbol': str,
                'primary_timeframe': str,
                'n_states': int,
                'regime_mapping': dict,  # {state_id: regime_name}
                'regime_names': list,  # 所有状态名称列表
                'model_paths': {
                    'lstm': str,
                    'hmm': str,
                    'scaler': str
                },
                'training_info': {
                    'sequence_length': int,
                    'feature_count': int
                }
            }
        """
        if primary_timeframe is None:
            primary_timeframe = self.config.PRIMARY_TIMEFRAME
        
        predictor = self._get_predictor(symbol, primary_timeframe)
        model_info = self._get_model_info(predictor)
        
        # 获取模型路径
        model_paths = {
            'lstm': self.config.get_model_path(symbol, 'lstm', primary_timeframe),
            'hmm': self.config.get_hmm_path(symbol, primary_timeframe),
            'scaler': self.config.get_scaler_path(symbol, primary_timeframe)
        }
        
        # 获取训练信息
        training_info = {
            'sequence_length': predictor.lstm_classifier.sequence_length,
            'feature_count': len(predictor.lstm_classifier.feature_names_) 
                if predictor.lstm_classifier.feature_names_ else None
        }
        
        result = {
            'symbol': symbol,
            'primary_timeframe': model_info['primary_timeframe'],
            'n_states': model_info['n_states'],
            'regime_mapping': model_info['regime_mapping'],
            'regime_names': list(model_info['regime_mapping'].values()),
            'model_paths': model_paths,
            'training_info': training_info
        }
        
        return result
    
    def _get_model_info(self, predictor: RealtimeRegimePredictor) -> Dict:
        """
        从预测器提取模型信息
        
        Args:
            predictor: 预测器实例
            
        Returns:
            模型信息字典
        """
        regime_mapping = predictor.regime_mapping or {}
        
        # 如果没有映射，使用默认状态名称
        if not regime_mapping:
            n_states = predictor.lstm_classifier.n_states
            regime_mapping = {i: f"State_{i}" for i in range(n_states)}
        
        return {
            'primary_timeframe': predictor.primary_timeframe,
            'n_states': predictor.lstm_classifier.n_states,
            'regime_mapping': regime_mapping
        }
    
    def predict_multi_timeframe_regimes(
        self,
        symbol: str,
        timeframes: List[str] = None,
        include_history: bool = True
    ) -> Dict:
        """
        同时预测多个时间框架的 market regime（多步预测）
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表（如 ["5m", "15m"]），如果为 None 则使用 ENABLED_MODELS
            include_history: 是否包含历史regime序列
            
        Returns:
            包含多个时间框架预测结果的字典:
            {
                'symbol': str,
                'timestamp': datetime,
                'regimes': {
                    '5m': {...多步预测结果...},
                    '15m': {...多步预测结果...}
                }
            }
        """
        if timeframes is None:
            timeframes = self.config.ENABLED_MODELS
        
        results = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'regimes': {}
        }
        
        for tf in timeframes:
            try:
                result = self.predict_regimes(
                    symbol=symbol,
                    primary_timeframe=tf,
                    include_history=include_history
                )
                results['regimes'][tf] = result
            except Exception as e:
                logger.error(f"预测 {symbol} 的 {tf} regime 失败: {e}")
                results['regimes'][tf] = {'error': str(e)}
        
        return results
    
    def predict_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = None
    ) -> Dict:
        """
        [已废弃] 同时预测多个时间框架的 market regime
        
        注意：此方法已废弃，请使用 predict_multi_timeframe_regimes() 以获取多步预测。
        
        Args:
            symbol: 交易对
            timeframes: 时间框架列表（如 ["5m", "15m"]），如果为 None 则使用 ENABLED_MODELS
            
        Returns:
            包含多个时间框架预测结果的字典（只包含 t+1 预测）
        """
        return self.predict_multi_timeframe_regimes(symbol, timeframes, include_history=False)
    
    def list_available_models(self, primary_timeframe: str = None) -> List[str]:
        """
        列出所有可用的模型（已训练的交易对）
        
        Args:
            primary_timeframe: 主时间框架，如果为 None 则检查所有启用的时间框架
            
        Returns:
            交易对列表
        """
        available = []
        
        if primary_timeframe:
            timeframes_to_check = [primary_timeframe]
        else:
            timeframes_to_check = self.config.ENABLED_MODELS
        
        for symbol in self.config.SYMBOLS:
            for tf in timeframes_to_check:
                model_path = self.config.get_model_path(symbol, 'lstm', tf)
                scaler_path = self.config.get_scaler_path(symbol, tf)
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    available.append(symbol)
                    break  # 只要有一个时间框架的模型存在就认为可用
        
        return available
    
    def list_available_models_by_timeframe(self) -> Dict[str, List[str]]:
        """
        列出每个时间框架可用的模型
        
        Returns:
            {timeframe: [symbol, ...]} 格式的字典
        """
        result = {}
        
        for tf in self.config.MODEL_CONFIGS.keys():
            result[tf] = []
            for symbol in self.config.SYMBOLS:
                model_path = self.config.get_model_path(symbol, 'lstm', tf)
                scaler_path = self.config.get_scaler_path(symbol, tf)
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    result[tf].append(symbol)
        
        return result
    
    def batch_predict(
        self,
        symbols: List[str],
        primary_timeframe: str = None
    ) -> Dict[str, Dict]:
        """
        批量预测多个交易对的下一根K线
        
        Args:
            symbols: 交易对列表
            primary_timeframe: 主时间框架
            
        Returns:
            {symbol: prediction_result} 字典
        """
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.predict_next_regime(
                    symbol=symbol,
                    primary_timeframe=primary_timeframe
                )
            except Exception as e:
                logger.error(f"预测 {symbol} 失败: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def get_regime_probability(
        self,
        symbol: str,
        regime_name: str,
        primary_timeframe: str = None
    ) -> float:
        """
        获取下一根K线特定状态的概率（便捷方法）
        
        Args:
            symbol: 交易对
            regime_name: 状态名称（如 "Strong_Trend"）
            primary_timeframe: 主时间框架
            
        Returns:
            该状态的概率（0.0-1.0）
        """
        result = self.predict_next_regime(
            symbol=symbol,
            primary_timeframe=primary_timeframe
        )
        
        regime_probs = result['regime_probabilities']
        
        # 尝试直接匹配
        if regime_name in regime_probs:
            return regime_probs[regime_name]
        
        # 尝试不区分大小写匹配
        for name, prob in regime_probs.items():
            if name.lower() == regime_name.lower():
                return prob
        
        # 如果找不到，返回0.0
        logger.warning(f"未找到状态 '{regime_name}'，可用状态: {list(regime_probs.keys())}")
        return 0.0


# ==================== 便捷函数 ====================

def predict_regime(
    symbol: str,
    primary_timeframe: str = None,
    config: TrainingConfig = None
) -> Dict:
    """
    便捷函数：预测下一根K线的market regime
    
    Args:
        symbol: 交易对
        primary_timeframe: 主时间框架（如 "5m", "15m"）
        config: 配置（可选）
        
    Returns:
        预测结果字典
        
    示例:
        result = predict_regime("BTCUSDT", "15m")
        print(result['most_likely_regime']['name'])
        print(result['regime_probabilities'])
    """
    api = ModelAPI(config)
    return api.predict_next_regime(symbol, primary_timeframe=primary_timeframe)


def predict_multi_timeframe(
    symbol: str,
    timeframes: List[str] = None,
    config: TrainingConfig = None
) -> Dict:
    """
    便捷函数：同时预测多个时间框架的market regime
    
    Args:
        symbol: 交易对
        timeframes: 时间框架列表（如 ["5m", "15m"]）
        config: 配置（可选）
        
    Returns:
        多时间框架预测结果
        
    示例:
        result = predict_multi_timeframe("BTCUSDT", ["5m", "15m"])
        print(result['regimes']['5m']['most_likely_regime']['name'])
        print(result['regimes']['15m']['most_likely_regime']['name'])
    """
    api = ModelAPI(config)
    return api.predict_multi_timeframe(symbol, timeframes)


def get_regime_probability(
    symbol: str,
    regime_name: str,
    primary_timeframe: str = None,
    config: TrainingConfig = None
) -> float:
    """
    便捷函数：获取下一根K线特定状态的概率
    
    Args:
        symbol: 交易对
        regime_name: 状态名称
        primary_timeframe: 主时间框架
        config: 配置（可选）
        
    Returns:
        该状态的概率（0.0-1.0）
        
    示例:
        prob = get_regime_probability("BTCUSDT", "Strong_Trend")
        print(f"Strong_Trend 概率: {prob:.2%}")
    """
    api = ModelAPI(config)
    return api.get_regime_probability(symbol, regime_name, primary_timeframe)


# ==================== 主函数（示例） ====================

def main():
    """示例用法"""
    api = ModelAPI()
    
    # 列出可用的模型
    available = api.list_available_models()
    print(f"\n可用的模型: {available}")
    
    if not available:
        print("\n⚠️  没有可用的模型，请先训练模型")
        return
    
    # 使用第一个可用的交易对
    symbol = available[0]
    
    # 预测下一根15分钟K线的market regime
    print(f"\n预测 {symbol} 下一根15分钟K线的market regime:")
    print("=" * 70)
    
    result = api.predict_next_regime(symbol, "15m")
    
    print(f"交易对: {result['symbol']}")
    print(f"时间框架: {result['timeframe']}")
    print(f"预测时间: {result['timestamp']}")
    print(f"使用历史K线数: {result['model_info']['sequence_length']}")
    print(f"\n最可能的状态: {result['most_likely_regime']['name']}")
    print(f"概率: {result['most_likely_regime']['probability']:.2%}")
    print(f"置信度: {result['confidence']:.2%}")
    
    print(f"\n所有状态概率分布:")
    print("-" * 70)
    for regime_name, prob in sorted(
        result['regime_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = "█" * int(prob * 50)
        print(f"{regime_name:25s} {prob:6.2%} {bar}")
    
    # 获取模型元数据
    print(f"\n模型元数据:")
    print("=" * 70)
    metadata = api.get_model_metadata(symbol)
    print(f"状态数量: {metadata['n_states']}")
    print(f"状态映射: {metadata['regime_mapping']}")
    print(f"主时间框架: {metadata['primary_timeframe']}")
    
    # 使用便捷函数
    print(f"\n使用便捷函数:")
    print("=" * 70)
    prob = get_regime_probability(symbol, "Strong_Trend")
    print(f"Strong_Trend 概率: {prob:.2%}")


if __name__ == "__main__":
    main()

