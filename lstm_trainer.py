"""
LSTM 训练模块 - 用于市场状态预测

修复数据泄漏问题：
- 支持 train/val/test 三分（而不是 train/test 二分）
- Scaler 只在训练集上 fit
- 验证集用于早停和模型选择
- 测试集只用于最终评估

多步预测：
- 支持 t+1 到 t+4 的多步预测
- t+1 使用硬标签（sparse_categorical_crossentropy）
- t+2-t+4 使用软标签（KL divergence）
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# 使用 keras.layers 和 keras.callbacks 而不是直接导入，避免 linter 警告
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from typing import Tuple, Dict, Optional, List
import os

logger = logging.getLogger(__name__)

class LSTMRegimeClassifier:
    """LSTM 市场状态分类器（支持多步预测）"""
    
    def __init__(
        self,
        n_states: int = 6,
        sequence_length: int = 64,
        lstm_units: list = [128, 64],
        dense_units: list = [64],
        dropout_rate: float = 0.2,
        l2_lambda: float = 1e-4,
        use_batch_norm: bool = True,
        learning_rate: float = 1e-3,
        prediction_horizons: List[int] = None,
        horizon_loss_weights: Dict[str, float] = None
    ):
        """
        初始化
        
        Args:
            n_states: 状态数量
            sequence_length: 输入序列长度
            lstm_units: LSTM 层单元数列表
            dense_units: 全连接层单元数列表
            dropout_rate: Dropout 比率
            l2_lambda: L2 正则化强度
            use_batch_norm: 是否使用 BatchNormalization
            learning_rate: Adam 优化器学习率
            prediction_horizons: 预测步数列表，默认 [1, 2, 3, 4]
            horizon_loss_weights: 各步损失权重，默认 {t+1: 1.0, t+2: 0.8, ...}
        """
        self.n_states = n_states
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.history = None
        self.feature_names_ = None  # 保存训练时使用的特征名称
        
        # 多步预测配置
        self.prediction_horizons = prediction_horizons or [1, 2, 3, 4]
        self.horizon_loss_weights = horizon_loss_weights or {
            't+1': 1.0,
            't+2': 0.8,
            't+3': 0.6,
            't+4': 0.4,
        }
        # 现在总是使用多步预测
        self.is_multistep = True
    
    def build_model(self, n_features: int):
        """
        构建多步 LSTM 模型（t+1 到 t+4）
        
        Args:
            n_features: 输入特征数量
        """
        self._build_multistep_model(n_features)
    
    def _build_multistep_model(self, n_features: int):
        """
        构建多步 LSTM 模型（4 个输出头：t+1 到 t+4）
        
        Args:
            n_features: 输入特征数量
        """
        # L2 正则化器
        l2_reg = keras.regularizers.l2(self.l2_lambda) if self.l2_lambda > 0 else None
        
        # 输入层
        inputs = keras.Input(shape=(self.sequence_length, n_features), name='input')
        
        # 共享 LSTM 编码器
        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg,
                name=f'lstm_{i}'
            )(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(name=f'bn_lstm_{i}')(x)
            x = keras.layers.Dropout(self.dropout_rate, name=f'dropout_lstm_{i}')(x)
        
        # 共享 Dense 层
        shared = x
        for i, units in enumerate(self.dense_units):
            shared = keras.layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=l2_reg,
                name=f'shared_dense_{i}'
            )(shared)
            if self.use_batch_norm:
                shared = keras.layers.BatchNormalization(name=f'bn_dense_{i}')(shared)
            shared = keras.layers.Dropout(self.dropout_rate, name=f'dropout_dense_{i}')(shared)
        
        # 4 个独立的输出头
        outputs = []
        for h in self.prediction_horizons:
            # 每个 horizon 有自己的小型 Dense 网络
            head = keras.layers.Dense(
                32, 
                activation='relu',
                kernel_regularizer=l2_reg,
                name=f'head_dense_t{h}'
            )(shared)
            head = keras.layers.Dropout(self.dropout_rate, name=f'head_dropout_t{h}')(head)
            output = keras.layers.Dense(
                self.n_states,
                activation='softmax',
                name=f'output_t{h}'
            )(head)
            outputs.append(output)
        
        # 创建模型
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='multistep_lstm')
        
        # 编译模型（每个输出头有自己的损失函数和权重）
        losses = {}
        loss_weights = {}
        metrics = {}
        
        for i, h in enumerate(self.prediction_horizons):
            output_name = f'output_t{h}'
            if h == 1:
                # t+1 使用硬标签的 sparse categorical crossentropy
                losses[output_name] = 'sparse_categorical_crossentropy'
            else:
                # t+2-t+4 使用软标签的 KL divergence
                losses[output_name] = 'kl_divergence'
            
            loss_weights[output_name] = self.horizon_loss_weights.get(f't+{h}', 1.0)
            metrics[output_name] = ['accuracy']
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        logger.info(f"LSTM 多步模型已构建 (horizons={self.prediction_horizons})")
        logger.info(f"损失权重: {loss_weights}")
        self.model.summary(print_fn=logger.info)
    
    def _create_sequences(
        self, 
        features_scaled: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据的辅助方法
        
        Args:
            features_scaled: 标准化后的特征数组
            labels: 标签数组
            
        Returns:
            (X, y) - 序列特征和对应标签
        """
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i+self.sequence_length])
            y.append(labels[i+self.sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_data_split(
        self,
        train_features: pd.DataFrame,
        train_labels: np.ndarray,
        val_features: pd.DataFrame,
        val_labels: np.ndarray,
        test_features: Optional[pd.DataFrame] = None,
        test_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
               Optional[np.ndarray], Optional[np.ndarray]]:
        """
        准备训练数据（推荐方法：支持 train/val/test 三分）
        
        此方法接收已经划分好的数据，确保：
        - Scaler 只在训练集上 fit
        - 验证集和测试集只做 transform
        - 避免任何形式的数据泄漏
        
        Args:
            train_features: 训练集特征
            train_labels: 训练集标签
            val_features: 验证集特征
            val_labels: 验证集标签
            test_features: 测试集特征（可选）
            test_labels: 测试集标签（可选）
            
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
            如果没有测试集，X_test 和 y_test 为 None
        """
        # 保存特征名称
        self.feature_names_ = list(train_features.columns)
        
        logger.info(f"数据划分: 训练集 {len(train_features)} 行, "
                   f"验证集 {len(val_features)} 行, "
                   f"测试集 {len(test_features) if test_features is not None else 0} 行")
        
        # 只在训练集上 fit scaler（避免数据泄露）
        self.scaler = StandardScaler()
        train_scaled = self.scaler.fit_transform(train_features)
        
        # 用训练集的 scaler 参数 transform 验证集
        val_scaled = self.scaler.transform(val_features)
        
        # 创建训练和验证的序列数据
        X_train, y_train = self._create_sequences(train_scaled, train_labels)
        X_val, y_val = self._create_sequences(val_scaled, val_labels)
        
        logger.info(f"序列数据形状: X_train={X_train.shape}, X_val={X_val.shape}")
        
        # 处理测试集（如果有）
        X_test, y_test = None, None
        if test_features is not None and test_labels is not None:
            test_scaled = self.scaler.transform(test_features)
            X_test, y_test = self._create_sequences(test_scaled, test_labels)
            logger.info(f"测试集序列形状: X_test={X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _create_multistep_sequences(
        self,
        features_scaled: np.ndarray,
        labels_dict: Dict[str, np.ndarray],
        max_horizon: int = 4
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        创建多步预测的序列数据
        
        Args:
            features_scaled: 标准化后的特征数组
            labels_dict: 多步标签字典 {'hard_t+1': ..., 'soft_t+2': ..., ...}
            max_horizon: 最大预测步数
            
        Returns:
            (X, y_dict) - 序列特征和对应的多步标签字典
        """
        n_samples = len(features_scaled) - self.sequence_length - max_horizon
        
        if n_samples <= 0:
            raise ValueError(
                f"数据量不足：需要至少 {self.sequence_length + max_horizon + 1} 个样本，"
                f"当前只有 {len(features_scaled)} 个"
            )
        
        X = []
        y_dict = {f't+{h}': [] for h in self.prediction_horizons}
        
        for i in range(n_samples):
            # 输入序列
            X.append(features_scaled[i:i + self.sequence_length])
            
            # 各步的标签
            for h in self.prediction_horizons:
                target_idx = i + self.sequence_length - 1 + h
                if h == 1:
                    # t+1 使用硬标签
                    y_dict[f't+{h}'].append(labels_dict[f'hard_t+{h}'][target_idx])
                else:
                    # t+2-t+4 使用软标签
                    y_dict[f't+{h}'].append(labels_dict[f'soft_t+{h}'][target_idx])
        
        X = np.array(X)
        for key in y_dict:
            y_dict[key] = np.array(y_dict[key])
        
        return X, y_dict
    
    def prepare_multistep_data_split(
        self,
        train_features: pd.DataFrame,
        train_labels: Dict[str, np.ndarray],
        val_features: pd.DataFrame,
        val_labels: Dict[str, np.ndarray],
        test_features: Optional[pd.DataFrame] = None,
        test_labels: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], 
               np.ndarray, Dict[str, np.ndarray],
               Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        准备多步预测的训练数据
        
        Args:
            train_features: 训练集特征
            train_labels: 训练集多步标签字典
            val_features: 验证集特征
            val_labels: 验证集多步标签字典
            test_features: 测试集特征（可选）
            test_labels: 测试集多步标签字典（可选）
            
        Returns:
            (X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict)
        """
        # 保存特征名称
        self.feature_names_ = list(train_features.columns)
        
        max_horizon = max(self.prediction_horizons)
        
        logger.info(f"多步数据划分: 训练集 {len(train_features)} 行, "
                   f"验证集 {len(val_features)} 行, "
                   f"测试集 {len(test_features) if test_features is not None else 0} 行")
        logger.info(f"预测步数: {self.prediction_horizons}, 最大 horizon: {max_horizon}")
        
        # 只在训练集上 fit scaler（避免数据泄露）
        self.scaler = StandardScaler()
        train_scaled = self.scaler.fit_transform(train_features)
        
        # 用训练集的 scaler 参数 transform 验证集
        val_scaled = self.scaler.transform(val_features)
        
        # 创建训练和验证的多步序列数据
        X_train, y_train_dict = self._create_multistep_sequences(
            train_scaled, train_labels, max_horizon
        )
        X_val, y_val_dict = self._create_multistep_sequences(
            val_scaled, val_labels, max_horizon
        )
        
        logger.info(f"多步序列数据形状: X_train={X_train.shape}")
        for key in y_train_dict:
            logger.info(f"  {key}: train={y_train_dict[key].shape}, val={y_val_dict[key].shape}")
        
        # 处理测试集（如果有）
        X_test, y_test_dict = None, None
        if test_features is not None and test_labels is not None:
            test_scaled = self.scaler.transform(test_features)
            X_test, y_test_dict = self._create_multistep_sequences(
                test_scaled, test_labels, max_horizon
            )
            logger.info(f"测试集多步序列形状: X_test={X_test.shape}")
        
        return X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping: bool = True,
        early_stopping_patience: int = 8,
        model_path: str = None,
        use_class_weight: bool = True,
        lr_reduce_patience: int = 5
    ) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            early_stopping: 是否使用早停
            early_stopping_patience: 早停耐心值（验证损失不改善的epoch数）
            model_path: 模型保存路径
            use_class_weight: 是否使用类权重（处理类别不平衡）
            lr_reduce_patience: 学习率衰减耐心值
            
        Returns:
            训练历史
        """
        if self.model is None:
            self.build_model(X_train.shape[2])
        
        # ============ 计算类权重（处理类别不平衡） ============
        class_weight_dict = None
        if use_class_weight:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train
            )
            class_weight_dict = dict(zip(classes, class_weights))
            
            # 输出类别分布和权重信息
            class_counts = np.bincount(y_train.astype(int))
            logger.info(f"训练集类别分布: {class_counts}")
            logger.info(f"类权重: {class_weight_dict}")
        
        # 回调函数
        callback_list = []
        
        # 早停
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            )
            callback_list.append(early_stop)
        
        # 学习率衰减
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=lr_reduce_patience,
            min_lr=1e-6
        )
        callback_list.append(lr_scheduler)
        
        # 模型检查点
        if model_path:
            checkpoint = keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            callback_list.append(checkpoint)
        
        # 训练（添加类权重）
        logger.info("开始训练 LSTM 模型...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        return self.history.history
    
    def train_multistep(
        self,
        X_train: np.ndarray,
        y_train_dict: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val_dict: Dict[str, np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping: bool = True,
        early_stopping_patience: int = 8,
        model_path: str = None,
        use_class_weight: bool = True,
        lr_reduce_patience: int = 5
    ) -> Dict:
        """
        训练多步预测模型
        
        Args:
            X_train: 训练特征
            y_train_dict: 训练标签字典 {'t+1': ..., 't+2': ..., ...}
            X_val: 验证特征
            y_val_dict: 验证标签字典
            epochs: 训练轮数
            batch_size: 批次大小
            early_stopping: 是否使用早停
            early_stopping_patience: 早停耐心值
            model_path: 模型保存路径
            use_class_weight: 是否使用类权重（仅对 t+1 硬标签）
            lr_reduce_patience: 学习率衰减耐心值
            
        Returns:
            训练历史
        """
        if self.model is None:
            self.build_model(X_train.shape[2])
        
        # 构建训练数据格式（Keras 多输出需要列表或字典）
        y_train_list = []
        y_val_list = []
        for h in self.prediction_horizons:
            y_train_list.append(y_train_dict[f't+{h}'])
            y_val_list.append(y_val_dict[f't+{h}'])
        
        # 计算类权重（仅对 t+1 硬标签）
        class_weight_dict = None
        if use_class_weight:
            from sklearn.utils.class_weight import compute_class_weight
            y_t1 = y_train_dict['t+1']
            classes = np.unique(y_t1)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_t1
            )
            class_weight_dict = dict(zip(classes, class_weights))
            
            # 输出类别分布和权重信息
            class_counts = np.bincount(y_t1.astype(int))
            logger.info(f"t+1 训练集类别分布: {class_counts}")
            logger.info(f"t+1 类权重: {class_weight_dict}")
        
        # 回调函数
        callback_list = []
        
        # 早停（监控总损失）
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            )
            callback_list.append(early_stop)
        
        # 学习率衰减
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=lr_reduce_patience,
            min_lr=1e-6
        )
        callback_list.append(lr_scheduler)
        
        # 模型检查点
        if model_path:
            checkpoint = keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callback_list.append(checkpoint)
        
        # 训练
        logger.info("开始训练多步 LSTM 模型...")
        logger.info(f"  预测步数: {self.prediction_horizons}")
        logger.info(f"  训练样本数: {len(X_train)}")
        
        # 注意：多输出模型不支持 class_weight，需要使用 sample_weight
        # 这里我们暂时不使用 class_weight，后续可以通过自定义训练循环实现
        self.history = self.model.fit(
            X_train, y_train_list,
            validation_data=(X_val, y_val_list),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        return self.history.history
    
    def incremental_train(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        validation_split: float = 0.2,
        early_stopping_patience: int = 3,
        use_class_weight: bool = True
    ):
        """
        增量训练（在现有模型基础上继续训练，带验证集和早停）
        
        对于多步模型，会自动将单标签转换为多步标签格式。
        
        Args:
            X_new: 新的训练数据
            y_new: 新的标签（单步标签，会自动扩展为多步）
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 增量训练学习率（比完整训练小）
            validation_split: 验证集比例
            early_stopping_patience: 早停耐心值
            use_class_weight: 是否使用类权重
        """
        if self.model is None:
            raise ValueError("模型尚未初始化，请先进行完整训练")
        
        logger.info("开始增量训练...")
        logger.info(f"  数据量: {len(X_new)} 样本")
        logger.info(f"  验证集比例: {validation_split:.0%}")
        logger.info(f"  学习率: {learning_rate}")
        
        # ============ 计算类权重 ============
        class_weight_dict = None
        if use_class_weight:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_new)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_new
            )
            class_weight_dict = dict(zip(classes, class_weights))
            
            class_counts = np.bincount(y_new.astype(int))
            logger.info(f"  类别分布: {class_counts}")
            logger.info(f"  类权重: {class_weight_dict}")
        
        # ============ 多步模型：重新编译并准备多输出标签 ============
        losses = {}
        loss_weights = {}
        metrics = {}
        
        for h in self.prediction_horizons:
            output_name = f'output_t{h}'
            if h == 1:
                losses[output_name] = 'sparse_categorical_crossentropy'
            else:
                # 对于 t+2-t+4，增量训练时也使用硬标签
                # （理想情况下应该用 HMM 生成软标签，但增量训练通常数据量小）
                losses[output_name] = 'sparse_categorical_crossentropy'
            
            loss_weights[output_name] = self.horizon_loss_weights.get(f't+{h}', 1.0)
            metrics[output_name] = ['accuracy']
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        # 将单标签扩展为多输出格式
        # 对于增量训练，我们简单地复制标签给所有 horizon
        # （因为增量训练数据量小，无法可靠生成软标签）
        y_list = [y_new for _ in self.prediction_horizons]
        logger.info(f"  多步标签: {len(y_list)} 个输出头，每个 {len(y_new)} 样本")
        
        # ============ 回调函数 ============
        callback_list = []
        
        # 早停（比完整训练更敏感）
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        
        # 学习率衰减（更敏感的参数）
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(lr_scheduler)
        
        # 训练（带验证集、早停）
        # 注意：多输出模型不支持 class_weight
        history = self.model.fit(
            X_new, y_list,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        # 输出训练结果
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        # 多输出模型的 accuracy 键名是 output_t1_accuracy 等
        acc_key = 'output_t1_accuracy'
        val_acc_key = 'val_output_t1_accuracy'
        final_acc = history.history.get(acc_key, [0])[-1]
        final_val_acc = history.history.get(val_acc_key, [0])[-1]
        
        logger.info(f"增量训练完成:")
        logger.info(f"  训练损失: {final_loss:.4f}, 验证损失: {final_val_loss:.4f}")
        logger.info(f"  训练准确率: {final_acc:.4f}, 验证准确率: {final_val_acc:.4f}")
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测市场状态（返回 t+1 的预测结果）
        
        Args:
            X: 输入序列
            
        Returns:
            预测的状态标签
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        predictions = self.model.predict(X, verbose=0)
        # 多步模型返回列表，取 t+1（第一个输出）
        return np.argmax(predictions[0], axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测市场状态的概率分布（返回 t+1 的概率分布）
        
        Args:
            X: 输入序列
            
        Returns:
            状态概率矩阵
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        predictions = self.model.predict(X, verbose=0)
        # 多步模型返回列表，取 t+1（第一个输出）
        return predictions[0]
    
    def predict_multistep(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        预测多步市场状态（t+1 到 t+4）
        
        Args:
            X: 输入序列
            
        Returns:
            字典 {'t+1': proba, 't+2': proba, ...}
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        predictions = self.model.predict(X, verbose=0)
        
        result = {}
        for i, h in enumerate(self.prediction_horizons):
            result[f't+{h}'] = predictions[i]
        
        return result
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            评估指标
        """
        loss, accuracy = self.model.evaluate(X_test, y_test)
        
        # 详细评估
        y_pred = self.predict(X_test)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"测试集 Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        logger.info(f"\n混淆矩阵:\n{cm}")
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save(self, model_path: str, scaler_path: str):
        """保存模型和标准化器"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 保存模型
        self.model.save(model_path)
        logger.info(f"LSTM 模型已保存到 {model_path}")
        
        # 保存标准化器和配置
        config_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names_,
            'n_states': self.n_states,
            'sequence_length': self.sequence_length,
            'prediction_horizons': self.prediction_horizons,
            'horizon_loss_weights': self.horizon_loss_weights,
            'is_multistep': self.is_multistep,
        }
        with open(scaler_path, 'wb') as f:
            pickle.dump(config_data, f)
        logger.info(f"Scaler 和配置已保存到 {scaler_path}")
        
        # 向后兼容：也保存特征名称到单独文件
        if self.feature_names_ is not None:
            feature_names_path = scaler_path.replace('.pkl', '_feature_names.pkl')
            with open(feature_names_path, 'wb') as f:
                pickle.dump(self.feature_names_, f)
            logger.info(f"特征名称已保存到 {feature_names_path}")
    
    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> 'LSTMRegimeClassifier':
        """加载模型和标准化器"""
        # 加载模型
        model = keras.models.load_model(model_path)
        
        # 加载标准化器和配置
        with open(scaler_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # 检测是新格式（字典）还是旧格式（只有 scaler）
        if isinstance(saved_data, dict) and 'scaler' in saved_data:
            # 新格式（多步预测）
            scaler = saved_data['scaler']
            feature_names = saved_data.get('feature_names')
            n_states = saved_data.get('n_states', 6)
            sequence_length = saved_data.get('sequence_length', 64)
            prediction_horizons = saved_data.get('prediction_horizons', [1, 2, 3, 4])
            horizon_loss_weights = saved_data.get('horizon_loss_weights')
        else:
            # 旧格式（向后兼容，旧模型是单步预测）
            scaler = saved_data
            feature_names = None
            n_states = 6
            sequence_length = 64
            prediction_horizons = [1, 2, 3, 4]  # 强制转换为多步预测格式
            horizon_loss_weights = None
            
            # 尝试从单独文件加载特征名称
            feature_names_path = scaler_path.replace('.pkl', '_feature_names.pkl')
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'rb') as f:
                    feature_names = pickle.load(f)
            
            logger.warning(
                "检测到旧格式模型。旧模型将被视为多步预测模型（t+1 到 t+4），"
                "但实际预测能力可能受限。建议重新训练模型。"
            )
        
        # 创建实例（现在总是多步预测）
        classifier = cls(
            n_states=n_states,
            sequence_length=sequence_length,
            prediction_horizons=prediction_horizons,
            horizon_loss_weights=horizon_loss_weights
        )
        classifier.model = model
        classifier.scaler = scaler
        classifier.feature_names_ = feature_names
        # 现在总是多步预测
        classifier.is_multistep = True
        
        logger.info(f"LSTM 模型已从 {model_path} 加载")
        logger.info(f"  多步预测: True, 预测步数: {prediction_horizons}")
        if feature_names:
            logger.info(f"  特征数: {len(feature_names)}")
        else:
            logger.warning("未找到特征名称，这是旧版本模型。建议重新训练。")
        
        return classifier
    
    def prepare_live_data(
        self,
        features: pd.DataFrame
    ) -> np.ndarray:
        """
        准备实时数据（用于推理）
        
        Args:
            features: 特征 DataFrame（至少包含 sequence_length 行）
            
        Returns:
            准备好的序列数据
        """
        if len(features) < self.sequence_length:
            raise ValueError(
                f"数据长度不足，需要至少 {self.sequence_length} 行，当前只有 {len(features)} 行"
            )
        
        # 只取最后 sequence_length 行
        features = features.iloc[-self.sequence_length:]
        
        # 对齐特征（如果保存了特征名称）
        # 优先使用保存的特征名称，如果没有则使用 scaler 的 feature_names_in_
        feature_names = self.feature_names_
        if feature_names is None and hasattr(self.scaler, 'feature_names_in_'):
            feature_names = list(self.scaler.feature_names_in_)
        
        if feature_names is not None:
            missing_features = set(feature_names) - set(features.columns)
            extra_features = set(features.columns) - set(feature_names)
            
            if missing_features or extra_features:
                # 对齐特征：添加缺失的特征（填充0），移除多余的特征
                features = features.reindex(columns=feature_names, fill_value=0)
            else:
                # 特征名称一致，但需要确保顺序一致
                features = features[feature_names]
        elif hasattr(self.scaler, 'n_features_in_'):
            # 旧版本模型：只检查特征数量
            if len(features.columns) != self.scaler.n_features_in_:
                raise ValueError(
                    f"特征数量不匹配！训练时: {self.scaler.n_features_in_} 个特征, "
                    f"当前: {len(features.columns)} 个特征"
                )
        
        # 标准化
        features_scaled = self.scaler.transform(features)
        
        # 添加 batch 维度
        X = np.array([features_scaled])
        
        return X
