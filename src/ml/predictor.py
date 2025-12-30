"""
Unbreakable Predictor
======================

The main prediction engine that combines all components:
1. SVMD Signal Decomposition
2. GMM-HMM Regime Detection
3. TCN-LSTM-Attention + XGBoost + LightGBM Base Models
4. Stacking Meta-Learner
5. Risk Management (Kelly + Dynamic SL/TP)
6. Continuous Learning (EWC + Concept Drift)

This is the central orchestrator for making trading predictions.
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import joblib
from datetime import datetime

from .decomposition import SVMDDecomposer
from .regime import RegimeDetector, MarketRegime
from .models import create_base_models
from .ensemble import StackingEnsemble, EnsemblePrediction
from .risk import RiskManager, RiskAssessment
from .learning import ContinualLearner
from .features import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Complete trading signal with all components."""
    # Core signal
    direction: str  # BUY, SELL, NEUTRAL
    confidence: float  # 0.0 to 1.0
    probability: float  # Raw probability

    # Price levels
    entry_price: float
    stop_loss: float
    take_profit: float

    # Risk metrics
    position_size_pct: float
    risk_reward_ratio: float
    expected_value: float

    # Analysis details
    regime: str
    regime_confidence: float
    base_model_predictions: Dict[str, float]

    # Technical indicators
    rsi: float
    macd_hist: float
    atr: float
    bb_position: float

    # Metadata
    timestamp: datetime
    model_confidence: float  # Model's internal confidence
    drift_score: float  # Concept drift score

    # Warnings
    warnings: list


class UnbreakablePredictor:
    """
    The Unbreakable Trading Prediction System.

    Combines all research-backed components into a single, robust predictor.

    Architecture:
    1. Data → SVMD Decomposition → Feature Engineering
    2. Features → Regime Detection (GMM-HMM)
    3. Features + Regime → Base Models (TCN-LSTM, XGBoost, LightGBM)
    4. Base Predictions → Stacking Meta-Learner
    5. Final Prediction → Risk Management
    6. Continuous Learning monitors and adapts
    """

    # Signal thresholds (configurable)
    BUY_THRESHOLD = 0.55
    SELL_THRESHOLD = 0.45

    def __init__(
        self,
        config_path: str = "config.yaml",
        model_dir: str = "models/unbreakable",
        sequence_length: int = 60,
        hidden_size: int = 128,
        use_gpu: bool = True,
        capital: float = 100000.0
    ):
        """
        Initialize the Unbreakable Predictor.

        Args:
            config_path: Path to config file
            model_dir: Directory for model storage
            sequence_length: Sequence length for LSTM
            hidden_size: Hidden layer size
            use_gpu: Whether to use GPU
            capital: Trading capital for risk calculations
        """
        self.config_path = config_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.capital = capital
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.decomposer = SVMDDecomposer(max_modes=5)
        self.regime_detector = RegimeDetector()
        self.feature_engineer = FeatureEngineer(
            sequence_length=sequence_length,
            include_svmd=True,
            include_regime=True
        )
        self.risk_manager = RiskManager(
            kelly_fraction=0.25,
            max_risk_per_trade=0.02,
            atr_multiplier_sl=2.0,
            atr_multiplier_tp=4.0
        )

        # These will be initialized after fitting
        self.base_models: Dict[str, Any] = {}
        self.ensemble: Optional[StackingEnsemble] = None
        self.continual_learner: Optional[ContinualLearner] = None

        self._is_fitted = False
        self._last_prediction: Optional[TradingSignal] = None

        logger.info(f"UnbreakablePredictor initialized on {self.device}")

    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> 'UnbreakablePredictor':
        """
        Train the complete prediction system.

        Args:
            df: DataFrame with OHLCV data
            epochs: Training epochs for neural networks
            batch_size: Batch size for training
            validation_split: Validation split ratio
            verbose: Print progress

        Returns:
            self
        """
        logger.info("="*60)
        logger.info("TRAINING UNBREAKABLE PREDICTION SYSTEM")
        logger.info("="*60)

        # 1. Fit regime detector
        if verbose:
            logger.info("Step 1: Fitting regime detector (GMM-HMM)...")
        self.regime_detector.fit(df)

        # 2. Engineer features
        if verbose:
            logger.info("Step 2: Engineering features with SVMD...")
        feature_set = self.feature_engineer.fit_transform(df, self.regime_detector)

        n_features = feature_set.tabular_data.shape[1]
        logger.info(f"   Created {n_features} features from {len(df)} samples")

        # 3. Create base models
        if verbose:
            logger.info("Step 3: Creating base models...")
        self.base_models = create_base_models(
            input_size=n_features,
            hidden_size=self.hidden_size,
            sequence_length=self.sequence_length,
            use_gpu=(self.device.type == 'cuda')
        )

        # 4. Train base models
        if verbose:
            logger.info("Step 4: Training base models...")

        X = feature_set.tabular_data
        y = feature_set.targets
        X_seq = feature_set.sequence_data

        # Time series split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if X_seq is not None and len(X_seq) > 0:
            # Align sequence data with tabular
            X_seq_train = X_seq[:split_idx]
            X_seq_val = X_seq[split_idx:]
        else:
            X_seq_train, X_seq_val = None, None

        # Train each base model
        for name, model in self.base_models.items():
            if verbose:
                logger.info(f"   Training {name}...")

            if isinstance(model, torch.nn.Module):
                self._train_pytorch_model(
                    model, X_seq_train, y_train,
                    X_val=X_seq_val, y_val=y_val,
                    epochs=epochs, batch_size=batch_size
                )
            else:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False
                )

        # 5. Create and train stacking ensemble
        if verbose:
            logger.info("Step 5: Training stacking ensemble...")

        self.ensemble = StackingEnsemble(
            base_models=self.base_models,
            meta_learner='ridge',
            n_folds=5,
            use_probabilities=True
        )

        self.ensemble.fit(
            X_train, y_train,
            X_seq=X_seq_train,
            fit_base_models=False,  # Already trained
            verbose=verbose
        )

        # 6. Initialize continual learner
        if verbose:
            logger.info("Step 6: Initializing continual learning...")

        # Get the PyTorch model for continual learning
        pytorch_model = None
        for name, model in self.base_models.items():
            if isinstance(model, torch.nn.Module):
                pytorch_model = model
                break

        if pytorch_model is not None:
            self.continual_learner = ContinualLearner(
                model=pytorch_model,
                ewc_lambda=1000.0,
                replay_buffer_size=10000
            )

            # Create data loader for consolidation
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_seq_train if X_seq_train is not None else X_train),
                torch.FloatTensor(y_train)
            )
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

            self.continual_learner.consolidate(train_loader)

        self._is_fitted = True

        # 7. Evaluate
        if verbose:
            logger.info("Step 7: Evaluating model...")
            self._evaluate(X_val, y_val, X_seq_val)

        # 8. Save models
        self.save()

        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)

        return self

    def _train_pytorch_model(
        self,
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001
    ):
        """Train a PyTorch model."""
        model = model.to(self.device)
        model.train()

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                    val_outputs = model(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info(f"   Early stopping at epoch {epoch+1}")
                        break

    def _evaluate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_seq_val: np.ndarray = None
    ):
        """Evaluate model performance."""
        probs = self.ensemble.predict_proba(X_val, X_seq_val)
        preds = (probs > 0.5).astype(int)

        accuracy = (preds == y_val).mean()
        logger.info(f"   Validation Accuracy: {accuracy:.2%}")

        # High confidence accuracy
        high_conf_mask = (probs > 0.6) | (probs < 0.4)
        if high_conf_mask.sum() > 0:
            high_conf_acc = (preds[high_conf_mask] == y_val[high_conf_mask]).mean()
            logger.info(f"   High Confidence Accuracy: {high_conf_acc:.2%} ({high_conf_mask.sum()} samples)")

    def predict(self, df: pd.DataFrame) -> TradingSignal:
        """
        Make a trading prediction.

        Args:
            df: DataFrame with recent OHLCV data

        Returns:
            TradingSignal with complete analysis

        Raises:
            ValueError: If model not fitted or required columns missing
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Validate input DataFrame
        required_cols = ['close', 'high', 'low', 'open', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(df) < self.sequence_length:
            raise ValueError(f"Insufficient data: need at least {self.sequence_length} rows, got {len(df)}")

        # Get current price
        current_price = df['close'].iloc[-1]
        current_time = df['datetime'].iloc[-1] if 'datetime' in df.columns else datetime.now()

        warnings = []

        # 1. Detect regime
        regime_result = self.regime_detector.detect(df)
        regime = regime_result.current_regime.name
        regime_confidence = regime_result.confidence

        # 2. Engineer features
        feature_set = self.feature_engineer.transform(df, self.regime_detector)

        if len(feature_set.tabular_data) == 0:
            return self._create_neutral_signal(current_price, current_time, "Insufficient data")

        # 3. Get ensemble prediction
        X = feature_set.tabular_data[-1:] if len(feature_set.tabular_data) > 0 else None
        X_seq = feature_set.sequence_data[-1:] if feature_set.sequence_data is not None and len(feature_set.sequence_data) > 0 else None

        ensemble_pred = self.ensemble.predict_detailed(X, X_seq)

        probability = ensemble_pred.probability
        model_confidence = ensemble_pred.confidence

        # 4. Determine direction
        if probability > self.BUY_THRESHOLD:
            direction = 'BUY'
        elif probability < self.SELL_THRESHOLD:
            direction = 'SELL'
        else:
            direction = 'NEUTRAL'

        # 5. Get technical indicators from features
        df_features = feature_set.features
        rsi = df_features['rsi_14'].iloc[-1] if 'rsi_14' in df_features else 50
        macd_hist = df_features['macd_hist'].iloc[-1] if 'macd_hist' in df_features else 0
        atr = df_features['atr_14'].iloc[-1] if 'atr_14' in df_features else current_price * 0.02
        bb_position = df_features['bb_position'].iloc[-1] if 'bb_position' in df_features else 0.5

        # 6. Risk assessment
        win_probability = probability if direction == 'BUY' else (1 - probability)

        risk_assessment = self.risk_manager.assess_trade(
            capital=self.capital,
            entry_price=current_price,
            direction=direction,
            win_probability=win_probability,
            atr=atr,
            prediction_confidence=model_confidence
        )

        warnings.extend(risk_assessment.warnings)

        # 7. Check for concept drift
        drift_score = 0.0
        if self.continual_learner is not None:
            drift_score = self.continual_learner.drift_detector.get_drift_score()
            if drift_score > 0.5:
                warnings.append(f"High concept drift detected: {drift_score:.2f}")

        # 8. Adjust confidence based on regime
        final_confidence = model_confidence

        # Reduce confidence in uncertain regimes
        if regime_confidence < 0.5:
            final_confidence *= 0.8
            warnings.append("Low regime confidence - reduced signal confidence")

        # Reduce confidence if regime doesn't match signal
        if regime == 'BEAR' and direction == 'BUY':
            final_confidence *= 0.7
            warnings.append("BUY signal in BEAR regime - use caution")
        elif regime == 'BULL' and direction == 'SELL':
            final_confidence *= 0.7
            warnings.append("SELL signal in BULL regime - use caution")

        signal = TradingSignal(
            direction=direction,
            confidence=final_confidence,
            probability=probability,
            entry_price=current_price,
            stop_loss=risk_assessment.stop_loss,
            take_profit=risk_assessment.take_profit,
            position_size_pct=risk_assessment.position_size_pct,
            risk_reward_ratio=risk_assessment.risk_reward_ratio,
            expected_value=risk_assessment.expected_value,
            regime=regime,
            regime_confidence=regime_confidence,
            base_model_predictions=ensemble_pred.base_model_predictions,
            rsi=rsi,
            macd_hist=macd_hist,
            atr=atr,
            bb_position=bb_position,
            timestamp=current_time,
            model_confidence=model_confidence,
            drift_score=drift_score,
            warnings=warnings
        )

        self._last_prediction = signal
        return signal

    def _create_neutral_signal(
        self,
        price: float,
        timestamp: datetime,
        reason: str
    ) -> TradingSignal:
        """Create a neutral signal when prediction isn't possible."""
        return TradingSignal(
            direction='NEUTRAL',
            confidence=0.0,
            probability=0.5,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            position_size_pct=0.0,
            risk_reward_ratio=0.0,
            expected_value=0.0,
            regime='UNKNOWN',
            regime_confidence=0.0,
            base_model_predictions={},
            rsi=50,
            macd_hist=0,
            atr=0,
            bb_position=0.5,
            timestamp=timestamp,
            model_confidence=0.0,
            drift_score=0.0,
            warnings=[reason]
        )

    def update(
        self,
        df: pd.DataFrame,
        actual_outcome: int
    ):
        """
        Update model with new observation (for continuous learning).

        Args:
            df: DataFrame with OHLCV data
            actual_outcome: Actual outcome (1 if price went up, 0 if down)
        """
        if self.continual_learner is None:
            return

        # Get features
        feature_set = self.feature_engineer.transform(df, self.regime_detector)

        if len(feature_set.tabular_data) == 0:
            return

        # Get last prediction
        last_pred = self._last_prediction
        if last_pred is None:
            return

        # Update drift detector
        alert = self.continual_learner.update_drift_detector(
            prediction=last_pred.probability,
            actual=actual_outcome,
            confidence=last_pred.model_confidence
        )

        if alert:
            logger.warning(f"Drift alert: {alert.recommendation}")

        # Add to replay buffer
        features = feature_set.tabular_data[-1]
        self.continual_learner.add_to_replay(
            features=features,
            target=actual_outcome,
            regime=last_pred.regime
        )

    def save(self, path: str = None):
        """Save all models to disk."""
        if path is None:
            path = self.model_dir

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save ensemble
        if self.ensemble is not None:
            self.ensemble.save(path / 'ensemble')

        # Save regime detector
        joblib.dump(self.regime_detector, path / 'regime_detector.joblib')

        # Save feature engineer parameters
        joblib.dump({
            'feature_means': self.feature_engineer._feature_means,
            'feature_stds': self.feature_engineer._feature_stds,
            'feature_names': self.feature_engineer._feature_names
        }, path / 'feature_params.joblib')

        logger.info(f"Models saved to {path}")

    def load(self, path: str = None):
        """Load all models from disk."""
        if path is None:
            path = self.model_dir

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        # Load regime detector
        self.regime_detector = joblib.load(path / 'regime_detector.joblib')

        # Load feature parameters
        feature_params = joblib.load(path / 'feature_params.joblib')
        self.feature_engineer._feature_means = feature_params['feature_means']
        self.feature_engineer._feature_stds = feature_params['feature_stds']
        self.feature_engineer._feature_names = feature_params['feature_names']
        self.feature_engineer._is_fitted = True

        # Load ensemble
        if (path / 'ensemble').exists():
            # Recreate base models
            n_features = len(self.feature_engineer._feature_names)
            self.base_models = create_base_models(
                input_size=n_features,
                hidden_size=self.hidden_size,
                use_gpu=(self.device.type == 'cuda')
            )

            self.ensemble = StackingEnsemble(base_models=self.base_models)
            self.ensemble.load(path / 'ensemble')

        self._is_fitted = True
        logger.info(f"Models loaded from {path}")

    def get_status(self) -> Dict[str, Any]:
        """Get predictor status."""
        return {
            'is_fitted': self._is_fitted,
            'device': str(self.device),
            'n_base_models': len(self.base_models),
            'base_models': list(self.base_models.keys()),
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'continual_learning': self.continual_learner.get_status() if self.continual_learner else None,
            'last_prediction': self._last_prediction.direction if self._last_prediction else None
        }
