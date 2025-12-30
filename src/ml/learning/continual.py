"""
Continual Learning System
==========================

Prevents catastrophic forgetting while allowing adaptation to new market conditions.

Key Techniques:
1. EWC (Elastic Weight Consolidation) - Preserve important weights
2. Experience Replay Buffer - Store and replay important samples
3. Concept Drift Detection - Detect when model needs retraining
4. Progressive Networks - Expand capacity for new tasks

The stability-plasticity dilemma:
- Too stable: Can't adapt to new patterns
- Too plastic: Forgets old patterns

EWC finds the optimal balance by identifying which weights are important
for past tasks and constraining their updates.

Sources:
- Kirkpatrick et al. 2017: "Overcoming catastrophic forgetting"
- Springer 2025: "Dynamic Neuroplastic Networks for Financial Decision Making"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert when concept drift is detected."""
    timestamp: datetime
    drift_type: str  # 'sudden', 'gradual', 'recurring'
    severity: float  # 0-1
    metric_name: str
    old_value: float
    new_value: float
    recommendation: str


class EWC:
    """
    Elastic Weight Consolidation (EWC).

    EWC prevents catastrophic forgetting by:
    1. Computing Fisher Information matrix after training on task A
    2. Using Fisher as importance weights when training on task B
    3. Penalizing changes to important weights

    Loss = L_B + λ * Σ F_i * (θ_i - θ*_A)²

    Where:
    - L_B: Loss on new task
    - λ: EWC importance weight
    - F_i: Fisher information for parameter i
    - θ_i: Current parameter value
    - θ*_A: Optimal parameter from task A
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        fisher_sample_size: int = 200
    ):
        """
        Initialize EWC.

        Args:
            model: PyTorch model to protect
            ewc_lambda: Importance weight for EWC penalty (higher = more conservative)
            fisher_sample_size: Number of samples to estimate Fisher information
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size

        self._fisher: Dict[str, torch.Tensor] = {}
        self._optimal_params: Dict[str, torch.Tensor] = {}
        self._is_consolidated = False

    def compute_fisher(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module = None
    ):
        """
        Compute Fisher Information matrix for current weights.

        Fisher Information measures how sensitive the output is to each parameter.
        High Fisher = important parameter for current task.

        Args:
            data_loader: DataLoader with training data
            criterion: Loss function (default: BCELoss)
        """
        if criterion is None:
            criterion = nn.BCELoss()

        self.model.eval()

        # Initialize Fisher
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        n_samples = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if n_samples >= self.fisher_sample_size:
                break

            self.model.zero_grad()

            outputs = self.model(inputs).squeeze()
            loss = criterion(outputs, targets.float())
            loss.backward()

            # Accumulate squared gradients (empirical Fisher)
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2

            n_samples += len(inputs)

        # Normalize by number of samples
        for n in fisher:
            fisher[n] /= n_samples

        self._fisher = fisher

        # Store optimal parameters
        self._optimal_params = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self._is_consolidated = True
        logger.info(f"Fisher computed from {n_samples} samples")

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty term.

        Returns:
            Penalty to add to loss function
        """
        if not self._is_consolidated:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for n, p in self.model.named_parameters():
            if n in self._fisher and n in self._optimal_params:
                penalty += (self._fisher[n] * (p - self._optimal_params[n]) ** 2).sum()

        return self.ewc_lambda * penalty

    def training_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = None
    ) -> float:
        """
        Perform one training step with EWC regularization.

        Args:
            inputs: Input batch
            targets: Target batch
            optimizer: PyTorch optimizer
            criterion: Loss function

        Returns:
            Total loss value
        """
        if criterion is None:
            criterion = nn.BCELoss()

        self.model.train()
        optimizer.zero_grad()

        outputs = self.model(inputs).squeeze()
        task_loss = criterion(outputs, targets.float())
        ewc_loss = self.penalty()
        total_loss = task_loss + ewc_loss

        total_loss.backward()
        optimizer.step()

        return total_loss.item()


class ExperienceReplayBuffer:
    """
    Experience Replay Buffer for storing important samples.

    Stores samples from past market conditions to replay during training,
    helping the model remember old patterns while learning new ones.

    Uses reservoir sampling for memory-efficient storage.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of samples to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self._total_seen = 0

    def add(
        self,
        features: np.ndarray,
        target: float,
        metadata: Optional[Dict] = None
    ):
        """
        Add sample to buffer using reservoir sampling.

        Args:
            features: Feature vector
            target: Target value
            metadata: Optional metadata (timestamp, regime, etc.)
        """
        sample = {
            'features': features,
            'target': target,
            'metadata': metadata or {},
            'added_at': self._total_seen
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            # Reservoir sampling
            idx = np.random.randint(0, self._total_seen + 1)
            if idx < self.capacity:
                self.buffer[idx] = sample

        self._total_seen += 1

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a batch from the buffer.

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            Tuple of (features, targets)
        """
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        features = np.array([self.buffer[i]['features'] for i in indices])
        targets = np.array([self.buffer[i]['target'] for i in indices])

        return features, targets

    def sample_by_regime(
        self,
        regime: str,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from specific market regime."""
        regime_samples = [
            s for s in self.buffer
            if s.get('metadata', {}).get('regime') == regime
        ]

        if not regime_samples:
            return self.sample(batch_size)

        batch_size = min(batch_size, len(regime_samples))
        indices = np.random.choice(len(regime_samples), batch_size, replace=False)

        features = np.array([regime_samples[i]['features'] for i in indices])
        targets = np.array([regime_samples[i]['target'] for i in indices])

        return features, targets

    def __len__(self) -> int:
        return len(self.buffer)


class ConceptDriftDetector:
    """
    Concept Drift Detection for financial time series.

    Detects when the underlying data distribution changes, signaling
    that models may need retraining.

    Methods:
    1. Performance monitoring - Track prediction accuracy
    2. Distribution monitoring - Track feature distributions
    3. Model confidence monitoring - Track prediction confidence

    Types of drift:
    - Sudden: Abrupt regime change (e.g., market crash)
    - Gradual: Slow shift over time
    - Recurring: Seasonal patterns
    """

    def __init__(
        self,
        window_size: int = 100,
        performance_threshold: float = 0.10,
        confidence_threshold: float = 0.15,
        distribution_threshold: float = 0.05
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Size of sliding window for statistics
            performance_threshold: Max allowed performance drop
            confidence_threshold: Max allowed confidence drop
            distribution_threshold: Max allowed distribution shift (KL divergence)
        """
        self.window_size = window_size
        self.performance_threshold = performance_threshold
        self.confidence_threshold = confidence_threshold
        self.distribution_threshold = distribution_threshold

        # Tracking windows
        self._performance_window = deque(maxlen=window_size)
        self._confidence_window = deque(maxlen=window_size)
        self._feature_windows: Dict[str, deque] = {}

        # Baseline statistics
        self._baseline_performance: Optional[float] = None
        self._baseline_confidence: Optional[float] = None
        self._baseline_distributions: Dict[str, Tuple[float, float]] = {}

        # Alert history
        self._alerts: List[DriftAlert] = []

    def update(
        self,
        prediction: float,
        actual: float,
        confidence: float,
        features: Optional[Dict[str, float]] = None
    ):
        """
        Update drift detector with new observation.

        Args:
            prediction: Model prediction (0-1)
            actual: Actual outcome (0 or 1)
            confidence: Model confidence (0-1)
            features: Optional feature values for distribution monitoring
        """
        # Performance (accuracy)
        correct = int((prediction > 0.5) == actual)
        self._performance_window.append(correct)

        # Confidence
        self._confidence_window.append(confidence)

        # Feature distributions
        if features:
            for name, value in features.items():
                if name not in self._feature_windows:
                    self._feature_windows[name] = deque(maxlen=self.window_size)
                self._feature_windows[name].append(value)

    def set_baseline(self):
        """Set baseline statistics from current windows."""
        if len(self._performance_window) > 0:
            self._baseline_performance = np.mean(self._performance_window)

        if len(self._confidence_window) > 0:
            self._baseline_confidence = np.mean(self._confidence_window)

        for name, window in self._feature_windows.items():
            if len(window) > 0:
                self._baseline_distributions[name] = (
                    np.mean(window),
                    np.std(window) + 1e-6
                )

        logger.info(f"Baseline set: perf={self._baseline_performance:.3f}, conf={self._baseline_confidence:.3f}")

    def detect(self) -> Optional[DriftAlert]:
        """
        Check for concept drift.

        Returns:
            DriftAlert if drift detected, None otherwise
        """
        if self._baseline_performance is None:
            return None

        alerts = []

        # Check performance drift
        if len(self._performance_window) >= self.window_size // 2:
            current_perf = np.mean(self._performance_window)
            perf_drop = self._baseline_performance - current_perf

            if perf_drop > self.performance_threshold:
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    drift_type='sudden' if perf_drop > 0.2 else 'gradual',
                    severity=min(perf_drop / self.performance_threshold, 1.0),
                    metric_name='accuracy',
                    old_value=self._baseline_performance,
                    new_value=current_perf,
                    recommendation='Consider retraining model'
                ))

        # Check confidence drift
        if len(self._confidence_window) >= self.window_size // 2:
            current_conf = np.mean(self._confidence_window)
            conf_drop = self._baseline_confidence - current_conf

            if conf_drop > self.confidence_threshold:
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    drift_type='gradual',
                    severity=min(conf_drop / self.confidence_threshold, 1.0),
                    metric_name='confidence',
                    old_value=self._baseline_confidence,
                    new_value=current_conf,
                    recommendation='Model uncertainty increasing - consider retraining'
                ))

        # Check feature distribution drift
        for name, (base_mean, base_std) in self._baseline_distributions.items():
            if name in self._feature_windows and len(self._feature_windows[name]) >= self.window_size // 2:
                current_values = list(self._feature_windows[name])
                current_mean = np.mean(current_values)
                current_std = np.std(current_values) + 1e-6

                # Simplified KL divergence for Gaussians
                kl_div = np.log(current_std / base_std) + \
                         (base_std**2 + (base_mean - current_mean)**2) / (2 * current_std**2) - 0.5

                if kl_div > self.distribution_threshold:
                    alerts.append(DriftAlert(
                        timestamp=datetime.now(),
                        drift_type='gradual',
                        severity=min(kl_div / self.distribution_threshold, 1.0),
                        metric_name=f'distribution_{name}',
                        old_value=base_mean,
                        new_value=current_mean,
                        recommendation=f'Feature {name} distribution shifted'
                    ))

        # Return most severe alert
        if alerts:
            most_severe = max(alerts, key=lambda a: a.severity)
            self._alerts.append(most_severe)
            return most_severe

        return None

    def get_drift_score(self) -> float:
        """
        Get overall drift score (0-1).

        0 = No drift
        1 = Severe drift
        """
        scores = []

        if self._baseline_performance is not None and len(self._performance_window) > 0:
            current_perf = np.mean(self._performance_window)
            perf_score = max(0, (self._baseline_performance - current_perf) / self.performance_threshold)
            scores.append(perf_score)

        if self._baseline_confidence is not None and len(self._confidence_window) > 0:
            current_conf = np.mean(self._confidence_window)
            conf_score = max(0, (self._baseline_confidence - current_conf) / self.confidence_threshold)
            scores.append(conf_score)

        return min(np.mean(scores), 1.0) if scores else 0.0


class ContinualLearner:
    """
    Complete Continual Learning System.

    Combines EWC, Experience Replay, and Concept Drift Detection
    for robust model adaptation.
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        replay_buffer_size: int = 10000,
        replay_batch_size: int = 32,
        drift_window: int = 100
    ):
        """
        Initialize continual learner.

        Args:
            model: PyTorch model
            ewc_lambda: EWC importance weight
            replay_buffer_size: Size of experience replay buffer
            replay_batch_size: Batch size for replay
            drift_window: Window size for drift detection
        """
        self.model = model
        self.ewc = EWC(model, ewc_lambda=ewc_lambda)
        self.replay_buffer = ExperienceReplayBuffer(capacity=replay_buffer_size)
        self.drift_detector = ConceptDriftDetector(window_size=drift_window)
        self.replay_batch_size = replay_batch_size

        self._retrain_count = 0

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = None,
        use_replay: bool = True
    ) -> Dict[str, float]:
        """
        Perform one continual learning training step.

        Args:
            inputs: Input batch
            targets: Target batch
            optimizer: Optimizer
            criterion: Loss function
            use_replay: Whether to include experience replay

        Returns:
            Dictionary with loss components
        """
        if criterion is None:
            criterion = nn.BCELoss()

        losses = {}

        # Main training with EWC
        total_loss = self.ewc.training_step(inputs, targets, optimizer, criterion)
        losses['total'] = total_loss

        # Experience replay
        if use_replay and len(self.replay_buffer) >= self.replay_batch_size:
            replay_features, replay_targets = self.replay_buffer.sample(self.replay_batch_size)
            replay_inputs = torch.FloatTensor(replay_features)
            replay_targets = torch.FloatTensor(replay_targets)

            replay_loss = self.ewc.training_step(replay_inputs, replay_targets, optimizer, criterion)
            losses['replay'] = replay_loss

        return losses

    def add_to_replay(
        self,
        features: np.ndarray,
        target: float,
        regime: Optional[str] = None
    ):
        """Add sample to experience replay buffer."""
        self.replay_buffer.add(
            features=features,
            target=target,
            metadata={'regime': regime, 'timestamp': datetime.now()}
        )

    def update_drift_detector(
        self,
        prediction: float,
        actual: float,
        confidence: float,
        features: Optional[Dict[str, float]] = None
    ) -> Optional[DriftAlert]:
        """
        Update drift detector and check for drift.

        Returns:
            DriftAlert if drift detected
        """
        self.drift_detector.update(prediction, actual, confidence, features)
        return self.drift_detector.detect()

    def consolidate(self, data_loader: torch.utils.data.DataLoader):
        """
        Consolidate current knowledge (compute Fisher for EWC).

        Call this after training on a new market period.
        """
        self.ewc.compute_fisher(data_loader)
        self.drift_detector.set_baseline()
        logger.info("Knowledge consolidated")

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if model should be retrained.

        Returns:
            Tuple of (should_retrain, reason)
        """
        drift_score = self.drift_detector.get_drift_score()

        if drift_score > 0.7:
            return True, f"High drift score: {drift_score:.2f}"

        # Check recent alerts
        recent_alerts = [
            a for a in self.drift_detector._alerts
            if (datetime.now() - a.timestamp).total_seconds() < 3600  # Last hour
        ]

        if len(recent_alerts) >= 3:
            return True, f"Multiple drift alerts: {len(recent_alerts)} in last hour"

        return False, "No retraining needed"

    def get_status(self) -> Dict[str, Any]:
        """Get continual learning status."""
        return {
            'ewc_consolidated': self.ewc._is_consolidated,
            'replay_buffer_size': len(self.replay_buffer),
            'drift_score': self.drift_detector.get_drift_score(),
            'retrain_count': self._retrain_count,
            'recent_alerts': len(self.drift_detector._alerts)
        }
