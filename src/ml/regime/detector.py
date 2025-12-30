"""
GMM-HMM Market Regime Detection
================================

Hidden Markov Model with Gaussian Mixture emissions for market regime detection.
Captures temporal dependencies in regime transitions.

Why HMM > GMM:
- GMM treats observations independently
- HMM models temporal transitions (today's regime depends on yesterday's)
- Market regimes have persistence (bull markets don't flip daily)

Research shows HMM outperformed buy-and-hold 2006-2023.

Sources:
- QuantStart: Market Regime Detection Using Hidden Markov Models
- LSEG Developer Portal 2024
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime states."""
    BULL = 0  # Trending up, low volatility
    BEAR = 1  # Trending down, high volatility
    SIDEWAYS = 2  # Mean-reverting, medium volatility


@dataclass
class RegimeResult:
    """Result of regime detection."""
    current_regime: MarketRegime
    regime_probabilities: Dict[MarketRegime, float]
    regime_history: List[MarketRegime]
    transition_matrix: np.ndarray
    regime_features: Dict[str, float]
    confidence: float


class GaussianHMM:
    """
    Gaussian Hidden Markov Model for regime detection.

    States emit observations from Gaussian distributions.
    Uses Baum-Welch algorithm for training and Viterbi for inference.

    Parameters:
    -----------
    n_states : int
        Number of hidden states (regimes)
    n_iter : int
        Maximum iterations for EM algorithm
    tol : float
        Convergence tolerance
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # Model parameters (to be learned)
        self.start_prob: Optional[np.ndarray] = None  # Initial state probabilities
        self.trans_prob: Optional[np.ndarray] = None  # Transition matrix
        self.means: Optional[np.ndarray] = None  # Emission means per state
        self.stds: Optional[np.ndarray] = None  # Emission stds per state

        self._is_fitted = False

    def fit(self, X: np.ndarray) -> 'GaussianHMM':
        """
        Fit HMM using Baum-Welch (EM) algorithm.

        Args:
            X: Observations array of shape (n_samples, n_features)

        Returns:
            self
        """
        np.random.seed(self.random_state)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Initialize parameters
        self._initialize_params(X)

        prev_log_likelihood = -np.inf

        for iteration in range(self.n_iter):
            # E-step: Compute responsibilities
            log_likelihood, gamma, xi = self._e_step(X)

            # M-step: Update parameters
            self._m_step(X, gamma, xi)

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                logger.debug(f"HMM converged at iteration {iteration}")
                break

            prev_log_likelihood = log_likelihood

        self._is_fitted = True
        return self

    def _initialize_params(self, X: np.ndarray):
        """Initialize model parameters using K-means-like approach."""
        n_samples, n_features = X.shape

        # Initialize start probabilities (uniform)
        self.start_prob = np.ones(self.n_states) / self.n_states

        # Initialize transition probabilities (high self-transition)
        self.trans_prob = np.eye(self.n_states) * 0.8 + 0.2 / self.n_states

        # Initialize emission parameters using quantiles
        self.means = np.zeros((self.n_states, n_features))
        self.stds = np.zeros((self.n_states, n_features))

        quantiles = np.linspace(0, 1, self.n_states + 2)[1:-1]

        for f in range(n_features):
            for s, q in enumerate(quantiles):
                self.means[s, f] = np.quantile(X[:, f], q)
                self.stds[s, f] = np.std(X[:, f]) / self.n_states

        # Ensure positive stds
        self.stds = np.maximum(self.stds, 1e-6)

    def _emission_prob(self, X: np.ndarray) -> np.ndarray:
        """Calculate emission probabilities P(x|state)."""
        n_samples = X.shape[0]
        emission = np.zeros((n_samples, self.n_states))

        for s in range(self.n_states):
            log_prob = np.sum(
                norm.logpdf(X, self.means[s], self.stds[s]),
                axis=1
            )
            emission[:, s] = np.exp(log_prob)

        # Normalize to avoid underflow
        emission = emission / (emission.sum(axis=1, keepdims=True) + 1e-10)
        return emission

    def _e_step(self, X: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """E-step: Forward-backward algorithm."""
        n_samples = X.shape[0]
        emission = self._emission_prob(X)

        # Forward pass (alpha)
        alpha = np.zeros((n_samples, self.n_states))
        alpha[0] = self.start_prob * emission[0]
        alpha[0] /= alpha[0].sum() + 1e-10

        for t in range(1, n_samples):
            alpha[t] = emission[t] * (alpha[t-1] @ self.trans_prob)
            alpha[t] /= alpha[t].sum() + 1e-10

        # Backward pass (beta)
        beta = np.zeros((n_samples, self.n_states))
        beta[-1] = 1.0

        for t in range(n_samples - 2, -1, -1):
            beta[t] = self.trans_prob @ (emission[t+1] * beta[t+1])
            beta[t] /= beta[t].sum() + 1e-10

        # Compute gamma (state probabilities)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10

        # Compute xi (transition probabilities)
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))

        for t in range(n_samples - 1):
            numerator = np.outer(alpha[t], emission[t+1] * beta[t+1]) * self.trans_prob
            xi[t] = numerator / (numerator.sum() + 1e-10)

        # Log likelihood
        log_likelihood = np.sum(np.log(alpha.sum(axis=1) + 1e-10))

        return log_likelihood, gamma, xi

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-step: Update model parameters."""
        # Update start probabilities
        self.start_prob = gamma[0] / (gamma[0].sum() + 1e-10)

        # Update transition probabilities
        xi_sum = xi.sum(axis=0)
        self.trans_prob = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-10)

        # Update emission parameters
        for s in range(self.n_states):
            weights = gamma[:, s:s+1]
            total_weight = weights.sum() + 1e-10

            self.means[s] = (weights * X).sum(axis=0) / total_weight
            diff = X - self.means[s]
            self.stds[s] = np.sqrt((weights * diff**2).sum(axis=0) / total_weight)

        # Ensure positive stds
        self.stds = np.maximum(self.stds, 1e-6)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.

        Args:
            X: Observations array

        Returns:
            Array of predicted states
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        emission = self._emission_prob(X)

        # Viterbi algorithm
        delta = np.zeros((n_samples, self.n_states))
        psi = np.zeros((n_samples, self.n_states), dtype=int)

        # Initialization
        delta[0] = np.log(self.start_prob + 1e-10) + np.log(emission[0] + 1e-10)

        # Recursion
        for t in range(1, n_samples):
            for s in range(self.n_states):
                trans_probs = delta[t-1] + np.log(self.trans_prob[:, s] + 1e-10)
                psi[t, s] = np.argmax(trans_probs)
                delta[t, s] = trans_probs[psi[t, s]] + np.log(emission[t, s] + 1e-10)

        # Backtracking
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(delta[-1])

        for t in range(n_samples - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get state probabilities for each observation.

        Args:
            X: Observations array

        Returns:
            Array of state probabilities (n_samples, n_states)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        _, gamma, _ = self._e_step(X)
        return gamma


class RegimeDetector:
    """
    Market Regime Detection using GMM-HMM.

    Detects three regimes:
    - BULL: High returns, low volatility (trending up)
    - BEAR: Low returns, high volatility (trending down)
    - SIDEWAYS: Near-zero returns, medium volatility (range-bound)

    Parameters:
    -----------
    lookback_window : int
        Number of periods to use for regime features
    min_regime_duration : int
        Minimum periods before regime can change
    """

    def __init__(
        self,
        lookback_window: int = 60,
        min_regime_duration: int = 5,
        n_regimes: int = 3
    ):
        self.lookback_window = lookback_window
        self.min_regime_duration = min_regime_duration
        self.n_regimes = n_regimes

        self.hmm = GaussianHMM(n_states=n_regimes)
        self._is_fitted = False
        self._regime_mapping: Dict[int, MarketRegime] = {}

    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit regime detector on historical data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            self
        """
        features = self._extract_features(df)
        self.hmm.fit(features)
        self._is_fitted = True

        # Map HMM states to regime labels based on characteristics
        self._map_regimes(df, features)

        logger.info("RegimeDetector fitted successfully")
        return self

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection."""
        close = df['close'].values

        # Returns
        returns = np.diff(np.log(close + 1e-10))
        returns = np.concatenate([[0], returns])

        # Volatility (rolling std of returns)
        volatility = pd.Series(returns).rolling(self.lookback_window).std().fillna(0).values

        # Trend (rolling mean of returns)
        trend = pd.Series(returns).rolling(self.lookback_window).mean().fillna(0).values

        # Stack features
        features = np.column_stack([returns, volatility, trend])

        # Remove NaN rows
        valid_mask = ~np.isnan(features).any(axis=1)
        return features[valid_mask]

    def _map_regimes(self, df: pd.DataFrame, features: np.ndarray):
        """Map HMM states to meaningful regime labels."""
        states = self.hmm.predict(features)

        # Calculate mean returns and volatility for each state
        state_characteristics = {}

        for state in range(self.n_regimes):
            mask = states == state
            if mask.sum() > 0:
                state_characteristics[state] = {
                    'mean_return': features[mask, 0].mean(),
                    'mean_volatility': features[mask, 1].mean()
                }

        # Sort states by return to assign labels
        sorted_states = sorted(
            state_characteristics.keys(),
            key=lambda s: state_characteristics[s]['mean_return']
        )

        # Assign: lowest return = BEAR, highest = BULL, middle = SIDEWAYS
        if len(sorted_states) >= 3:
            self._regime_mapping = {
                sorted_states[0]: MarketRegime.BEAR,
                sorted_states[1]: MarketRegime.SIDEWAYS,
                sorted_states[2]: MarketRegime.BULL
            }
        elif len(sorted_states) == 2:
            self._regime_mapping = {
                sorted_states[0]: MarketRegime.BEAR,
                sorted_states[1]: MarketRegime.BULL
            }
        else:
            self._regime_mapping = {0: MarketRegime.SIDEWAYS}

        logger.debug(f"Regime mapping: {self._regime_mapping}")

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """
        Detect current market regime.

        Args:
            df: DataFrame with recent OHLCV data

        Returns:
            RegimeResult with current regime and probabilities
        """
        if not self._is_fitted:
            # Fit on provided data if not fitted
            self.fit(df)

        features = self._extract_features(df)

        # Get state predictions
        states = self.hmm.predict(features)
        probabilities = self.hmm.predict_proba(features)

        # Get current state
        current_state = states[-1]
        current_regime = self._regime_mapping.get(current_state, MarketRegime.SIDEWAYS)

        # Convert probabilities to regime probabilities
        regime_probs = {}
        for state, regime in self._regime_mapping.items():
            regime_probs[regime] = probabilities[-1, state]

        # Ensure all regimes have a probability
        for regime in MarketRegime:
            if regime not in regime_probs:
                regime_probs[regime] = 0.0

        # Convert state history to regime history
        regime_history = [
            self._regime_mapping.get(s, MarketRegime.SIDEWAYS)
            for s in states
        ]

        # Calculate confidence (max probability)
        confidence = max(regime_probs.values())

        # Extract regime features
        regime_features = {
            'returns': float(features[-1, 0]),
            'volatility': float(features[-1, 1]),
            'trend': float(features[-1, 2]) if features.shape[1] > 2 else 0.0
        }

        return RegimeResult(
            current_regime=current_regime,
            regime_probabilities=regime_probs,
            regime_history=regime_history,
            transition_matrix=self.hmm.trans_prob,
            regime_features=regime_features,
            confidence=confidence
        )

    def get_regime_for_training(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get regime labels for each row in DataFrame.
        Useful for training regime-specific models.

        Returns:
            Array of regime indices (0=BULL, 1=BEAR, 2=SIDEWAYS)
        """
        if not self._is_fitted:
            self.fit(df)

        features = self._extract_features(df)
        states = self.hmm.predict(features)

        # Pad to match original DataFrame length
        n_original = len(df)
        n_features = len(states)
        n_pad = n_original - n_features

        if n_pad > 0:
            states = np.concatenate([np.full(n_pad, states[0]), states])

        return states
