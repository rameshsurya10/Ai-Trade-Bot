"""
Advanced Predictor - Mathematical Analysis Algorithms
======================================================
Ensemble of advanced mathematical algorithms for price prediction.

Algorithms:
1. Fourier Transform - Cycle detection
2. Kalman Filter - State estimation and trend smoothing
3. Shannon Entropy - Market regime detection
4. Markov Chain - Transition probabilities
5. Monte Carlo - Risk assessment via GBM simulations
6. Ensemble - Weighted combination of all algorithms

All algorithms include numerical stability safeguards (epsilon checks).
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)

# Numerical stability constant
EPSILON = 1e-10


# =============================================================================
# RESULT CLASSES
# =============================================================================

@dataclass
class PredictionResult:
    """Result from AdvancedPredictor with comprehensive metrics."""
    direction: str  # "BUY", "SELL", "NEUTRAL"
    confidence: float  # 0.0 to 1.0

    # Individual algorithm outputs
    fourier_signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    fourier_cycle_phase: float  # 0.0 to 1.0
    fourier_dominant_period: float  # Dominant cycle length

    kalman_trend: str  # "UP", "DOWN", "SIDEWAYS"
    kalman_smoothed_price: float
    kalman_velocity: float  # Price momentum
    kalman_error_covariance: float  # Estimation uncertainty

    entropy_regime: str  # "TRENDING", "CHOPPY", "VOLATILE", "NORMAL"
    entropy_value: float  # Normalized entropy 0-1
    entropy_raw_value: float  # Raw entropy value
    entropy_n_samples: int  # Number of samples analyzed

    markov_probability: float  # P(up | current_state)
    markov_state: str  # Current market state
    markov_prob_down: float  # P(down | current_state)
    markov_prob_neutral: float  # P(neutral | current_state)

    monte_carlo_risk: float  # Risk score 0-1
    monte_carlo_expected_return: float  # Expected return
    monte_carlo_prob_profit: float  # Probability of profit
    monte_carlo_prob_stop_loss: float  # Probability of hitting SL
    monte_carlo_prob_take_profit: float  # Probability of hitting TP
    monte_carlo_var_5pct: float  # Value at Risk (5th percentile)
    monte_carlo_volatility_daily: float  # Daily volatility
    monte_carlo_volatility_annual: float  # Annualized volatility
    monte_carlo_drift_annual: float  # Annualized drift

    # Price levels
    stop_loss: float
    take_profit: float

    # Risk metrics (calculated)
    risk_reward_ratio: float  # R:R ratio
    expected_profit_pct: float  # Expected profit %
    expected_loss_pct: float  # Expected loss %
    kelly_fraction: float  # Kelly criterion position size (0-1)

    # Sentiment features (optional)
    sentiment_score: Optional[float] = None  # Overall sentiment -1 to 1
    sentiment_1h: Optional[float] = None  # 1-hour sentiment
    sentiment_6h: Optional[float] = None  # 6-hour sentiment
    sentiment_momentum: Optional[float] = None  # Sentiment trend
    news_volume_1h: Optional[int] = None  # Number of articles in last hour

    # 8 out of 10 Rule Validation
    # Each rule returns (passed: bool, description: str)
    rules_passed: int = 0  # Number of rules that passed
    rules_total: int = 10  # Total rules checked
    rules_details: List[Tuple[str, bool, str]] = field(default_factory=list)  # [(name, passed, reason)]

    # Meta
    ensemble_weights: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# FOURIER TRANSFORM ANALYZER
# =============================================================================

class FourierAnalyzer:
    """
    Fourier Transform for cycle detection.

    Detects dominant price cycles to identify where we are
    in the current market cycle (near top/bottom).
    """

    def __init__(self, n_harmonics: int = 5):
        """
        Args:
            n_harmonics: Number of dominant frequencies to use
        """
        self.n_harmonics = n_harmonics

    def analyze(self, prices: np.ndarray) -> Dict:
        """
        Analyze price series for dominant cycles.

        Args:
            prices: Price array (close prices)

        Returns:
            Dict with cycle analysis results
        """
        if len(prices) < 32:
            return self._default_result()

        try:
            # Detrend prices
            detrended = self._detrend(prices)

            # Apply FFT
            n = len(detrended)
            yf = fft(detrended)
            xf = fftfreq(n, 1.0)

            # Get power spectrum (positive frequencies only)
            power = np.abs(yf[:n // 2]) ** 2
            freqs = xf[:n // 2]

            # Find dominant frequencies (skip DC component)
            valid_idx = freqs > EPSILON
            power_valid = power[valid_idx]
            freqs_valid = freqs[valid_idx]

            if len(power_valid) == 0:
                return self._default_result()

            # Get top harmonics
            top_idx = np.argsort(power_valid)[-self.n_harmonics:]
            dominant_freqs = freqs_valid[top_idx]
            dominant_powers = power_valid[top_idx]

            # Calculate dominant period
            if dominant_freqs[-1] > EPSILON:
                dominant_period = 1.0 / dominant_freqs[-1]
            else:
                dominant_period = len(prices)

            # Determine cycle phase (0 = trough, 0.5 = peak)
            position_in_cycle = (len(prices) % dominant_period) / (dominant_period + EPSILON)
            cycle_phase = position_in_cycle

            # Signal based on phase
            if cycle_phase < 0.25:
                signal = "BULLISH"  # Rising from trough
            elif cycle_phase < 0.5:
                signal = "NEUTRAL"  # Near peak
            elif cycle_phase < 0.75:
                signal = "BEARISH"  # Falling from peak
            else:
                signal = "BULLISH"  # Near trough

            return {
                'signal': signal,
                'dominant_period': float(dominant_period),
                'cycle_phase': float(cycle_phase),
                'dominant_frequencies': dominant_freqs.tolist(),
                'power_spectrum': dominant_powers.tolist()
            }

        except Exception as e:
            logger.error(f"Fourier analysis error: {e}")
            return self._default_result()

    def _detrend(self, prices: np.ndarray) -> np.ndarray:
        """Remove linear trend from prices."""
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        trend = slope * x + intercept
        return prices - trend

    def _default_result(self) -> Dict:
        return {
            'signal': 'NEUTRAL',
            'dominant_period': 20.0,
            'cycle_phase': 0.5,
            'dominant_frequencies': [],
            'power_spectrum': []
        }


# =============================================================================
# KALMAN FILTER
# =============================================================================

class KalmanFilter:
    """
    Kalman Filter for price smoothing and trend estimation.

    Provides noise-filtered price estimate and trend direction.
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2
    ):
        """
        Args:
            process_variance: Q - process noise variance
            measurement_variance: R - measurement noise variance
        """
        self.Q = process_variance
        self.R = measurement_variance

    def filter(self, prices: np.ndarray) -> Dict:
        """
        Apply Kalman filter to price series.

        Args:
            prices: Price array

        Returns:
            Dict with filtered prices and trend info
        """
        if len(prices) < 2:
            return self._default_result(prices[-1] if len(prices) > 0 else 0)

        try:
            n = len(prices)

            # Initialize state
            x_est = prices[0]  # Initial estimate
            P = 1.0  # Initial error covariance

            filtered_prices = []
            velocities = []  # Price velocity (trend)

            for z in prices:
                # Prediction step
                x_pred = x_est
                P_pred = P + self.Q

                # Update step
                K = P_pred / (P_pred + self.R + EPSILON)  # Kalman gain
                x_est = x_pred + K * (z - x_pred)
                P = (1 - K) * P_pred

                filtered_prices.append(x_est)

            filtered_prices = np.array(filtered_prices)

            # Calculate velocity (trend)
            velocity = np.diff(filtered_prices)
            avg_velocity = np.mean(velocity[-10:]) if len(velocity) >= 10 else np.mean(velocity)

            # Determine trend
            if avg_velocity > EPSILON:
                trend = "UP"
            elif avg_velocity < -EPSILON:
                trend = "DOWN"
            else:
                trend = "SIDEWAYS"

            return {
                'smoothed_price': float(filtered_prices[-1]),
                'filtered_prices': filtered_prices.tolist(),
                'velocity': float(avg_velocity),
                'trend': trend,
                'error_covariance': float(P)
            }

        except Exception as e:
            logger.error(f"Kalman filter error: {e}")
            return self._default_result(prices[-1])

    def _default_result(self, price: float) -> Dict:
        return {
            'smoothed_price': float(price),
            'filtered_prices': [float(price)],
            'velocity': 0.0,
            'trend': 'SIDEWAYS',
            'error_covariance': 1.0
        }


# =============================================================================
# ENTROPY ANALYZER
# =============================================================================

class EntropyAnalyzer:
    """
    Shannon Entropy for market regime detection.

    High entropy = chaotic/choppy market
    Low entropy = trending market
    """

    def __init__(self, n_bins: int = 20, lookback: int = 50):
        """
        Args:
            n_bins: Number of bins for histogram
            lookback: Lookback period for entropy calculation
        """
        self.n_bins = n_bins
        self.lookback = lookback

    def analyze(self, returns: np.ndarray) -> Dict:
        """
        Calculate Shannon entropy of returns.

        Args:
            returns: Array of price returns

        Returns:
            Dict with entropy analysis
        """
        if len(returns) < self.lookback:
            return self._default_result()

        try:
            # Use recent returns
            recent_returns = returns[-self.lookback:]

            # Remove NaN/Inf
            recent_returns = recent_returns[np.isfinite(recent_returns)]
            if len(recent_returns) < 10:
                return self._default_result()

            # Calculate histogram (probability distribution)
            hist, bin_edges = np.histogram(recent_returns, bins=self.n_bins, density=True)

            # Calculate bin widths
            bin_widths = np.diff(bin_edges)

            # Probability for each bin
            probs = hist * bin_widths
            probs = probs[probs > EPSILON]  # Remove zero probabilities

            # Shannon entropy: H = -sum(p * log2(p))
            entropy = -np.sum(probs * np.log2(probs + EPSILON))

            # Normalize entropy (0-1 scale)
            max_entropy = np.log2(self.n_bins)
            normalized_entropy = entropy / (max_entropy + EPSILON)

            # Determine regime
            if normalized_entropy < 0.3:
                regime = "TRENDING"
            elif normalized_entropy < 0.6:
                regime = "NORMAL"
            elif normalized_entropy < 0.8:
                regime = "CHOPPY"
            else:
                regime = "VOLATILE"

            return {
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'regime': regime,
                'n_samples': len(recent_returns)
            }

        except Exception as e:
            logger.error(f"Entropy analysis error: {e}")
            return self._default_result()

    def _default_result(self) -> Dict:
        return {
            'entropy': 0.5,
            'normalized_entropy': 0.5,
            'regime': 'NORMAL',
            'n_samples': 0
        }


# =============================================================================
# MARKOV CHAIN
# =============================================================================

class MarkovChain:
    """
    Markov Chain for state transition probabilities.

    Models market as discrete states and calculates
    probability of transitions.
    """

    def __init__(self, n_states: int = 3):
        """
        Args:
            n_states: Number of market states (3 = down/neutral/up)
        """
        self.n_states = n_states
        self.states = ['DOWN', 'NEUTRAL', 'UP']

    def analyze(self, returns: np.ndarray) -> Dict:
        """
        Build transition matrix and calculate probabilities.

        Args:
            returns: Array of price returns

        Returns:
            Dict with Markov analysis
        """
        if len(returns) < 20:
            return self._default_result()

        try:
            # Discretize returns into states
            states = self._discretize_returns(returns)

            # Build transition matrix
            transition_matrix = self._build_transition_matrix(states)

            # Current state
            current_state_idx = states[-1]
            current_state = self.states[current_state_idx]

            # Probability of going up from current state
            prob_up = transition_matrix[current_state_idx, 2]  # UP state index
            prob_down = transition_matrix[current_state_idx, 0]  # DOWN state index

            # Steady state (long-term probabilities)
            steady_state = self._calculate_steady_state(transition_matrix)

            return {
                'current_state': current_state,
                'prob_up': float(prob_up),
                'prob_down': float(prob_down),
                'prob_neutral': float(1 - prob_up - prob_down),
                'transition_matrix': transition_matrix.tolist(),
                'steady_state': steady_state.tolist()
            }

        except Exception as e:
            logger.error(f"Markov chain error: {e}")
            return self._default_result()

    def _discretize_returns(self, returns: np.ndarray) -> np.ndarray:
        """Convert continuous returns to discrete states."""
        states = np.zeros(len(returns), dtype=int)

        # Use percentiles for thresholds
        lower = np.percentile(returns, 33)
        upper = np.percentile(returns, 67)

        states[returns < lower] = 0  # DOWN
        states[(returns >= lower) & (returns <= upper)] = 1  # NEUTRAL
        states[returns > upper] = 2  # UP

        return states

    def _build_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Build transition probability matrix."""
        n = self.n_states
        counts = np.zeros((n, n))

        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            counts[from_state, to_state] += 1

        # Normalize rows (add epsilon to prevent division by zero)
        row_sums = counts.sum(axis=1, keepdims=True) + EPSILON
        transition_matrix = counts / row_sums

        return transition_matrix

    def _calculate_steady_state(self, P: np.ndarray) -> np.ndarray:
        """Calculate steady state distribution."""
        try:
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(P.T)

            # Find eigenvector for eigenvalue = 1
            idx = np.argmin(np.abs(eigenvalues - 1))
            steady_state = np.real(eigenvectors[:, idx])
            steady_state = steady_state / (steady_state.sum() + EPSILON)

            return steady_state
        except:
            # Equal distribution fallback
            return np.ones(self.n_states) / self.n_states

    def _default_result(self) -> Dict:
        return {
            'current_state': 'NEUTRAL',
            'prob_up': 0.33,
            'prob_down': 0.33,
            'prob_neutral': 0.34,
            'transition_matrix': [[0.33, 0.34, 0.33]] * 3,
            'steady_state': [0.33, 0.34, 0.33]
        }


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

class MonteCarlo:
    """
    Monte Carlo simulation for risk assessment.

    Uses Geometric Brownian Motion (GBM) to simulate
    potential future price paths.
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        time_horizon: int = 365,
        default_volatility: float = 0.02
    ):
        """
        Args:
            n_simulations: Number of simulation paths (10K is statistically robust)
            time_horizon: Days to simulate forward (365 = 1 year)
            default_volatility: Default daily volatility if calculation fails
        """
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.default_volatility = default_volatility

    def simulate(
        self,
        current_price: float,
        returns: np.ndarray,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03
    ) -> Dict:
        """
        Run Monte Carlo simulation using vectorized NumPy operations.

        Uses Geometric Brownian Motion (GBM) to simulate price paths.
        Fully vectorized for performance (100K simulations in ~1 second).

        Args:
            current_price: Current asset price
            returns: Historical returns array
            stop_loss_pct: Stop loss as percentage (e.g., 0.02 = 2%)
            take_profit_pct: Take profit as percentage (e.g., 0.03 = 3%)

        Returns:
            Dict with simulation results including probabilities and risk metrics
        """
        if len(returns) < 20 or current_price <= 0:
            return self._default_result()

        try:
            # Calculate drift and volatility from historical data
            clean_returns = returns[np.isfinite(returns)]
            if len(clean_returns) < 10:
                return self._default_result()

            drift = np.mean(clean_returns)
            volatility = np.std(clean_returns)

            # Handle edge cases
            if np.isnan(volatility) or volatility < EPSILON:
                volatility = self.default_volatility
            if np.isnan(drift):
                drift = 0.0

            # Annualize
            drift_annual = drift * 252
            vol_annual = volatility * np.sqrt(252)

            # Daily parameters
            dt = 1 / 252

            # Price targets
            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)

            # Vectorized GBM simulation (100-1000x faster than loops)
            # Generate all random shocks at once: shape (n_simulations, time_horizon)
            random_shocks = np.random.normal(0, 1, (self.n_simulations, self.time_horizon))

            # Calculate log returns for each step
            log_returns = (drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * random_shocks

            # Cumulative sum to get price paths (in log space)
            cumulative_log_returns = np.cumsum(log_returns, axis=1)

            # Convert to price paths
            price_paths = current_price * np.exp(cumulative_log_returns)

            # Check for stop loss and take profit hits along each path
            hit_sl = np.any(price_paths <= stop_loss_price, axis=1)
            hit_tp = np.any(price_paths >= take_profit_price, axis=1)

            # For paths that hit both, determine which was hit first
            # Find first index where SL or TP was hit
            sl_indices = np.argmax(price_paths <= stop_loss_price, axis=1)
            tp_indices = np.argmax(price_paths >= take_profit_price, axis=1)

            # argmax returns 0 if no True found, so we need to handle that
            sl_indices = np.where(hit_sl, sl_indices, self.time_horizon + 1)
            tp_indices = np.where(hit_tp, tp_indices, self.time_horizon + 1)

            # Count paths where SL was hit first vs TP was hit first
            sl_hit_first = hit_sl & (sl_indices < tp_indices)
            tp_hit_first = hit_tp & (tp_indices < sl_indices)

            # Get final prices (last column of price paths)
            final_prices = price_paths[:, -1]

            # Override final price for paths that hit stops
            # If SL hit first, final price is SL price; if TP hit first, final price is TP price
            final_prices = np.where(sl_hit_first, stop_loss_price, final_prices)
            final_prices = np.where(tp_hit_first, take_profit_price, final_prices)

            # Calculate statistics
            expected_return = (np.mean(final_prices) - current_price) / current_price
            prob_profit = np.mean(final_prices > current_price)
            prob_stop_loss = np.mean(sl_hit_first)
            prob_take_profit = np.mean(tp_hit_first)

            # Value at Risk (VaR) - 5th percentile
            var_5 = np.percentile(final_prices, 5)
            var_pct = (current_price - var_5) / current_price

            # Risk score (0-1, higher = riskier)
            risk_score = min(1.0, max(0.0, prob_stop_loss + var_pct * 0.5))

            return {
                'expected_return': float(expected_return),
                'prob_profit': float(prob_profit),
                'prob_stop_loss': float(prob_stop_loss),
                'prob_take_profit': float(prob_take_profit),
                'var_5_pct': float(var_pct),
                'risk_score': float(risk_score),
                'volatility_daily': float(volatility),
                'volatility_annual': float(vol_annual),
                'drift_annual': float(drift_annual),
                'n_simulations': self.n_simulations
            }

        except Exception as e:
            logger.error(f"Monte Carlo error: {e}")
            return self._default_result()

    def _default_result(self) -> Dict:
        return {
            'expected_return': 0.0,
            'prob_profit': 0.5,
            'prob_stop_loss': 0.1,
            'prob_take_profit': 0.1,
            'var_5_pct': 0.05,
            'risk_score': 0.5,
            'volatility_daily': self.default_volatility,
            'volatility_annual': self.default_volatility * np.sqrt(252),
            'drift_annual': 0.0,
            'n_simulations': self.n_simulations
        }


# =============================================================================
# ADVANCED PREDICTOR (ENSEMBLE)
# =============================================================================

class AdvancedPredictor:
    """
    Advanced Predictor - Ensemble of mathematical algorithms.

    Combines:
    1. Fourier Transform - Cycle detection
    2. Kalman Filter - Trend smoothing
    3. Shannon Entropy - Regime detection
    4. Markov Chain - State transitions
    5. Monte Carlo - Risk assessment

    Usage:
        predictor = AdvancedPredictor()
        result = predictor.predict(df, lstm_probability, atr)
    """

    # Algorithm weights for ensemble (configurable)
    DEFAULT_WEIGHTS = {
        'lstm': 0.35,
        'fourier': 0.15,
        'kalman': 0.20,
        'markov': 0.15,
        'entropy': 0.10,
        'monte_carlo': 0.05
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None, prediction_validator=None):
        """
        Args:
            weights: Custom algorithm weights (must sum to 1.0)
            prediction_validator: PredictionValidator instance for streak tracking
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.prediction_validator = prediction_validator

        # Normalize weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Initialize analyzers
        self.fourier = FourierAnalyzer()
        self.kalman = KalmanFilter()
        self.entropy = EntropyAnalyzer()
        self.markov = MarkovChain()
        self.monte_carlo = MonteCarlo()

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        lstm_probability: float = 0.5,
        atr: Optional[float] = None,
        sentiment_features: Optional[Dict] = None
    ) -> PredictionResult:
        """
        Generate prediction using ensemble of algorithms.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "1h", "4h")
            lstm_probability: Probability from LSTM model (0-1)
            atr: Average True Range for stop loss calculation
            sentiment_features: Optional sentiment data from news analysis

        Returns:
            PredictionResult with comprehensive analysis
        """
        # Extract price data
        prices = df['close'].values
        current_price = float(prices[-1])

        # Calculate returns
        returns = np.diff(prices) / (prices[:-1] + EPSILON)

        # Calculate ATR if not provided
        if atr is None or np.isnan(atr):
            high_low = df['high'] - df['low']
            if len(high_low) >= 14:
                atr = float(high_low.rolling(14).mean().iloc[-1])
            else:
                atr = current_price * 0.02  # Default 2%

        if np.isnan(atr) or atr < EPSILON:
            atr = current_price * 0.02

        # Run all analyzers
        fourier_result = self.fourier.analyze(prices)
        kalman_result = self.kalman.filter(prices)
        entropy_result = self.entropy.analyze(returns)
        markov_result = self.markov.analyze(returns)

        # Monte Carlo with risk-adjusted stops
        stop_loss_pct = min(0.05, max(0.01, atr / current_price * 2))
        take_profit_pct = min(0.10, max(0.015, atr / current_price * 3))
        monte_carlo_result = self.monte_carlo.simulate(
            current_price, returns, stop_loss_pct, take_profit_pct
        )

        # Process sentiment features (if available)
        sentiment_score = None
        sentiment_1h = None
        sentiment_6h = None
        sentiment_momentum = None
        news_volume_1h = None
        sentiment_prob = 0.5

        if sentiment_features:
            sentiment_1h = sentiment_features.get('sentiment_1h', 0.0)
            sentiment_6h = sentiment_features.get('sentiment_6h', 0.0)
            sentiment_momentum = sentiment_features.get('sentiment_momentum', 0.0)
            news_volume_1h = sentiment_features.get('news_volume_1h', 0)

            # Calculate weighted sentiment score
            sentiment_score = (sentiment_1h * 0.6 + sentiment_6h * 0.4) if sentiment_1h is not None and sentiment_6h is not None else 0.0

            # Convert sentiment to probability (sentiment range -1 to 1, prob 0 to 1)
            sentiment_prob = 0.5 + (sentiment_score * 0.5)
            sentiment_prob = max(0.2, min(0.8, sentiment_prob))  # Clamp to 0.2-0.8

        # Calculate ensemble probability
        prob_scores = []

        # LSTM contribution
        prob_scores.append(('lstm', lstm_probability))

        # Fourier contribution
        fourier_prob = 0.5
        if fourier_result['signal'] == 'BULLISH':
            fourier_prob = 0.7
        elif fourier_result['signal'] == 'BEARISH':
            fourier_prob = 0.3
        prob_scores.append(('fourier', fourier_prob))

        # Kalman contribution
        kalman_prob = 0.5
        if kalman_result['trend'] == 'UP':
            kalman_prob = 0.65
        elif kalman_result['trend'] == 'DOWN':
            kalman_prob = 0.35
        prob_scores.append(('kalman', kalman_prob))

        # Markov contribution
        markov_prob = markov_result['prob_up']
        prob_scores.append(('markov', markov_prob))

        # Entropy contribution (regime-based adjustment)
        entropy_prob = 0.5
        if entropy_result['regime'] == 'TRENDING':
            # In trending markets, go with momentum
            entropy_prob = 0.6 if kalman_result['trend'] == 'UP' else 0.4
        prob_scores.append(('entropy', entropy_prob))

        # Monte Carlo contribution (risk adjustment)
        mc_prob = 0.5 + (monte_carlo_result['expected_return'] * 5)  # Scale expected return
        mc_prob = max(0.3, min(0.7, mc_prob))
        prob_scores.append(('monte_carlo', mc_prob))

        # Sentiment contribution (if available)
        if sentiment_features:
            prob_scores.append(('sentiment', sentiment_prob))
            # Adjust weights to include sentiment (5%)
            weights_with_sentiment = self.weights.copy()
            weights_with_sentiment['sentiment'] = 0.05
            # Normalize other weights
            total_other = sum(self.weights.values())
            for key in self.weights:
                weights_with_sentiment[key] = self.weights[key] * 0.95 / total_other
        else:
            weights_with_sentiment = self.weights

        # Weighted ensemble
        ensemble_prob = sum(
            weights_with_sentiment.get(name, 0) * prob
            for name, prob in prob_scores
        )
        ensemble_prob = max(0.0, min(1.0, ensemble_prob))

        # Determine direction and confidence
        if ensemble_prob > 0.6:
            direction = "BUY"
            confidence = (ensemble_prob - 0.5) * 2
        elif ensemble_prob < 0.4:
            direction = "SELL"
            confidence = (0.5 - ensemble_prob) * 2
        else:
            direction = "NEUTRAL"
            confidence = 1 - abs(ensemble_prob - 0.5) * 4

        # Adjust confidence by regime
        if entropy_result['regime'] in ['CHOPPY', 'VOLATILE']:
            confidence *= 0.7  # Reduce confidence in choppy/volatile markets

        # Calculate stop loss and take profit
        if direction == "BUY":
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
        elif direction == "SELL":
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
        else:
            stop_loss = current_price - atr
            take_profit = current_price + atr

        # Calculate risk metrics
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(take_profit - current_price)
        risk_reward_ratio = reward_amount / (risk_amount + EPSILON)

        expected_profit_pct = abs((take_profit - current_price) / current_price)
        expected_loss_pct = abs((current_price - stop_loss) / current_price)

        # Position sizing using Kelly Criterion approximation
        win_prob = monte_carlo_result['prob_profit']
        kelly_fraction = (win_prob * (1 + risk_reward_ratio) - 1) / (risk_reward_ratio + EPSILON)
        kelly_fraction = max(0.0, min(0.25, kelly_fraction))  # Cap at 25% max

        # =================================================================
        # 8 OUT OF 10 RULE VALIDATION
        # =================================================================
        rules_details = []

        # Rule 1: Trend Alignment (Kalman agrees with signal)
        trend_aligned = (
            (direction == "BUY" and kalman_result['trend'] == "UP") or
            (direction == "SELL" and kalman_result['trend'] == "DOWN") or
            direction == "NEUTRAL"
        )
        rules_details.append(("Trend Alignment", trend_aligned,
            f"Kalman: {kalman_result['trend']}" if trend_aligned else f"Kalman shows {kalman_result['trend']}"))

        # Rule 2: Cycle Position (Fourier - not buying at top, not selling at bottom)
        cycle_phase = fourier_result['cycle_phase']
        cycle_ok = (
            (direction == "BUY" and cycle_phase < 0.7) or
            (direction == "SELL" and cycle_phase > 0.3) or
            direction == "NEUTRAL"
        )
        rules_details.append(("Cycle Position", cycle_ok,
            f"Phase {cycle_phase:.0%}" if cycle_ok else f"Phase {cycle_phase:.0%} - bad entry"))

        # Rule 3: Market Regime (Not choppy/volatile for trades)
        regime = entropy_result['regime']
        regime_ok = regime in ['TRENDING', 'NORMAL'] or direction == "NEUTRAL"
        rules_details.append(("Market Regime", regime_ok,
            f"{regime}" if regime_ok else f"{regime} - too risky"))

        # Rule 4: Markov Probability (>55% for direction)
        markov_ok = (
            (direction == "BUY" and markov_result['prob_up'] > 0.55) or
            (direction == "SELL" and markov_result['prob_down'] > 0.55) or
            direction == "NEUTRAL"
        )
        prob_used = markov_result['prob_up'] if direction == "BUY" else markov_result['prob_down']
        rules_details.append(("Markov Probability", markov_ok,
            f"{prob_used:.0%}" if markov_ok else f"Only {prob_used:.0%}"))

        # Rule 5: Monte Carlo Win Rate (>50%)
        mc_win_ok = monte_carlo_result['prob_profit'] > 0.50
        rules_details.append(("Win Probability", mc_win_ok,
            f"{monte_carlo_result['prob_profit']:.0%}" if mc_win_ok else f"Only {monte_carlo_result['prob_profit']:.0%}"))

        # Rule 6: Risk/Reward Ratio (>1.5)
        rr_ok = risk_reward_ratio >= 1.5
        rules_details.append(("Risk/Reward", rr_ok,
            f"1:{risk_reward_ratio:.1f}" if rr_ok else f"1:{risk_reward_ratio:.1f} - too low"))

        # Rule 7: Volatility Check (Daily vol < 5%)
        vol_ok = monte_carlo_result['volatility_daily'] < 0.05
        rules_details.append(("Volatility", vol_ok,
            f"{monte_carlo_result['volatility_daily']*100:.1f}%" if vol_ok else f"{monte_carlo_result['volatility_daily']*100:.1f}% - too high"))

        # Rule 8: Confidence Level (>60%)
        conf_ok = confidence > 0.60
        rules_details.append(("Confidence", conf_ok,
            f"{confidence*100:.0f}%" if conf_ok else f"Only {confidence*100:.0f}%"))

        # Rule 9: Position Size Valid (Kelly > 1%)
        kelly_ok = kelly_fraction > 0.01 or direction == "NEUTRAL"
        rules_details.append(("Position Size", kelly_ok,
            f"{kelly_fraction*100:.1f}%" if kelly_ok else "Too small"))

        # Rule 10: Fourier Signal Agreement
        fourier_ok = (
            (direction == "BUY" and fourier_result['signal'] in ["BULLISH", "NEUTRAL"]) or
            (direction == "SELL" and fourier_result['signal'] in ["BEARISH", "NEUTRAL"]) or
            direction == "NEUTRAL"
        )
        rules_details.append(("Fourier Signal", fourier_ok,
            fourier_result['signal'] if fourier_ok else f"{fourier_result['signal']} disagrees"))

        # Count passed rules
        rules_passed = sum(1 for _, passed, _ in rules_details if passed)

        # Record prediction for validation tracking (8/10 streak system)
        if self.prediction_validator:
            try:
                self.prediction_validator.record_prediction(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    current_price=current_price,
                    confidence=float(confidence),
                    target_price=current_price,  # Will be compared to next candle close
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    rules_passed=rules_passed,
                    rules_total=10,
                    market_context={
                        'regime': entropy_result['regime'],
                        'trend': kalman_result['trend'],
                        'volatility': monte_carlo_result['volatility_daily'],
                        'cycle_phase': fourier_result['cycle_phase']
                    }
                )
                logger.info(
                    f"✅ Prediction recorded: {symbol} {timeframe} {direction} "
                    f"@ ${current_price:,.2f} (conf: {confidence:.1%}, rules: {rules_passed}/10)"
                )
            except Exception as e:
                logger.error(f"❌ Failed to record prediction: {e}")

        return PredictionResult(
            # Core signal
            direction=direction,
            confidence=float(confidence),

            # Fourier analysis
            fourier_signal=fourier_result['signal'],
            fourier_cycle_phase=fourier_result['cycle_phase'],
            fourier_dominant_period=fourier_result['dominant_period'],

            # Kalman filter
            kalman_trend=kalman_result['trend'],
            kalman_smoothed_price=kalman_result['smoothed_price'],
            kalman_velocity=kalman_result['velocity'],
            kalman_error_covariance=kalman_result['error_covariance'],

            # Entropy & regime
            entropy_regime=entropy_result['regime'],
            entropy_value=entropy_result['normalized_entropy'],
            entropy_raw_value=entropy_result['entropy'],
            entropy_n_samples=entropy_result['n_samples'],

            # Markov chain
            markov_probability=markov_result['prob_up'],
            markov_state=markov_result['current_state'],
            markov_prob_down=markov_result['prob_down'],
            markov_prob_neutral=markov_result['prob_neutral'],

            # Monte Carlo simulation
            monte_carlo_risk=monte_carlo_result['risk_score'],
            monte_carlo_expected_return=monte_carlo_result['expected_return'],
            monte_carlo_prob_profit=monte_carlo_result['prob_profit'],
            monte_carlo_prob_stop_loss=monte_carlo_result['prob_stop_loss'],
            monte_carlo_prob_take_profit=monte_carlo_result['prob_take_profit'],
            monte_carlo_var_5pct=monte_carlo_result['var_5_pct'],
            monte_carlo_volatility_daily=monte_carlo_result['volatility_daily'],
            monte_carlo_volatility_annual=monte_carlo_result['volatility_annual'],
            monte_carlo_drift_annual=monte_carlo_result['drift_annual'],

            # Price levels
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),

            # Risk metrics
            risk_reward_ratio=float(risk_reward_ratio),
            expected_profit_pct=float(expected_profit_pct),
            expected_loss_pct=float(expected_loss_pct),
            kelly_fraction=float(kelly_fraction),

            # Sentiment (optional)
            sentiment_score=sentiment_score,
            sentiment_1h=sentiment_1h,
            sentiment_6h=sentiment_6h,
            sentiment_momentum=sentiment_momentum,
            news_volume_1h=news_volume_1h,

            # 8 out of 10 Rule Validation
            rules_passed=rules_passed,
            rules_total=10,
            rules_details=rules_details,

            # Meta
            ensemble_weights=weights_with_sentiment.copy()
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AdvancedPredictor',
    'PredictionResult',
    'FourierAnalyzer',
    'KalmanFilter',
    'EntropyAnalyzer',
    'MarkovChain',
    'MonteCarlo',
]
