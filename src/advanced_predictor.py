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
    """Result from AdvancedPredictor."""
    direction: str  # "BUY", "SELL", "NEUTRAL"
    confidence: float  # 0.0 to 1.0

    # Individual algorithm outputs
    fourier_signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    fourier_cycle_phase: float  # 0.0 to 1.0

    kalman_trend: str  # "UP", "DOWN", "SIDEWAYS"
    kalman_smoothed_price: float

    entropy_regime: str  # "TRENDING", "CHOPPY", "VOLATILE"
    entropy_value: float

    markov_probability: float  # P(up | current_state)
    markov_state: str

    monte_carlo_risk: float  # Risk score 0-1
    monte_carlo_expected_return: float

    # Price levels
    stop_loss: float
    take_profit: float

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
        n_simulations: int = 1000,
        time_horizon: int = 10,
        default_volatility: float = 0.02
    ):
        """
        Args:
            n_simulations: Number of simulation paths
            time_horizon: Days to simulate forward
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
        Run Monte Carlo simulation.

        Args:
            current_price: Current asset price
            returns: Historical returns array
            stop_loss_pct: Stop loss as percentage
            take_profit_pct: Take profit as percentage

        Returns:
            Dict with simulation results
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

            # Run simulations
            final_prices = []
            hit_stop_loss = 0
            hit_take_profit = 0

            for _ in range(self.n_simulations):
                price = current_price
                for _ in range(self.time_horizon):
                    # GBM step
                    random_shock = np.random.normal()
                    price = price * np.exp(
                        (drift - 0.5 * volatility ** 2) * dt +
                        volatility * np.sqrt(dt) * random_shock
                    )

                    # Check stops
                    if price <= stop_loss_price:
                        hit_stop_loss += 1
                        break
                    elif price >= take_profit_price:
                        hit_take_profit += 1
                        break

                final_prices.append(price)

            final_prices = np.array(final_prices)

            # Calculate statistics
            expected_return = (np.mean(final_prices) - current_price) / current_price
            prob_profit = np.mean(final_prices > current_price)
            prob_stop_loss = hit_stop_loss / self.n_simulations
            prob_take_profit = hit_take_profit / self.n_simulations

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

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: Custom algorithm weights (must sum to 1.0)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

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
        lstm_probability: float = 0.5,
        atr: Optional[float] = None
    ) -> PredictionResult:
        """
        Generate prediction using ensemble of algorithms.

        Args:
            df: DataFrame with OHLCV data
            lstm_probability: Probability from LSTM model (0-1)
            atr: Average True Range for stop loss calculation

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

        # Weighted ensemble
        ensemble_prob = sum(
            self.weights[name] * prob
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

        return PredictionResult(
            direction=direction,
            confidence=float(confidence),
            fourier_signal=fourier_result['signal'],
            fourier_cycle_phase=fourier_result['cycle_phase'],
            kalman_trend=kalman_result['trend'],
            kalman_smoothed_price=kalman_result['smoothed_price'],
            entropy_regime=entropy_result['regime'],
            entropy_value=entropy_result['normalized_entropy'],
            markov_probability=markov_result['prob_up'],
            markov_state=markov_result['current_state'],
            monte_carlo_risk=monte_carlo_result['risk_score'],
            monte_carlo_expected_return=monte_carlo_result['expected_return'],
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            ensemble_weights=self.weights.copy()
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
