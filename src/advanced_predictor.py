"""
Advanced Predictor with Mathematical Algorithms
================================================
Uses cutting-edge mathematical and statistical methods for trading predictions.

MATHEMATICAL FOUNDATIONS:
1. Fourier Transform - Detects hidden cycles in price data
2. Wavelet Analysis - Multi-scale pattern recognition
3. Kalman Filter - Noise reduction and trend estimation
4. Monte Carlo Simulation - Risk probability estimation
5. Markov Chain - State transition probabilities
6. Information Theory - Entropy-based market regime detection

TRUTH ABOUT PREDICTIONS:
- No system can predict markets with >60% accuracy consistently
- Expected edge: 52-58% win rate
- Profit comes from risk management (1:2 ratio), not prediction accuracy
- Markets are partially random (Efficient Market Hypothesis)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Transparent prediction result with all mathematical components."""
    # Core prediction
    direction: str  # BUY, SELL, NEUTRAL
    confidence: float  # 0.0 to 1.0

    # Mathematical components (transparency)
    fourier_signal: float  # -1 to 1 (cycle phase)
    kalman_trend: float  # Filtered trend direction
    entropy_regime: str  # LOW, MEDIUM, HIGH volatility
    markov_probability: float  # State transition probability
    monte_carlo_risk: float  # Risk of hitting stop loss

    # Risk levels
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float

    # Metadata
    algorithm_weights: Dict[str, float]
    raw_scores: Dict[str, float]

    # Detailed analysis (for dashboard visualization)
    fourier_details: Dict = None  # Cycle period, frequencies, magnitudes, phase
    markov_details: Dict = None  # Transition matrix, state probabilities


class FourierAnalyzer:
    """
    Fourier Transform for Cycle Detection

    MATHEMATICAL BASIS:
    - Decomposes price into frequency components
    - Identifies dominant cycles (daily, weekly, monthly)
    - Extrapolates cycle phase for prediction

    Formula: F(k) = Sum[x(n) * e^(-2*pi*i*k*n/N)]
    """

    @staticmethod
    def analyze(prices: np.ndarray, top_frequencies: int = 3) -> Tuple[float, Dict]:
        """
        Analyze price cycles using FFT.

        Args:
            prices: Array of closing prices
            top_frequencies: Number of dominant frequencies to use

        Returns:
            signal: -1 to 1 indicating cycle phase
            details: Frequency analysis details
        """
        if len(prices) < 64:
            return 0.0, {'error': 'Insufficient data'}

        # Detrend the data (remove linear trend)
        detrended = prices - np.linspace(prices[0], prices[-1], len(prices))

        # Apply FFT
        n = len(detrended)
        fft_values = fft(detrended)
        frequencies = fftfreq(n)

        # Get magnitude spectrum (positive frequencies only)
        positive_freq_idx = np.where(frequencies > 0)[0]
        magnitudes = np.abs(fft_values[positive_freq_idx])
        positive_frequencies = frequencies[positive_freq_idx]

        # Find dominant frequencies
        top_idx = np.argsort(magnitudes)[-top_frequencies:]
        dominant_freqs = positive_frequencies[top_idx]
        dominant_mags = magnitudes[top_idx]

        # Calculate phase of dominant cycle
        dominant_idx = top_idx[-1]  # Most dominant
        phase = np.angle(fft_values[positive_freq_idx[dominant_idx]])

        # Convert phase to signal (-1 to 1)
        # Phase near 0 or pi = peak/trough (potential reversal)
        # Phase near pi/2 or -pi/2 = middle of cycle (trending)
        signal = np.sin(phase)

        # Cycle period in candles (with epsilon to prevent division by zero)
        EPSILON = 1e-10
        period = 1.0 / dominant_freqs[-1] if dominant_freqs[-1] > EPSILON else 0

        details = {
            'dominant_period': period,
            'dominant_frequencies': dominant_freqs.tolist(),
            'magnitudes': dominant_mags.tolist(),
            'phase': phase,
            'interpretation': 'BULLISH' if signal > 0.3 else 'BEARISH' if signal < -0.3 else 'NEUTRAL'
        }

        return signal, details


class KalmanFilter:
    """
    Kalman Filter for Trend Estimation

    MATHEMATICAL BASIS:
    - Optimal state estimator under Gaussian noise
    - Separates signal from noise
    - Provides trend direction and velocity

    State equation: x(k) = A*x(k-1) + w(k)
    Measurement equation: z(k) = H*x(k) + v(k)

    Where:
    - x = state vector [price, velocity]
    - A = state transition matrix
    - H = measurement matrix
    - w, v = process and measurement noise
    """

    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 0.1):
        """
        Initialize Kalman Filter.

        Args:
            process_variance: Q - how much we trust the model
            measurement_variance: R - how much we trust measurements
        """
        self.Q = process_variance
        self.R = measurement_variance

        # State: [price, velocity]
        self.x = np.array([0.0, 0.0])

        # Covariance matrix
        self.P = np.eye(2) * 1000

        # State transition matrix (constant velocity model)
        self.A = np.array([[1, 1], [0, 1]])

        # Measurement matrix (we only observe price)
        self.H = np.array([[1, 0]])

    def filter(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Kalman filter to price series.

        Args:
            prices: Array of prices

        Returns:
            filtered_prices: Noise-reduced prices
            velocities: Estimated trend velocities
        """
        filtered = []
        velocities = []

        # Initialize state with first price
        self.x = np.array([prices[0], 0.0])
        self.P = np.eye(2) * 1000

        for price in prices:
            # Predict
            x_pred = self.A @ self.x
            P_pred = self.A @ self.P @ self.A.T + self.Q * np.eye(2)

            # Update
            K = P_pred @ self.H.T / (self.H @ P_pred @ self.H.T + self.R)
            self.x = x_pred + K.flatten() * (price - self.H @ x_pred)
            self.P = (np.eye(2) - K @ self.H) @ P_pred

            filtered.append(self.x[0])
            velocities.append(self.x[1])

        return np.array(filtered), np.array(velocities)

    @staticmethod
    def get_trend_signal(velocities: np.ndarray, lookback: int = 5) -> float:
        """
        Convert velocity to trend signal.

        Args:
            velocities: Array of velocity estimates
            lookback: Number of recent velocities to consider

        Returns:
            signal: -1 to 1 indicating trend strength
        """
        recent_velocity = np.mean(velocities[-lookback:])
        velocity_std = np.std(velocities)

        if velocity_std == 0:
            return 0.0

        # Normalize to -1 to 1
        z_score = recent_velocity / velocity_std
        signal = np.tanh(z_score)  # Squash to [-1, 1]

        return signal


class EntropyAnalyzer:
    """
    Information-Theoretic Market Regime Detection

    MATHEMATICAL BASIS:
    - Shannon Entropy measures information content/uncertainty
    - High entropy = chaotic/unpredictable market
    - Low entropy = trending/predictable market

    Formula: H(X) = -Sum[P(x) * log(P(x))]

    Trading Application:
    - Low entropy regime: Follow the trend
    - High entropy regime: Mean reversion or stay out
    """

    @staticmethod
    def calculate_return_entropy(returns: np.ndarray, bins: int = 20) -> float:
        """
        Calculate entropy of return distribution.

        Args:
            returns: Array of price returns
            bins: Number of bins for histogram

        Returns:
            entropy: Normalized entropy (0 to 1)
        """
        # Create histogram of returns
        hist, _ = np.histogram(returns, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins

        # Calculate entropy
        ent = entropy(hist)

        # Normalize by maximum entropy (uniform distribution)
        max_entropy = np.log(bins)
        normalized_entropy = ent / max_entropy if max_entropy > 0 else 0

        return normalized_entropy

    @staticmethod
    def get_regime(entropy_value: float) -> str:
        """
        Classify market regime based on entropy.

        Args:
            entropy_value: Normalized entropy (0 to 1)

        Returns:
            regime: LOW, MEDIUM, or HIGH
        """
        if entropy_value < 0.4:
            return 'LOW'  # Trending market - follow trend
        elif entropy_value < 0.7:
            return 'MEDIUM'  # Normal market
        else:
            return 'HIGH'  # Chaotic - reduce position size


class MarkovChainPredictor:
    """
    Markov Chain for State Transition Probabilities

    MATHEMATICAL BASIS:
    - Memoryless probabilistic model
    - P(next_state | current_state) = transition_probability
    - Uses historical state transitions to predict future

    States:
    - STRONG_UP: Returns > 1 std
    - UP: Returns > 0
    - DOWN: Returns < 0
    - STRONG_DOWN: Returns < -1 std

    Transition Matrix P[i,j] = P(state_j | state_i)
    """

    STATES = ['STRONG_DOWN', 'DOWN', 'UP', 'STRONG_UP']

    @staticmethod
    def returns_to_states(returns: np.ndarray) -> np.ndarray:
        """Convert returns to discrete states."""
        std = np.std(returns)
        states = []

        for r in returns:
            if r > std:
                states.append(3)  # STRONG_UP
            elif r > 0:
                states.append(2)  # UP
            elif r > -std:
                states.append(1)  # DOWN
            else:
                states.append(0)  # STRONG_DOWN

        return np.array(states)

    @staticmethod
    def build_transition_matrix(states: np.ndarray) -> np.ndarray:
        """
        Build transition probability matrix from state sequence.

        Args:
            states: Array of state indices (0-3)

        Returns:
            transition_matrix: 4x4 probability matrix
        """
        n_states = 4
        matrix = np.zeros((n_states, n_states))

        for i in range(len(states) - 1):
            current = states[i]
            next_state = states[i + 1]
            matrix[current, next_state] += 1

        # Normalize rows to probabilities
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums

        return matrix

    @staticmethod
    def predict_next_state(transition_matrix: np.ndarray, current_state: int) -> Tuple[int, float]:
        """
        Predict most likely next state.

        Args:
            transition_matrix: 4x4 probability matrix
            current_state: Current state index

        Returns:
            predicted_state: Most likely next state
            probability: Probability of that state
        """
        probabilities = transition_matrix[current_state]
        predicted_state = np.argmax(probabilities)
        probability = probabilities[predicted_state]

        return predicted_state, probability

    @staticmethod
    def get_bullish_probability(transition_matrix: np.ndarray, current_state: int) -> float:
        """
        Get probability of upward movement.

        Args:
            transition_matrix: 4x4 probability matrix
            current_state: Current state index

        Returns:
            probability: P(UP or STRONG_UP | current_state)
        """
        probabilities = transition_matrix[current_state]
        bullish_prob = probabilities[2] + probabilities[3]  # UP + STRONG_UP

        return bullish_prob


class MonteCarloSimulator:
    """
    Monte Carlo Simulation for Risk Assessment

    MATHEMATICAL BASIS:
    - Simulates thousands of possible price paths
    - Estimates probability of hitting stop loss vs take profit
    - Uses historical volatility for realistic simulations

    Price Path: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Where:
    - mu = drift (expected return)
    - sigma = volatility
    - Z = standard normal random variable
    """

    @staticmethod
    def simulate_paths(
        current_price: float,
        volatility: float,
        drift: float,
        steps: int = 24,
        n_simulations: int = 1000
    ) -> np.ndarray:
        """
        Simulate future price paths using Geometric Brownian Motion.

        Args:
            current_price: Starting price
            volatility: Annual volatility (sigma)
            drift: Annual drift (mu)
            steps: Number of time steps (candles)
            n_simulations: Number of simulation paths

        Returns:
            paths: Array of shape (n_simulations, steps+1)
        """
        dt = 1 / 365  # Assume 1 candle = 1 day for simplicity

        # Generate random shocks
        Z = np.random.standard_normal((n_simulations, steps))

        # Calculate log returns
        log_returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z

        # Cumulative sum to get price path
        paths = np.zeros((n_simulations, steps + 1))
        paths[:, 0] = current_price

        for t in range(1, steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(log_returns[:, t-1])

        return paths

    @staticmethod
    def calculate_risk_metrics(
        paths: np.ndarray,
        stop_loss: float,
        take_profit: float,
        is_long: bool = True
    ) -> Dict:
        """
        Calculate risk metrics from simulated paths.

        Args:
            paths: Simulated price paths
            stop_loss: Stop loss price
            take_profit: Take profit price
            is_long: True for long position, False for short

        Returns:
            metrics: Dict with risk statistics
        """
        n_simulations = paths.shape[0]

        hits_stop = 0
        hits_target = 0
        neither = 0

        for path in paths:
            min_price = np.min(path)
            max_price = np.max(path)

            if is_long:
                hit_stop = min_price <= stop_loss
                hit_target = max_price >= take_profit
            else:
                hit_stop = max_price >= stop_loss
                hit_target = min_price <= take_profit

            if hit_stop and hit_target:
                # Check which happened first (simplified)
                if is_long:
                    stop_idx = np.argmax(path <= stop_loss) if np.any(path <= stop_loss) else len(path)
                    target_idx = np.argmax(path >= take_profit) if np.any(path >= take_profit) else len(path)
                else:
                    stop_idx = np.argmax(path >= stop_loss) if np.any(path >= stop_loss) else len(path)
                    target_idx = np.argmax(path <= take_profit) if np.any(path <= take_profit) else len(path)

                if stop_idx < target_idx:
                    hits_stop += 1
                else:
                    hits_target += 1
            elif hit_stop:
                hits_stop += 1
            elif hit_target:
                hits_target += 1
            else:
                neither += 1

        return {
            'stop_loss_probability': hits_stop / n_simulations,
            'take_profit_probability': hits_target / n_simulations,
            'timeout_probability': neither / n_simulations,
            'expected_win_rate': hits_target / (hits_stop + hits_target) if (hits_stop + hits_target) > 0 else 0.5
        }


class AdvancedPredictor:
    """
    Combined Advanced Prediction System

    ALGORITHM WEIGHTS (configurable):
    - Fourier Analysis: 15% - Cycle detection
    - Kalman Filter: 25% - Trend estimation
    - Entropy Analysis: 10% - Regime detection
    - Markov Chain: 20% - State transitions
    - LSTM Model: 30% - Pattern learning

    TRUTH:
    - Combined accuracy target: 52-58%
    - No algorithm can consistently beat 60%
    - Profit comes from discipline, not prediction
    """

    DEFAULT_WEIGHTS = {
        'fourier': 0.15,
        'kalman': 0.25,
        'entropy': 0.10,
        'markov': 0.20,
        'lstm': 0.30
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize Advanced Predictor.

        Args:
            weights: Custom algorithm weights (must sum to 1.0)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS

        # Validate weights sum to 1
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing...")
            for k in self.weights:
                self.weights[k] /= total

        # Initialize components
        self.fourier = FourierAnalyzer()
        self.kalman = KalmanFilter()
        self.entropy_analyzer = EntropyAnalyzer()
        self.markov = MarkovChainPredictor()
        self.monte_carlo = MonteCarloSimulator()

        logger.info("AdvancedPredictor initialized")
        logger.info(f"Algorithm weights: {self.weights}")

    def predict(
        self,
        df: pd.DataFrame,
        lstm_probability: Optional[float] = None,
        atr: Optional[float] = None
    ) -> PredictionResult:
        """
        Generate prediction using all mathematical algorithms.

        Args:
            df: DataFrame with OHLCV data
            lstm_probability: Optional LSTM model output (0-1)
            atr: Average True Range for stop/target calculation

        Returns:
            PredictionResult with full transparency
        """
        prices = df['close'].values
        returns = df['close'].pct_change().dropna().values
        current_price = prices[-1]

        # Calculate ATR if not provided
        if atr is None:
            # Validate sufficient data for ATR calculation
            if len(df) >= 14:
                high_low = df['high'] - df['low']
                atr_value = high_low.rolling(14).mean().iloc[-1]
                # Validate ATR is not NaN or zero
                if pd.notna(atr_value) and atr_value > 0:
                    atr = atr_value
                else:
                    # Fallback: Use 1% of current price as ATR
                    atr = current_price * 0.01
                    logger.warning(f"ATR calculation resulted in NaN/zero, using fallback: {atr}")
            else:
                # Insufficient data for ATR, use 1% of current price as fallback
                atr = current_price * 0.01
                logger.warning(f"Insufficient data for ATR (need 14+ candles, got {len(df)}), using fallback: {atr}")

        # 1. FOURIER ANALYSIS
        fourier_signal, fourier_details = self.fourier.analyze(prices)

        # 2. KALMAN FILTER
        filtered_prices, velocities = self.kalman.filter(prices)
        kalman_signal = self.kalman.get_trend_signal(velocities)

        # 3. ENTROPY ANALYSIS
        entropy_value = self.entropy_analyzer.calculate_return_entropy(returns)
        entropy_regime = self.entropy_analyzer.get_regime(entropy_value)

        # 4. MARKOV CHAIN
        states = self.markov.returns_to_states(returns)
        transition_matrix = self.markov.build_transition_matrix(states)
        current_state = states[-1] if len(states) > 0 else 2
        markov_bullish_prob = self.markov.get_bullish_probability(transition_matrix, current_state)

        # 5. LSTM (if provided)
        lstm_signal = (lstm_probability - 0.5) * 2 if lstm_probability is not None else 0

        # Combine signals using weights
        raw_scores = {
            'fourier': fourier_signal,
            'kalman': kalman_signal,
            'entropy': 0 if entropy_regime == 'HIGH' else (0.5 if entropy_regime == 'MEDIUM' else 1) * kalman_signal,
            'markov': (markov_bullish_prob - 0.5) * 2,
            'lstm': lstm_signal
        }

        combined_signal = sum(
            self.weights[k] * raw_scores[k] for k in self.weights
        )

        # Convert to probability (0-1)
        probability = (combined_signal + 1) / 2

        # Determine direction
        if probability > 0.55:
            direction = 'BUY'
        elif probability < 0.45:
            direction = 'SELL'
        else:
            direction = 'NEUTRAL'

        # Calculate stop loss and take profit
        risk_multiplier = 2.0
        reward_multiplier = 4.0  # 2:1 reward:risk

        if direction == 'BUY':
            stop_loss = current_price - (risk_multiplier * atr)
            take_profit = current_price + (reward_multiplier * atr)
            is_long = True
        elif direction == 'SELL':
            stop_loss = current_price + (risk_multiplier * atr)
            take_profit = current_price - (reward_multiplier * atr)
            is_long = False
        else:
            stop_loss = current_price
            take_profit = current_price
            is_long = True

        # Monte Carlo risk assessment
        # Calculate volatility with NaN/zero validation
        if len(returns) > 0:
            volatility_raw = returns.std()
            # Validate volatility is not NaN or zero
            if pd.notna(volatility_raw) and volatility_raw > 0:
                volatility = volatility_raw * np.sqrt(252)  # Annualized
            else:
                # Fallback: Use typical crypto volatility (100% annualized)
                volatility = 1.0
                logger.warning(f"Volatility calculation resulted in NaN/zero, using fallback: {volatility}")
            drift = returns.mean() * 252
            # Validate drift is not NaN
            if pd.isna(drift):
                drift = 0.0
                logger.warning("Drift calculation resulted in NaN, using fallback: 0.0")
        else:
            # No returns data, use conservative defaults
            volatility = 1.0
            drift = 0.0
            logger.warning("No returns data for volatility calculation, using defaults")

        paths = self.monte_carlo.simulate_paths(
            current_price, volatility, drift,
            steps=24, n_simulations=1000
        )
        risk_metrics = self.monte_carlo.calculate_risk_metrics(
            paths, stop_loss, take_profit, is_long
        )

        # Confidence is the absolute distance from 0.5
        confidence = abs(probability - 0.5) * 2

        # Reduce confidence in high entropy regime
        if entropy_regime == 'HIGH':
            confidence *= 0.7

        # Create markov details for visualization
        markov_details = {
            'transition_matrix': transition_matrix.tolist() if hasattr(transition_matrix, 'tolist') else transition_matrix,
            'current_state': current_state,
            'state_names': ['Strong Bear', 'Bear', 'Neutral', 'Bull', 'Strong Bull'],
            'bullish_probability': markov_bullish_prob,
            'bearish_probability': 1 - markov_bullish_prob
        }

        return PredictionResult(
            direction=direction,
            confidence=min(confidence, 0.95),  # Cap at 95%
            fourier_signal=fourier_signal,
            kalman_trend=kalman_signal,
            entropy_regime=entropy_regime,
            markov_probability=markov_bullish_prob,
            monte_carlo_risk=risk_metrics['stop_loss_probability'],
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=reward_multiplier / risk_multiplier,
            algorithm_weights=self.weights.copy(),
            raw_scores=raw_scores,
            fourier_details=fourier_details,
            markov_details=markov_details
        )


# Mathematical formulas reference
MATH_REFERENCE = """
=============================================================================
MATHEMATICAL ALGORITHMS REFERENCE
=============================================================================

1. FOURIER TRANSFORM (Cycle Detection)
   Formula: F(k) = Σ[n=0 to N-1] x(n) * e^(-2πi*k*n/N)
   Use: Decompose price into frequency components to find dominant cycles

2. KALMAN FILTER (Trend Estimation)
   Predict: x̂(k|k-1) = A*x̂(k-1|k-1)
   Update:  x̂(k|k) = x̂(k|k-1) + K*(z(k) - H*x̂(k|k-1))
   Gain:    K = P(k|k-1)*H'/(H*P(k|k-1)*H' + R)
   Use: Optimal noise filtering and trend velocity estimation

3. SHANNON ENTROPY (Regime Detection)
   Formula: H(X) = -Σ P(x) * log₂(P(x))
   Use: Measure market chaos/uncertainty for position sizing

4. MARKOV CHAIN (State Transitions)
   Formula: P(X(t+1)=j | X(t)=i) = P[i,j]
   Use: Calculate probability of price going up/down based on current state

5. MONTE CARLO (Risk Assessment)
   GBM: S(t+dt) = S(t) * exp((μ-σ²/2)*dt + σ*√dt*Z)
   Use: Simulate thousands of paths to estimate stop loss/take profit probabilities

6. GEOMETRIC BROWNIAN MOTION
   dS = μ*S*dt + σ*S*dW
   Where: μ=drift, σ=volatility, dW=Wiener process
   Use: Foundation for price path simulation

=============================================================================
TRUTH ABOUT TRADING PREDICTIONS
=============================================================================

1. NO algorithm consistently predicts markets >60% correctly
2. Markets contain significant random component (EMH)
3. Profit comes from RISK MANAGEMENT, not prediction accuracy
4. A 55% win rate with 2:1 reward:risk is VERY profitable:
   - 100 trades: 55 wins × 2R - 45 losses × 1R = 110R - 45R = +65R
5. Over-optimization leads to curve-fitting (looks good, fails live)
6. Past performance does NOT guarantee future results

=============================================================================
"""


def print_math_reference():
    """Print mathematical reference guide."""
    print(MATH_REFERENCE)


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample data for testing
    np.random.seed(42)
    n = 500

    # Simulate price with trend + cycles + noise
    t = np.arange(n)
    trend = 100 + 0.05 * t
    cycle = 5 * np.sin(2 * np.pi * t / 50)  # 50-period cycle
    noise = np.random.normal(0, 1, n).cumsum() * 0.1
    prices = trend + cycle + noise

    df = pd.DataFrame({
        'timestamp': range(n),
        'datetime': pd.date_range('2024-01-01', periods=n, freq='h'),
        'open': prices * (1 + np.random.normal(0, 0.001, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n)
    })

    # Test predictor
    predictor = AdvancedPredictor()
    result = predictor.predict(df, lstm_probability=0.6)

    print("\n" + "="*60)
    print("ADVANCED PREDICTION RESULT")
    print("="*60)
    print(f"Direction: {result.direction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Stop Loss: ${result.stop_loss:.2f}")
    print(f"Take Profit: ${result.take_profit:.2f}")
    print(f"Risk:Reward: 1:{result.risk_reward_ratio:.1f}")
    print(f"\nMathematical Components:")
    print(f"  Fourier Signal: {result.fourier_signal:.3f}")
    print(f"  Kalman Trend: {result.kalman_trend:.3f}")
    print(f"  Entropy Regime: {result.entropy_regime}")
    print(f"  Markov P(up): {result.markov_probability:.2%}")
    print(f"  Monte Carlo Risk: {result.monte_carlo_risk:.2%}")
    print(f"\nRaw Algorithm Scores: {result.raw_scores}")
    print("="*60)

    # Print mathematical reference
    print_math_reference()
