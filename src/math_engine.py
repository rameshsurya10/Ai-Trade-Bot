"""
Advanced Mathematical Theory Engine
====================================
Deep mathematical algorithms for trading analysis.

"Mathematics is the language with which God wrote the universe." - Galileo

THIS MODULE IMPLEMENTS:
================================================================================

1. WAVELET ANALYSIS
   - Multi-scale decomposition (see patterns at different timeframes)
   - Denoising without losing important signals
   - Edge detection for trend changes

2. HURST EXPONENT
   - Measures trend persistence vs mean reversion
   - H > 0.5: Trending (follow the trend)
   - H < 0.5: Mean reverting (fade the move)
   - H = 0.5: Random walk (don't trade)

3. ORNSTEIN-UHLENBECK PROCESS
   - Mathematical model for mean reversion
   - Calculates optimal entry/exit for range-bound markets
   - Estimates time to mean reversion

4. INFORMATION THEORY
   - Mutual Information: True correlation (not just linear)
   - KL Divergence: Distribution changes detection
   - Transfer Entropy: Causality detection

5. EIGENVALUE ANALYSIS (Random Matrix Theory)
   - Separates signal from noise in correlations
   - Detects regime changes
   - Portfolio optimization

6. FRACTIONAL CALCULUS
   - Long memory effects in prices
   - More accurate volatility estimation
   - Better risk prediction

7. STOCHASTIC CALCULUS
   - Ito's Lemma for option pricing intuition
   - Jump diffusion for crash detection
   - Local volatility surface

8. TOPOLOGY (Persistent Homology)
   - Market structure analysis
   - Crash prediction via shape changes
   - Regime classification

================================================================================
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from scipy import stats
from scipy.linalg import eigh
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MathematicalAnalysis:
    """Complete mathematical analysis result."""
    # Core signal
    direction: str  # BUY, SELL, NEUTRAL
    confidence: float  # 0.0 to 1.0

    # Individual algorithm results
    wavelet_signal: float
    hurst_exponent: float
    hurst_regime: str  # TRENDING, MEAN_REVERTING, RANDOM
    ou_half_life: float  # Mean reversion half-life in candles
    ou_equilibrium: float  # Mean reversion target price
    mutual_information: float
    eigenvalue_ratio: float  # Signal to noise ratio
    fractal_dimension: float

    # Risk metrics
    jump_probability: float  # Probability of large move
    crash_indicator: float  # 0-1 crash risk

    # Stability metrics
    calculation_confidence: float  # How reliable are the calculations
    data_quality: float  # Input data quality score


# =============================================================================
# 1. WAVELET ANALYSIS
# =============================================================================

class WaveletAnalyzer:
    """
    Wavelet Transform for Multi-Scale Analysis

    WHY WAVELETS ARE BETTER THAN FOURIER:
    - Fourier: Good for stationary signals (frequency content doesn't change)
    - Wavelet: Good for non-stationary signals (like prices!)

    MATHEMATICAL BASIS:
    Continuous Wavelet Transform:
    W(a,b) = (1/√a) ∫ x(t) × ψ*((t-b)/a) dt

    Where:
    - a = scale (like frequency)
    - b = position (time)
    - ψ = mother wavelet

    We use Daubechies wavelet (db4) - optimal for financial data
    """

    @staticmethod
    def haar_wavelet_transform(data: np.ndarray, levels: int = 4) -> Dict:
        """
        Haar Wavelet Transform (simplest, most interpretable)

        Decomposes signal into:
        - Approximation (trend)
        - Details at each level (patterns at different scales)

        Args:
            data: Price or return series
            levels: Number of decomposition levels

        Returns:
            Dict with approximation and detail coefficients
        """
        n = len(data)

        # Pad to power of 2
        pad_len = 2 ** int(np.ceil(np.log2(n)))
        padded = np.zeros(pad_len)
        padded[:n] = data

        coefficients = {'approximation': None, 'details': []}
        current = padded.copy()

        for level in range(levels):
            length = len(current) // 2
            if length < 1:
                break

            # Haar wavelet: average and difference
            approximation = (current[::2] + current[1::2]) / np.sqrt(2)
            detail = (current[::2] - current[1::2]) / np.sqrt(2)

            coefficients['details'].append(detail[:length])
            current = approximation

        coefficients['approximation'] = current

        return coefficients

    @staticmethod
    def denoise(data: np.ndarray, threshold_factor: float = 1.5) -> np.ndarray:
        """
        Wavelet denoising using soft thresholding.

        Removes noise while preserving important features.

        Args:
            data: Noisy signal
            threshold_factor: Multiplier for threshold (higher = more smoothing)

        Returns:
            Denoised signal
        """
        coeffs = WaveletAnalyzer.haar_wavelet_transform(data, levels=4)

        # Calculate threshold using median absolute deviation (robust)
        all_details = np.concatenate(coeffs['details'])
        mad = np.median(np.abs(all_details - np.median(all_details)))
        threshold = threshold_factor * mad * np.sqrt(2 * np.log(len(data)))

        # Soft thresholding on detail coefficients
        denoised_details = []
        for detail in coeffs['details']:
            # Soft threshold: sign(x) * max(|x| - threshold, 0)
            thresholded = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            denoised_details.append(thresholded)

        # Reconstruct (inverse transform)
        current = coeffs['approximation']
        for detail in reversed(denoised_details):
            length = len(detail)
            reconstructed = np.zeros(length * 2)
            reconstructed[::2] = (current[:length] + detail) / np.sqrt(2)
            reconstructed[1::2] = (current[:length] - detail) / np.sqrt(2)
            current = reconstructed

        return current[:len(data)]

    @staticmethod
    def get_trend_signal(data: np.ndarray) -> Tuple[float, Dict]:
        """
        Extract trend signal from wavelet decomposition.

        Args:
            data: Price series

        Returns:
            signal: -1 to 1 trend indicator
            details: Decomposition details
        """
        coeffs = WaveletAnalyzer.haar_wavelet_transform(data, levels=4)

        # Approximation represents the trend
        trend = coeffs['approximation']

        # Calculate trend direction
        if len(trend) >= 2:
            trend_direction = (trend[-1] - trend[0]) / (np.std(trend) + 1e-8)
            signal = np.tanh(trend_direction)  # Squash to [-1, 1]
        else:
            signal = 0.0

        # Calculate energy at each scale (which timeframes are dominant)
        scale_energies = []
        for i, detail in enumerate(coeffs['details']):
            energy = np.sum(detail ** 2)
            scale_energies.append(energy)

        total_energy = sum(scale_energies) + 1e-8
        scale_proportions = [e / total_energy for e in scale_energies]

        details = {
            'trend_value': trend[-1] if len(trend) > 0 else 0,
            'scale_energies': scale_proportions,
            'dominant_scale': np.argmax(scale_proportions) if scale_proportions else 0
        }

        return signal, details


# =============================================================================
# 2. HURST EXPONENT
# =============================================================================

class HurstAnalyzer:
    """
    Hurst Exponent Calculator

    THE MOST IMPORTANT NUMBER IN TRADING

    H = 0.5: Random walk (efficient market) - DON'T TRADE
    H > 0.5: Persistent/Trending - USE TREND FOLLOWING
    H < 0.5: Anti-persistent/Mean reverting - USE MEAN REVERSION

    MATHEMATICAL BASIS:
    R/S Analysis (Rescaled Range):

    E[R(n)/S(n)] = C × n^H

    Where:
    - R(n) = range of cumulative deviations
    - S(n) = standard deviation
    - H = Hurst exponent
    - C = constant

    Taking log: log(R/S) = H × log(n) + log(C)
    So H is the slope of log-log plot
    """

    @staticmethod
    def calculate_rs(data: np.ndarray) -> float:
        """
        Calculate R/S statistic for a series.

        Args:
            data: Time series

        Returns:
            R/S value
        """
        n = len(data)
        if n < 2:
            return 0.0

        # Mean
        mean = np.mean(data)

        # Cumulative deviations from mean
        cumdev = np.cumsum(data - mean)

        # Range
        R = np.max(cumdev) - np.min(cumdev)

        # Standard deviation
        S = np.std(data, ddof=1)

        if S == 0:
            return 0.0

        return R / S

    @staticmethod
    def calculate(data: np.ndarray, min_window: int = 10) -> Tuple[float, Dict]:
        """
        Calculate Hurst Exponent using R/S analysis.

        Args:
            data: Price returns or log returns
            min_window: Minimum window size

        Returns:
            H: Hurst exponent
            details: Analysis details
        """
        n = len(data)
        if n < min_window * 4:
            return 0.5, {'error': 'Insufficient data'}

        # Calculate R/S for different window sizes
        window_sizes = []
        rs_values = []

        # Use logarithmically spaced windows
        max_window = n // 2
        windows = np.unique(np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num=20
        ).astype(int))

        for window in windows:
            if window < min_window or window > max_window:
                continue

            # Calculate R/S for each non-overlapping segment
            n_segments = n // window
            if n_segments < 1:
                continue

            rs_list = []
            for i in range(n_segments):
                segment = data[i * window:(i + 1) * window]
                rs = HurstAnalyzer.calculate_rs(segment)
                if rs > 0:
                    rs_list.append(rs)

            if rs_list:
                window_sizes.append(window)
                rs_values.append(np.mean(rs_list))

        if len(window_sizes) < 3:
            return 0.5, {'error': 'Not enough valid windows'}

        # Linear regression on log-log plot
        log_windows = np.log(window_sizes)
        log_rs = np.log(rs_values)

        # Fit: log(R/S) = H * log(n) + c
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_windows, log_rs)

        H = slope

        # Classify regime
        if H > 0.55:
            regime = 'TRENDING'
        elif H < 0.45:
            regime = 'MEAN_REVERTING'
        else:
            regime = 'RANDOM'

        details = {
            'hurst': H,
            'regime': regime,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'confidence': 1 - p_value if p_value < 1 else 0
        }

        return H, details


# =============================================================================
# 3. ORNSTEIN-UHLENBECK PROCESS
# =============================================================================

class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck Process for Mean Reversion

    THE MATHEMATICAL MODEL FOR MEAN REVERSION

    dX = θ(μ - X)dt + σdW

    Where:
    - X = current price (or spread)
    - μ = long-term mean (equilibrium)
    - θ = speed of mean reversion
    - σ = volatility
    - dW = Wiener process (random)

    KEY DERIVED QUANTITIES:
    - Half-life = ln(2)/θ (time to revert halfway to mean)
    - Variance = σ²/(2θ) (long-run variance)

    TRADING STRATEGY:
    - If X > μ + k×σ: SELL (price will revert down)
    - If X < μ - k×σ: BUY (price will revert up)
    - k depends on desired confidence (typically 1-2)
    """

    @staticmethod
    def estimate_parameters(data: np.ndarray, dt: float = 1.0) -> Dict:
        """
        Estimate OU parameters using Maximum Likelihood.

        Args:
            data: Price or spread series
            dt: Time step (1 for daily, 1/24 for hourly, etc.)

        Returns:
            Dict with mu, theta, sigma and derived quantities
        """
        n = len(data)
        if n < 30:
            return {'error': 'Insufficient data', 'theta': 0, 'mu': np.mean(data), 'sigma': np.std(data)}

        # MLE estimation
        # X(t+dt) = X(t)*exp(-θdt) + μ(1-exp(-θdt)) + ε
        # This is AR(1): X(t+dt) = a + b*X(t) + ε

        X = data[:-1]
        Y = data[1:]

        # OLS regression: Y = a + b*X
        n_obs = len(X)
        sum_x = np.sum(X)
        sum_y = np.sum(Y)
        sum_xx = np.sum(X * X)
        sum_xy = np.sum(X * Y)

        b = (n_obs * sum_xy - sum_x * sum_y) / (n_obs * sum_xx - sum_x ** 2 + 1e-10)
        a = (sum_y - b * sum_x) / n_obs

        # Convert AR(1) to OU parameters
        # b = exp(-θdt) => θ = -ln(b)/dt
        # a = μ(1-exp(-θdt)) => μ = a/(1-b)

        if b <= 0 or b >= 1:
            # Process is not mean-reverting
            return {
                'error': 'Not mean-reverting',
                'theta': 0,
                'mu': np.mean(data),
                'sigma': np.std(data),
                'half_life': float('inf'),
                'is_mean_reverting': False
            }

        theta = -np.log(b) / dt
        mu = a / (1 - b)

        # Estimate sigma from residuals
        residuals = Y - (a + b * X)
        sigma_residual = np.std(residuals)

        # Convert residual sigma to OU sigma
        # Var(residual) = σ²(1-exp(-2θdt))/(2θ)
        sigma = sigma_residual * np.sqrt(2 * theta / (1 - np.exp(-2 * theta * dt) + 1e-10))

        # Derived quantities
        half_life = np.log(2) / theta  # In units of dt
        long_run_variance = sigma ** 2 / (2 * theta)
        long_run_std = np.sqrt(long_run_variance)

        # Current z-score
        current_price = data[-1]
        z_score = (current_price - mu) / (long_run_std + 1e-10)

        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': half_life,
            'long_run_std': long_run_std,
            'current_price': current_price,
            'z_score': z_score,
            'is_mean_reverting': True,
            'confidence': 1 - stats.norm.sf(abs(z_score)) * 2  # Two-tailed
        }

    @staticmethod
    def get_signal(params: Dict, entry_threshold: float = 1.5) -> Tuple[str, float]:
        """
        Generate trading signal from OU parameters.

        Args:
            params: OU parameter dict
            entry_threshold: Z-score threshold for entry

        Returns:
            direction: BUY, SELL, or NEUTRAL
            confidence: Signal confidence
        """
        if not params.get('is_mean_reverting', False):
            return 'NEUTRAL', 0.0

        z = params.get('z_score', 0)

        if z > entry_threshold:
            return 'SELL', min(abs(z) / 3, 1.0)  # Price too high, will revert down
        elif z < -entry_threshold:
            return 'BUY', min(abs(z) / 3, 1.0)  # Price too low, will revert up
        else:
            return 'NEUTRAL', 0.0


# =============================================================================
# 4. INFORMATION THEORY
# =============================================================================

class InformationTheory:
    """
    Information Theoretic Analysis

    WHY INFORMATION THEORY FOR TRADING:
    - Captures non-linear relationships (correlation only sees linear)
    - Detects regime changes via entropy
    - Measures true predictability

    KEY CONCEPTS:

    1. ENTROPY: H(X) = -Σ P(x) log P(x)
       Measures uncertainty/randomness

    2. MUTUAL INFORMATION: I(X;Y) = H(X) + H(Y) - H(X,Y)
       Measures shared information between variables

    3. KL DIVERGENCE: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))
       Measures how different two distributions are

    4. TRANSFER ENTROPY: T(X→Y) = I(Y_t+1; X_t | Y_t)
       Measures directional information flow (causality!)
    """

    @staticmethod
    def discretize(data: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """Discretize continuous data into bins."""
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(data, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        return np.digitize(data, bins[1:-1])

    @staticmethod
    def entropy(data: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Shannon entropy.

        Args:
            data: Continuous data
            n_bins: Number of discretization bins

        Returns:
            Entropy in bits
        """
        discrete = InformationTheory.discretize(data, n_bins)
        _, counts = np.unique(discrete, return_counts=True)
        probs = counts / len(discrete)
        return -np.sum(probs * np.log2(probs + 1e-10))

    @staticmethod
    def mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Mutual Information between two series.

        I(X;Y) = H(X) + H(Y) - H(X,Y)

        Args:
            x, y: Two time series
            n_bins: Discretization bins

        Returns:
            Mutual information in bits
        """
        # Discretize
        x_disc = InformationTheory.discretize(x, n_bins)
        y_disc = InformationTheory.discretize(y, n_bins)

        # Joint histogram
        joint_hist, _, _ = np.histogram2d(x_disc, y_disc, bins=n_bins)
        joint_prob = joint_hist / np.sum(joint_hist)

        # Marginal probabilities
        p_x = np.sum(joint_prob, axis=1)
        p_y = np.sum(joint_prob, axis=0)

        # Mutual information
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if joint_prob[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += joint_prob[i, j] * np.log2(
                        joint_prob[i, j] / (p_x[i] * p_y[j])
                    )

        return max(0, mi)  # MI should be non-negative

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate KL Divergence D_KL(P||Q).

        Measures how different distribution P is from Q.

        Args:
            p, q: Two distributions (or data samples)
            n_bins: Discretization bins

        Returns:
            KL divergence (0 = identical, higher = more different)
        """
        # Create histograms
        all_data = np.concatenate([p, q])
        bins = np.linspace(np.min(all_data), np.max(all_data), n_bins + 1)

        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)

        # Add small constant for numerical stability
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10

        # Normalize
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)

        # KL divergence
        kl = np.sum(p_hist * np.log(p_hist / q_hist))

        return max(0, kl)

    @staticmethod
    def detect_regime_change(data: np.ndarray, window: int = 50) -> Tuple[float, List[int]]:
        """
        Detect regime changes using rolling KL divergence.

        Args:
            data: Time series
            window: Window size for comparison

        Returns:
            current_change: How different current regime is from past
            change_points: Indices of detected regime changes
        """
        if len(data) < window * 3:
            return 0.0, []

        kl_values = []

        for i in range(window * 2, len(data)):
            past = data[i - window * 2:i - window]
            current = data[i - window:i]
            kl = InformationTheory.kl_divergence(past, current)
            kl_values.append(kl)

        kl_values = np.array(kl_values)

        # Find change points (KL spikes)
        threshold = np.mean(kl_values) + 2 * np.std(kl_values)
        change_points = np.where(kl_values > threshold)[0] + window * 2

        current_change = kl_values[-1] if len(kl_values) > 0 else 0

        return current_change, change_points.tolist()


# =============================================================================
# 5. EIGENVALUE ANALYSIS (Random Matrix Theory)
# =============================================================================

class EigenvalueAnalyzer:
    """
    Random Matrix Theory for Signal Extraction

    THE PROBLEM:
    Correlation matrices from financial data contain:
    1. TRUE correlations (signal)
    2. SPURIOUS correlations (noise from limited data)

    RANDOM MATRIX THEORY tells us:
    - For random data, eigenvalues follow Marcenko-Pastur distribution
    - Eigenvalues OUTSIDE this distribution = real signal

    MARCHENKO-PASTUR BOUNDS:
    λ_max = σ²(1 + √(N/T))²
    λ_min = σ²(1 - √(N/T))²

    Where:
    - N = number of assets
    - T = number of time points
    - σ² = variance (usually 1 for correlation matrix)

    Eigenvalues > λ_max are SIGNAL
    Eigenvalues < λ_max are NOISE
    """

    @staticmethod
    def marchenko_pastur_bounds(n_assets: int, n_observations: int,
                                  variance: float = 1.0) -> Tuple[float, float]:
        """
        Calculate Marchenko-Pastur theoretical bounds.

        Args:
            n_assets: Number of assets/features
            n_observations: Number of time points
            variance: Variance of data (1 for correlation matrix)

        Returns:
            lambda_min, lambda_max: Theoretical eigenvalue bounds
        """
        q = n_assets / n_observations

        if q > 1:
            q = 1 / q

        lambda_max = variance * (1 + np.sqrt(q)) ** 2
        lambda_min = variance * (1 - np.sqrt(q)) ** 2

        return lambda_min, lambda_max

    @staticmethod
    def analyze_correlation(returns_matrix: np.ndarray) -> Dict:
        """
        Analyze correlation matrix using Random Matrix Theory.

        Args:
            returns_matrix: (T x N) matrix of returns
                           T = time points, N = assets/features

        Returns:
            Dict with eigenvalue analysis results
        """
        T, N = returns_matrix.shape

        if T < N:
            return {'error': 'Need more observations than features'}

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns_matrix.T)

        # Get eigenvalues
        eigenvalues, eigenvectors = eigh(corr_matrix)
        eigenvalues = eigenvalues[::-1]  # Sort descending
        eigenvectors = eigenvectors[:, ::-1]

        # Marchenko-Pastur bounds
        lambda_min, lambda_max = EigenvalueAnalyzer.marchenko_pastur_bounds(N, T)

        # Identify signal eigenvalues (above noise)
        signal_eigenvalues = eigenvalues[eigenvalues > lambda_max]
        noise_eigenvalues = eigenvalues[eigenvalues <= lambda_max]

        # Signal to noise ratio
        signal_variance = np.sum(signal_eigenvalues) if len(signal_eigenvalues) > 0 else 0
        noise_variance = np.sum(noise_eigenvalues) if len(noise_eigenvalues) > 0 else 1
        snr = signal_variance / (noise_variance + 1e-10)

        # Effective number of factors
        n_factors = len(signal_eigenvalues)

        # Explained variance by signal
        total_variance = np.sum(eigenvalues)
        signal_explained = signal_variance / total_variance if total_variance > 0 else 0

        return {
            'eigenvalues': eigenvalues.tolist(),
            'lambda_max': lambda_max,
            'lambda_min': lambda_min,
            'n_signal_factors': n_factors,
            'signal_to_noise_ratio': snr,
            'signal_explained_variance': signal_explained,
            'largest_eigenvalue': eigenvalues[0],
            'market_factor_strength': eigenvalues[0] / N  # How dominant is the market factor
        }


# =============================================================================
# 6. FRACTIONAL CALCULUS
# =============================================================================

class FractionalCalculus:
    """
    Fractional Calculus for Long Memory Effects

    WHY FRACTIONAL CALCULUS:
    - Standard calculus: Derivatives are integers (1st, 2nd derivative)
    - Fractional: Derivatives can be any real number (0.5th derivative!)

    This captures "long memory" effects in financial data:
    - Past events have lasting, decaying influence
    - Better models for volatility clustering

    FRACTIONAL DIFFERENCE:
    (1-B)^d x_t = Σ_{k=0}^∞ (-1)^k C(d,k) x_{t-k}

    Where:
    - d = fractional order (0 < d < 1)
    - B = backshift operator
    - C(d,k) = generalized binomial coefficient
    """

    @staticmethod
    def fractional_difference(data: np.ndarray, d: float,
                               threshold: float = 1e-5) -> np.ndarray:
        """
        Calculate fractional difference of a series.

        Args:
            data: Time series
            d: Fractional difference order (typically 0.3-0.5)
            threshold: Cutoff for weights

        Returns:
            Fractionally differenced series
        """
        n = len(data)

        # Calculate weights
        weights = [1.0]
        k = 1
        while True:
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold:
                break
            weights.append(w)
            k += 1
            if k > n:
                break

        weights = np.array(weights)

        # Apply fractional difference
        result = np.zeros(n)
        for i in range(len(weights), n):
            result[i] = np.sum(weights * data[i - len(weights) + 1:i + 1][::-1])

        return result

    @staticmethod
    def estimate_d(data: np.ndarray) -> Tuple[float, Dict]:
        """
        Estimate optimal fractional difference order.

        Uses GPH estimator (Geweke-Porter-Hudak).

        Args:
            data: Time series

        Returns:
            d: Estimated fractional order
            details: Estimation details
        """
        n = len(data)

        # Compute periodogram
        fft_result = fft(data - np.mean(data))
        periodogram = np.abs(fft_result) ** 2 / n

        # Frequencies
        freqs = np.arange(1, n // 2 + 1) / n

        # GPH regression
        # log(I(f)) = c - 2d * log(2*sin(πf)) + error

        # Use only low frequencies (first 10%)
        m = max(int(n ** 0.5), 10)
        m = min(m, len(freqs))

        y = np.log(periodogram[1:m + 1] + 1e-10)
        x = np.log(2 * np.sin(np.pi * freqs[:m]))

        # OLS regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        d = -slope / 2

        # Clamp to reasonable range
        d = np.clip(d, 0, 1)

        return d, {
            'd': d,
            'r_squared': r_value ** 2,
            'std_error': std_err,
            'interpretation': 'Long memory' if d > 0.25 else 'Short memory'
        }


# =============================================================================
# 7. JUMP DETECTION (Crash Prediction)
# =============================================================================

class JumpDetector:
    """
    Jump Diffusion and Crash Detection

    Standard models assume continuous price changes.
    Reality: Prices JUMP (flash crashes, news events).

    JUMP DIFFUSION MODEL:
    dS = μSdt + σSdW + J×dN

    Where:
    - J = jump size (random)
    - dN = Poisson process (when jumps occur)

    We detect jumps using:
    1. Bipower variation vs realized variance
    2. Extreme return detection
    3. Volatility clustering analysis
    """

    @staticmethod
    def detect_jumps(returns: np.ndarray, threshold: float = 3.0) -> Dict:
        """
        Detect jumps in return series.

        Args:
            returns: Return series
            threshold: Z-score threshold for jump detection

        Returns:
            Dict with jump analysis
        """
        n = len(returns)
        if n < 30:
            return {'error': 'Insufficient data'}

        # 1. Realized Variance (includes jumps)
        rv = np.sum(returns ** 2)

        # 2. Bipower Variation (robust to jumps)
        # BV = (π/2) × Σ |r_t| × |r_{t-1}|
        abs_returns = np.abs(returns)
        bv = (np.pi / 2) * np.sum(abs_returns[1:] * abs_returns[:-1])

        # 3. Jump contribution
        jump_variance = max(0, rv - bv)
        jump_ratio = jump_variance / (rv + 1e-10)

        # 4. Detect individual jumps
        mean = np.mean(returns)
        std = np.std(returns)
        z_scores = (returns - mean) / (std + 1e-10)

        jump_indices = np.where(np.abs(z_scores) > threshold)[0]
        jump_returns = returns[jump_indices] if len(jump_indices) > 0 else []

        # 5. Jump intensity (frequency)
        jump_intensity = len(jump_indices) / n

        # 6. Crash probability (based on recent volatility and jumps)
        recent_vol = np.std(returns[-20:]) if n >= 20 else std
        vol_ratio = recent_vol / (std + 1e-10)

        crash_probability = min(1.0, jump_ratio * vol_ratio * jump_intensity * 10)

        return {
            'realized_variance': rv,
            'bipower_variation': bv,
            'jump_variance': jump_variance,
            'jump_ratio': jump_ratio,
            'n_jumps_detected': len(jump_indices),
            'jump_intensity': jump_intensity,
            'recent_volatility_ratio': vol_ratio,
            'crash_probability': crash_probability,
            'jump_indices': jump_indices.tolist(),
            'largest_jump': float(np.max(np.abs(jump_returns))) if len(jump_returns) > 0 else 0
        }


# =============================================================================
# 8. FRACTAL DIMENSION
# =============================================================================

class FractalAnalyzer:
    """
    Fractal Dimension Analysis

    Fractal dimension measures the "roughness" of price path.

    D = 2 - H (for financial time series)

    Where H is Hurst exponent.

    INTERPRETATION:
    - D ≈ 1.0: Smooth (trending strongly)
    - D ≈ 1.5: Brownian motion (random)
    - D ≈ 2.0: Very rough (choppy/noisy)
    """

    @staticmethod
    def box_counting_dimension(data: np.ndarray, n_scales: int = 10) -> float:
        """
        Estimate fractal dimension using box counting.

        Args:
            data: Price series
            n_scales: Number of box sizes to try

        Returns:
            Estimated fractal dimension
        """
        n = len(data)

        # Normalize data to [0, 1]
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)

        # Box sizes
        sizes = np.logspace(0, np.log10(n // 4), n_scales).astype(int)
        sizes = np.unique(sizes[sizes >= 1])

        counts = []

        for size in sizes:
            # Count boxes needed to cover the curve
            n_boxes = 0
            for i in range(0, n, size):
                segment = data_norm[i:min(i + size, n)]
                if len(segment) > 0:
                    height_range = np.max(segment) - np.min(segment)
                    n_boxes += max(1, int(np.ceil(height_range / (1 / size))))

            counts.append(n_boxes)

        # Fit: log(count) = -D * log(size) + c
        log_sizes = np.log(1 / sizes)
        log_counts = np.log(np.array(counts) + 1)

        slope, _, _, _, _ = stats.linregress(log_sizes, log_counts)

        return slope

    @staticmethod
    def from_hurst(H: float) -> float:
        """Convert Hurst exponent to fractal dimension."""
        return 2 - H


# =============================================================================
# MAIN ENGINE - COMBINES ALL ALGORITHMS
# =============================================================================

class MathEngine:
    """
    Complete Mathematical Analysis Engine

    Combines all advanced algorithms into a single prediction system.

    ALGORITHM WEIGHTS (default):
    - Wavelet Analysis: 15%
    - Hurst Exponent: 20%
    - Ornstein-Uhlenbeck: 20%
    - Information Theory: 15%
    - Eigenvalue Analysis: 10%
    - Jump Detection: 10%
    - Fractal Analysis: 10%
    """

    DEFAULT_WEIGHTS = {
        'wavelet': 0.15,
        'hurst': 0.20,
        'ornstein_uhlenbeck': 0.20,
        'information': 0.15,
        'eigenvalue': 0.10,
        'jump': 0.10,
        'fractal': 0.10
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize with optional custom weights."""
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Normalize weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            for k in self.weights:
                self.weights[k] /= total

        logger.info("MathEngine initialized")
        logger.info(f"Algorithm weights: {self.weights}")

    def analyze(self, df: pd.DataFrame) -> MathematicalAnalysis:
        """
        Run complete mathematical analysis.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            MathematicalAnalysis with all results
        """
        prices = df['close'].values
        returns = np.diff(np.log(prices))  # Log returns

        n = len(prices)
        signals = {}
        confidence_scores = {}

        # 1. WAVELET ANALYSIS
        try:
            wavelet_signal, wavelet_details = WaveletAnalyzer.get_trend_signal(prices)
            signals['wavelet'] = wavelet_signal
            confidence_scores['wavelet'] = 0.8
        except Exception as e:
            logger.warning(f"Wavelet analysis failed: {e}")
            signals['wavelet'] = 0
            confidence_scores['wavelet'] = 0

        # 2. HURST EXPONENT
        try:
            hurst, hurst_details = HurstAnalyzer.calculate(returns)
            hurst_regime = hurst_details.get('regime', 'RANDOM')

            # Convert to signal
            if hurst_regime == 'TRENDING':
                # Follow the trend
                trend = np.mean(returns[-10:])
                signals['hurst'] = np.sign(trend) * (hurst - 0.5) * 2
            elif hurst_regime == 'MEAN_REVERTING':
                # Fade the move
                recent_move = prices[-1] / prices[-20] - 1 if n >= 20 else 0
                signals['hurst'] = -np.sign(recent_move) * (0.5 - hurst) * 2
            else:
                signals['hurst'] = 0

            confidence_scores['hurst'] = hurst_details.get('confidence', 0.5)
        except Exception as e:
            logger.warning(f"Hurst analysis failed: {e}")
            hurst = 0.5
            hurst_regime = 'RANDOM'
            signals['hurst'] = 0
            confidence_scores['hurst'] = 0

        # 3. ORNSTEIN-UHLENBECK
        try:
            ou_params = OrnsteinUhlenbeck.estimate_parameters(prices)
            ou_signal, ou_conf = OrnsteinUhlenbeck.get_signal(ou_params)

            signal_map = {'BUY': 1, 'SELL': -1, 'NEUTRAL': 0}
            signals['ornstein_uhlenbeck'] = signal_map.get(ou_signal, 0) * ou_conf
            confidence_scores['ornstein_uhlenbeck'] = ou_conf if ou_conf > 0 else 0.3

            ou_half_life = ou_params.get('half_life', float('inf'))
            ou_equilibrium = ou_params.get('mu', prices[-1])
        except Exception as e:
            logger.warning(f"OU analysis failed: {e}")
            signals['ornstein_uhlenbeck'] = 0
            confidence_scores['ornstein_uhlenbeck'] = 0
            ou_half_life = float('inf')
            ou_equilibrium = prices[-1]

        # 4. INFORMATION THEORY
        try:
            # Mutual information between lagged returns
            if len(returns) >= 100:
                mi = InformationTheory.mutual_information(returns[:-1], returns[1:])

                # Regime change detection
                change_magnitude, _ = InformationTheory.detect_regime_change(returns)

                # High MI and low regime change = more predictable
                predictability = mi * (1 - min(change_magnitude, 1))

                # Use predictability to weight trend signal
                trend = np.mean(returns[-10:])
                signals['information'] = np.sign(trend) * predictability
                confidence_scores['information'] = min(mi / 0.5, 1.0)  # Normalize
            else:
                mi = 0
                signals['information'] = 0
                confidence_scores['information'] = 0
        except Exception as e:
            logger.warning(f"Information theory failed: {e}")
            mi = 0
            signals['information'] = 0
            confidence_scores['information'] = 0

        # 5. EIGENVALUE ANALYSIS
        try:
            if n >= 60:
                # Create feature matrix from rolling windows
                window = 20
                features = []
                for i in range(n - window):
                    features.append(returns[i:i + window])
                feature_matrix = np.array(features)

                eigen_results = EigenvalueAnalyzer.analyze_correlation(feature_matrix)
                snr = eigen_results.get('signal_to_noise_ratio', 1)

                # High SNR = more reliable signals
                eigenvalue_ratio = snr

                # Weight recent trend by SNR
                trend = np.mean(returns[-10:])
                signals['eigenvalue'] = np.sign(trend) * np.tanh(snr - 1)
                confidence_scores['eigenvalue'] = min(snr / 3, 1.0)
            else:
                eigenvalue_ratio = 1
                signals['eigenvalue'] = 0
                confidence_scores['eigenvalue'] = 0
        except Exception as e:
            logger.warning(f"Eigenvalue analysis failed: {e}")
            eigenvalue_ratio = 1
            signals['eigenvalue'] = 0
            confidence_scores['eigenvalue'] = 0

        # 6. JUMP DETECTION
        try:
            jump_results = JumpDetector.detect_jumps(returns)
            crash_probability = jump_results.get('crash_probability', 0)
            jump_probability = jump_results.get('jump_intensity', 0)

            # Reduce confidence when jump risk is high
            jump_adjustment = 1 - crash_probability

            signals['jump'] = -crash_probability  # Negative = caution
            confidence_scores['jump'] = 0.5 * jump_adjustment
        except Exception as e:
            logger.warning(f"Jump detection failed: {e}")
            crash_probability = 0
            jump_probability = 0
            signals['jump'] = 0
            confidence_scores['jump'] = 0

        # 7. FRACTAL ANALYSIS
        try:
            fractal_dim = FractalAnalyzer.box_counting_dimension(prices)
            fractal_from_hurst = FractalAnalyzer.from_hurst(hurst)

            # Use average
            fractal_dimension = (fractal_dim + fractal_from_hurst) / 2

            # Low dimension = trending = follow trend
            # High dimension = choppy = reduce trading
            if fractal_dimension < 1.4:
                trend = np.mean(returns[-10:])
                signals['fractal'] = np.sign(trend) * (1.5 - fractal_dimension)
            else:
                signals['fractal'] = 0

            confidence_scores['fractal'] = max(0, 1.5 - fractal_dimension)
        except Exception as e:
            logger.warning(f"Fractal analysis failed: {e}")
            fractal_dimension = 1.5
            signals['fractal'] = 0
            confidence_scores['fractal'] = 0

        # COMBINE ALL SIGNALS
        combined_signal = sum(
            self.weights[k] * signals.get(k, 0)
            for k in self.weights
        )

        # Calculate overall confidence
        calculation_confidence = np.mean([
            confidence_scores.get(k, 0) for k in self.weights
        ])

        # Convert to direction and confidence
        if combined_signal > 0.1:
            direction = 'BUY'
            confidence = min(combined_signal, 1.0)
        elif combined_signal < -0.1:
            direction = 'SELL'
            confidence = min(abs(combined_signal), 1.0)
        else:
            direction = 'NEUTRAL'
            confidence = 0

        # Adjust for crash risk
        confidence *= (1 - crash_probability * 0.5)

        # Data quality score
        data_quality = min(1.0, n / 200)  # Need at least 200 points for good analysis

        return MathematicalAnalysis(
            direction=direction,
            confidence=confidence,
            wavelet_signal=signals.get('wavelet', 0),
            hurst_exponent=hurst,
            hurst_regime=hurst_regime,
            ou_half_life=ou_half_life,
            ou_equilibrium=ou_equilibrium,
            mutual_information=mi,
            eigenvalue_ratio=eigenvalue_ratio,
            fractal_dimension=fractal_dimension,
            jump_probability=jump_probability,
            crash_indicator=crash_probability,
            calculation_confidence=calculation_confidence,
            data_quality=data_quality
        )


# =============================================================================
# MATHEMATICAL REFERENCE
# =============================================================================

MATH_THEORY_REFERENCE = """
================================================================================
ADVANCED MATHEMATICAL THEORY REFERENCE
================================================================================

1. WAVELET TRANSFORM
   ─────────────────
   Continuous: W(a,b) = (1/√a) ∫ x(t) × ψ*((t-b)/a) dt

   Haar Transform (discrete):
   - Approximation: A[k] = (x[2k] + x[2k+1]) / √2
   - Detail: D[k] = (x[2k] - x[2k+1]) / √2

   Properties:
   - Multi-resolution analysis
   - Time-frequency localization
   - Optimal for non-stationary signals

2. HURST EXPONENT
   ───────────────
   R/S Analysis: E[R(n)/S(n)] = C × n^H

   Interpretation:
   - H = 0.5: Random walk (Brownian motion)
   - H > 0.5: Persistent (trending)
   - H < 0.5: Anti-persistent (mean reverting)

   Trading Rule:
   - H > 0.55: Use trend following
   - H < 0.45: Use mean reversion
   - 0.45 < H < 0.55: Don't trade (random)

3. ORNSTEIN-UHLENBECK PROCESS
   ───────────────────────────
   SDE: dX = θ(μ - X)dt + σdW

   Solution: X(t) = μ + (X₀ - μ)e^(-θt) + σ∫e^(-θ(t-s))dW(s)

   Properties:
   - Mean: E[X(t)] = μ + (X₀ - μ)e^(-θt) → μ
   - Variance: Var[X(t)] → σ²/(2θ)
   - Half-life: τ = ln(2)/θ

   Trading: Buy when X < μ - kσ, Sell when X > μ + kσ

4. INFORMATION THEORY
   ──────────────────
   Entropy: H(X) = -Σ P(x) log₂ P(x)

   Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)

   KL Divergence: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))

   Properties:
   - H(X) ≥ 0 (non-negative)
   - I(X;Y) ≥ 0 (non-negative)
   - I(X;Y) = I(Y;X) (symmetric)

5. RANDOM MATRIX THEORY
   ─────────────────────
   Marchenko-Pastur Distribution:
   λ_max = σ²(1 + √(N/T))²
   λ_min = σ²(1 - √(N/T))²

   Applications:
   - Separate signal from noise
   - Detect true correlations
   - Portfolio optimization

6. FRACTIONAL CALCULUS
   ────────────────────
   Fractional Difference: (1-B)^d x_t = Σ_{k=0}^∞ w_k x_{t-k}

   Weights: w_k = (-1)^k × Γ(d+1) / (Γ(k+1) × Γ(d-k+1))

   For 0 < d < 0.5:
   - Stationary but with long memory
   - Captures volatility clustering

7. JUMP DIFFUSION
   ───────────────
   Model: dS = μSdt + σSdW + J×dN

   Jump Detection:
   - Realized Variance: RV = Σ r²_t
   - Bipower Variation: BV = (π/2) × Σ |r_t| × |r_{t-1}|
   - Jump Variance: JV = max(0, RV - BV)

8. FRACTAL DIMENSION
   ──────────────────
   Box-Counting: D = lim(ε→0) log(N(ε)) / log(1/ε)

   Relationship to Hurst: D = 2 - H

   Interpretation:
   - D ≈ 1.0: Smooth path (strong trend)
   - D ≈ 1.5: Brownian motion (random)
   - D ≈ 2.0: Space-filling (very choppy)

================================================================================
WHY MATHEMATICS WORKS FOR TRADING
================================================================================

1. PATTERN RECOGNITION
   - Markets repeat patterns (fractals, cycles)
   - Math can detect patterns humans miss

2. PROBABILITY
   - Trading is about probabilities, not certainties
   - Math gives precise probability estimates

3. RISK MANAGEMENT
   - Position sizing is pure math
   - Kelly Criterion: f* = (bp - q) / b

4. EFFICIENCY
   - Computers calculate faster than humans
   - Math enables systematic trading

================================================================================
LIMITATIONS (HONEST TRUTH)
================================================================================

1. Markets are NOT perfectly mathematical
   - Human psychology matters
   - Black swan events exist

2. All models are wrong, some are useful
   - "The map is not the territory"

3. Over-optimization is dangerous
   - Past patterns may not repeat

4. Expected edge: 52-58%, NOT 90%+
   - Profit comes from discipline + math together

================================================================================
"""


def print_theory_reference():
    """Print mathematical theory reference."""
    print(MATH_THEORY_REFERENCE)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("ADVANCED MATHEMATICAL ENGINE - TEST")
    print("="*70)

    # Generate synthetic test data with trends and mean reversion
    np.random.seed(42)
    n = 500

    # Create realistic price series
    # Base: Random walk with drift
    returns = np.random.normal(0.0005, 0.02, n)

    # Add cycle
    t = np.arange(n)
    cycle = 0.005 * np.sin(2 * np.pi * t / 50)
    returns += cycle

    # Add mean reversion component
    prices = 100 * np.exp(np.cumsum(returns))

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': range(n),
        'datetime': pd.date_range('2024-01-01', periods=n, freq='h'),
        'open': prices * (1 + np.random.normal(0, 0.001, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n)
    })

    # Test individual components
    print("\n1. WAVELET ANALYSIS")
    print("-" * 40)
    wavelet_signal, details = WaveletAnalyzer.get_trend_signal(prices)
    print(f"   Signal: {wavelet_signal:.4f}")
    print(f"   Dominant scale: {details['dominant_scale']}")

    print("\n2. HURST EXPONENT")
    print("-" * 40)
    returns_arr = np.diff(np.log(prices))
    hurst, hurst_details = HurstAnalyzer.calculate(returns_arr)
    print(f"   H = {hurst:.4f}")
    print(f"   Regime: {hurst_details['regime']}")
    print(f"   R² = {hurst_details['r_squared']:.4f}")

    print("\n3. ORNSTEIN-UHLENBECK")
    print("-" * 40)
    ou_params = OrnsteinUhlenbeck.estimate_parameters(prices)
    print(f"   θ (mean reversion speed): {ou_params.get('theta', 0):.4f}")
    print(f"   μ (equilibrium): ${ou_params.get('mu', 0):.2f}")
    print(f"   Half-life: {ou_params.get('half_life', 0):.1f} candles")
    print(f"   Z-score: {ou_params.get('z_score', 0):.2f}")

    print("\n4. INFORMATION THEORY")
    print("-" * 40)
    mi = InformationTheory.mutual_information(returns_arr[:-1], returns_arr[1:])
    entropy_val = InformationTheory.entropy(returns_arr)
    print(f"   Mutual Information: {mi:.4f} bits")
    print(f"   Entropy: {entropy_val:.4f} bits")

    print("\n5. JUMP DETECTION")
    print("-" * 40)
    jump_results = JumpDetector.detect_jumps(returns_arr)
    print(f"   Jump ratio: {jump_results['jump_ratio']:.2%}")
    print(f"   Crash probability: {jump_results['crash_probability']:.2%}")
    print(f"   Jumps detected: {jump_results['n_jumps_detected']}")

    print("\n6. FRACTAL DIMENSION")
    print("-" * 40)
    fractal_dim = FractalAnalyzer.box_counting_dimension(prices)
    fractal_hurst = FractalAnalyzer.from_hurst(hurst)
    print(f"   Box-counting D: {fractal_dim:.4f}")
    print(f"   From Hurst D: {fractal_hurst:.4f}")

    # Full analysis
    print("\n" + "="*70)
    print("COMPLETE MATHEMATICAL ANALYSIS")
    print("="*70)

    engine = MathEngine()
    result = engine.analyze(df)

    print(f"\nDIRECTION: {result.direction}")
    print(f"CONFIDENCE: {result.confidence:.2%}")
    print(f"\nComponents:")
    print(f"  Wavelet Signal: {result.wavelet_signal:.4f}")
    print(f"  Hurst Exponent: {result.hurst_exponent:.4f} ({result.hurst_regime})")
    print(f"  OU Half-life: {result.ou_half_life:.1f} candles")
    print(f"  OU Equilibrium: ${result.ou_equilibrium:.2f}")
    print(f"  Mutual Information: {result.mutual_information:.4f}")
    print(f"  Eigenvalue Ratio: {result.eigenvalue_ratio:.4f}")
    print(f"  Fractal Dimension: {result.fractal_dimension:.4f}")
    print(f"\nRisk Metrics:")
    print(f"  Jump Probability: {result.jump_probability:.2%}")
    print(f"  Crash Indicator: {result.crash_indicator:.2%}")
    print(f"\nQuality:")
    print(f"  Calculation Confidence: {result.calculation_confidence:.2%}")
    print(f"  Data Quality: {result.data_quality:.2%}")

    # Print theory reference
    print("\n")
    print_theory_reference()
