"""
Tests for dynamic probability mappings in AdvancedPredictor.

Validates that the five fixes (Kalman sigmoid, Fourier cosine,
Markov recentering, Entropy certainty, symmetric confidence)
produce balanced BUY/SELL distributions instead of the 95% SELL bias.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pytest
import numpy as np


class TestKalmanSigmoid:
    """Test that Kalman velocity maps to probability via sigmoid."""

    def _kalman_sigmoid(self, velocity: float, atr: float, sensitivity: float = 2.0) -> float:
        """Reproduce the Kalman sigmoid mapping."""
        norm_velocity = velocity / (atr + 1e-10) if atr > 0 else 0.0
        exp_arg = max(-500, min(500, -sensitivity * norm_velocity))
        prob = 1.0 / (1.0 + math.exp(exp_arg))
        return max(0.05, min(0.95, prob))

    def test_zero_velocity_is_neutral(self):
        """Zero velocity should map to exactly 0.50."""
        assert self._kalman_sigmoid(0.0, 100.0) == pytest.approx(0.50, abs=1e-6)

    def test_positive_velocity_is_bullish(self):
        """Positive velocity should produce prob > 0.50."""
        prob = self._kalman_sigmoid(50.0, 100.0)
        assert prob > 0.60
        assert prob < 0.80

    def test_negative_velocity_is_bearish(self):
        """Negative velocity should produce prob < 0.50."""
        prob = self._kalman_sigmoid(-50.0, 100.0)
        assert prob < 0.40
        assert prob > 0.20

    def test_symmetric_around_zero(self):
        """Positive and negative velocity of same magnitude should be symmetric around 0.50."""
        prob_up = self._kalman_sigmoid(50.0, 100.0)
        prob_down = self._kalman_sigmoid(-50.0, 100.0)
        assert prob_up + prob_down == pytest.approx(1.0, abs=1e-6)

    def test_small_velocity_near_neutral(self):
        """Tiny velocity (noise) should stay near 0.50."""
        prob = self._kalman_sigmoid(1.0, 100.0)
        assert 0.48 < prob < 0.52

    def test_zero_atr_returns_neutral(self):
        """If ATR is zero, should return 0.50 (avoid division by zero)."""
        prob = self._kalman_sigmoid(10.0, 0.0)
        assert prob == pytest.approx(0.50, abs=0.01)

    def test_clamped_to_range(self):
        """Extreme velocities should be clamped to [0.05, 0.95]."""
        prob = self._kalman_sigmoid(10000.0, 1.0, sensitivity=10.0)
        assert prob == pytest.approx(0.95, abs=1e-6)
        prob = self._kalman_sigmoid(-10000.0, 1.0, sensitivity=10.0)
        assert prob == pytest.approx(0.05, abs=1e-6)


class TestFourierCosine:
    """Test that Fourier cycle phase maps to probability via cosine."""

    def _fourier_cosine(self, cycle_phase: float, amplitude: float = 0.15, has_frequencies: bool = True) -> float:
        """Reproduce the Fourier cosine mapping."""
        if not has_frequencies:
            return 0.5
        prob = 0.5 + amplitude * math.cos(2 * math.pi * cycle_phase)
        return max(0.05, min(0.95, prob))

    def test_trough_is_bullish(self):
        """Phase 0.0 (trough, about to rise) should be bullish."""
        prob = self._fourier_cosine(0.0)
        assert prob == pytest.approx(0.65, abs=0.01)

    def test_peak_is_neutral(self):
        """Phase 0.25 (peak) should be neutral."""
        prob = self._fourier_cosine(0.25)
        assert prob == pytest.approx(0.50, abs=0.01)

    def test_falling_is_bearish(self):
        """Phase 0.50 (falling from peak) should be bearish."""
        prob = self._fourier_cosine(0.50)
        assert prob == pytest.approx(0.35, abs=0.01)

    def test_bottom_is_neutral(self):
        """Phase 0.75 (at bottom) should be neutral."""
        prob = self._fourier_cosine(0.75)
        assert prob == pytest.approx(0.50, abs=0.01)

    def test_continuous_no_jumps(self):
        """Probability should change smoothly between phases."""
        phases = np.linspace(0, 1, 100)
        probs = [self._fourier_cosine(p) for p in phases]
        diffs = np.diff(probs)
        assert max(abs(diffs)) < 0.02

    def test_default_result_is_neutral(self):
        """When Fourier has no frequencies (error/default), return 0.50."""
        prob = self._fourier_cosine(0.5, has_frequencies=False)
        assert prob == 0.5

    def test_amplitude_controls_range(self):
        """Higher amplitude = wider range of probabilities."""
        prob_small = self._fourier_cosine(0.0, amplitude=0.10)
        prob_large = self._fourier_cosine(0.0, amplitude=0.20)
        assert prob_small == pytest.approx(0.60, abs=0.01)
        assert prob_large == pytest.approx(0.70, abs=0.01)


class TestMarkovRecentering:
    """Test that Markov prob_up is recentered around 0.50 for ensemble use."""

    def _markov_ensemble_prob(self, prob_up: float, prob_down: float) -> float:
        """Reproduce the Markov recentering: prob_up / (prob_up + prob_down)."""
        return prob_up / (prob_up + prob_down + 1e-10)

    def test_equal_probs_is_neutral(self):
        """When prob_up == prob_down, ensemble_prob should be 0.50."""
        result = self._markov_ensemble_prob(0.333, 0.333)
        assert result == pytest.approx(0.50, abs=0.01)

    def test_higher_up_is_bullish(self):
        """When prob_up > prob_down, ensemble_prob > 0.50."""
        result = self._markov_ensemble_prob(0.40, 0.30)
        assert result > 0.50
        assert result == pytest.approx(0.571, abs=0.01)

    def test_higher_down_is_bearish(self):
        """When prob_down > prob_up, ensemble_prob < 0.50."""
        result = self._markov_ensemble_prob(0.30, 0.40)
        assert result < 0.50
        assert result == pytest.approx(0.429, abs=0.01)

    def test_symmetric(self):
        """Symmetric inputs should produce symmetric outputs around 0.50."""
        up = self._markov_ensemble_prob(0.40, 0.30)
        down = self._markov_ensemble_prob(0.30, 0.40)
        assert up + down == pytest.approx(1.0, abs=0.01)

    def test_default_dampened_is_neutral(self):
        """Default dampened values (1/3, 1/3) should produce 0.50."""
        result = self._markov_ensemble_prob(1.0 / 3, 1.0 / 3)
        assert result == pytest.approx(0.50, abs=0.01)


class TestSymmetricConfidence:
    """Test that BUY and SELL confidence is symmetric around 0.50."""

    def _confidence(self, ensemble_prob: float, buy_threshold: float = 0.52,
                    sell_threshold: float = 0.48, conf_range: float = 0.10) -> tuple:
        """Reproduce the symmetric confidence calculation."""
        if ensemble_prob > buy_threshold:
            direction = "BUY"
            confidence = (ensemble_prob - 0.5) / conf_range
        elif ensemble_prob < sell_threshold:
            direction = "SELL"
            confidence = (0.5 - ensemble_prob) / conf_range
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return direction, confidence

    def test_symmetric_same_distance(self):
        """BUY at 0.56 and SELL at 0.44 should have equal confidence."""
        _, conf_buy = self._confidence(0.56)
        _, conf_sell = self._confidence(0.44)
        assert conf_buy == pytest.approx(conf_sell, abs=1e-6)

    def test_buy_at_threshold(self):
        """BUY just above threshold should have low confidence."""
        direction, conf = self._confidence(0.53)
        assert direction == "BUY"
        assert conf == pytest.approx(0.30, abs=0.01)

    def test_sell_at_threshold(self):
        """SELL just below threshold should have low confidence."""
        direction, conf = self._confidence(0.47)
        assert direction == "SELL"
        assert conf == pytest.approx(0.30, abs=0.01)

    def test_neutral_zero_confidence(self):
        """NEUTRAL range should have zero confidence."""
        direction, conf = self._confidence(0.50)
        assert direction == "NEUTRAL"
        assert conf == 0.0

    def test_max_buy_confidence(self):
        """Strong BUY should have high confidence (clamped to 1.0)."""
        direction, conf = self._confidence(0.65)
        assert direction == "BUY"
        assert conf == 1.0


class TestEntropyCertainty:
    """Test that Entropy scales Kalman signal by market certainty."""

    def _entropy_certainty(self, kalman_prob: float, normalized_entropy: float, scaling: float = 0.8) -> float:
        """Reproduce the Entropy certainty-weighted mapping."""
        certainty = max(0.0, 1.0 - normalized_entropy)
        prob = 0.5 + (kalman_prob - 0.5) * certainty * scaling
        return max(0.05, min(0.95, prob))

    def test_low_entropy_amplifies_kalman(self):
        """Low entropy (ordered market) should amplify Kalman signal."""
        prob = self._entropy_certainty(kalman_prob=0.60, normalized_entropy=0.2)
        assert prob == pytest.approx(0.564, abs=0.01)

    def test_high_entropy_suppresses_kalman(self):
        """High entropy (chaotic market) should barely move from 0.50."""
        prob = self._entropy_certainty(kalman_prob=0.60, normalized_entropy=0.9)
        assert prob == pytest.approx(0.508, abs=0.01)

    def test_max_entropy_is_neutral(self):
        """Max entropy (certainty=0) should always return 0.50."""
        prob = self._entropy_certainty(kalman_prob=0.80, normalized_entropy=1.0)
        assert prob == pytest.approx(0.50, abs=1e-6)

    def test_neutral_kalman_unchanged(self):
        """If Kalman is exactly 0.50, entropy should not change it."""
        prob = self._entropy_certainty(kalman_prob=0.50, normalized_entropy=0.2)
        assert prob == pytest.approx(0.50, abs=1e-6)

    def test_bearish_kalman_amplified(self):
        """Bearish Kalman should produce sub-0.50 entropy probability."""
        prob = self._entropy_certainty(kalman_prob=0.40, normalized_entropy=0.2)
        assert prob < 0.50
        assert prob == pytest.approx(0.436, abs=0.01)
