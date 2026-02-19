# Dynamic Probability Mapping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 95% SELL signal bias by replacing hardcoded probability mappings with dynamic, continuous mathematical functions in the prediction ensemble.

**Architecture:** Five targeted fixes in `src/advanced_predictor.py` — Kalman sigmoid mapping, Fourier cosine mapping, Markov recentering, Entropy certainty scaling, and symmetric confidence calculation. Each fix is independent (except Entropy depends on Kalman). All new parameters are config-driven with backward-compatible defaults.

**Tech Stack:** Python 3, NumPy, math (stdlib), pytest

**Design doc:** `docs/plans/2026-02-19-dynamic-probability-mapping-design.md`

---

### Task 1: Add `math` import and dynamic config parameters to AdvancedPredictor.__init__

**Files:**
- Modify: `src/advanced_predictor.py:17-25` (imports)
- Modify: `src/advanced_predictor.py:790-801` (init config loading)

**Step 1: Add `math` import**

At line 17, `import math` is needed for `math.exp` and `math.cos`. Add it after the existing imports:

```python
import math
```

Add after `import logging` (line 17), before `import threading`.

**Step 2: Add dynamic probability config parameters**

In `__init__`, after the existing ensemble config block (after line 803), add:

```python
        # Dynamic probability config
        fourier_cfg = pred_config.get('fourier', {})
        self.fourier_dynamic = fourier_cfg.get('dynamic_prob', True)
        self.fourier_amplitude = fourier_cfg.get('amplitude', 0.15)

        kalman_cfg = pred_config.get('kalman', {})
        self.kalman_dynamic = kalman_cfg.get('dynamic_prob', True)
        self.kalman_sigmoid_sensitivity = kalman_cfg.get('sigmoid_sensitivity', 2.0)

        entropy_cfg = pred_config.get('entropy', {})
        self.entropy_dynamic = entropy_cfg.get('dynamic_prob', True)
        self.entropy_certainty_scaling = entropy_cfg.get('certainty_scaling', 0.8)

        ens_cfg_sym = pred_config.get('ensemble', {})
        self.symmetric_confidence = ens_cfg_sym.get('symmetric_confidence', True)
```

Note: `ens_cfg` already exists at line 795. Reuse it or use a separate reference for the new key. Since `symmetric_confidence` is under `ensemble`, we can add it to the existing `ens_cfg` block:

```python
        self.symmetric_confidence = ens_cfg.get('symmetric_confidence', True)
```

Add this single line after line 803 (`self.mc_clamp_max = ...`).

**Step 3: Commit**

```bash
git add src/advanced_predictor.py
git commit -m "feat: add dynamic probability config params to AdvancedPredictor"
```

---

### Task 2: Write tests for Kalman sigmoid probability mapping

**Files:**
- Create: `tests/test_dynamic_probabilities.py`

**Step 1: Write the failing tests**

```python
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
        prob = 1.0 / (1.0 + math.exp(-sensitivity * norm_velocity))
        return max(0.05, min(0.95, prob))

    def test_zero_velocity_is_neutral(self):
        """Zero velocity should map to exactly 0.50."""
        assert self._kalman_sigmoid(0.0, 100.0) == pytest.approx(0.50, abs=1e-6)

    def test_positive_velocity_is_bullish(self):
        """Positive velocity should produce prob > 0.50."""
        prob = self._kalman_sigmoid(50.0, 100.0)  # 0.5 ATR
        assert prob > 0.60
        assert prob < 0.80

    def test_negative_velocity_is_bearish(self):
        """Negative velocity should produce prob < 0.50."""
        prob = self._kalman_sigmoid(-50.0, 100.0)  # -0.5 ATR
        assert prob < 0.40
        assert prob > 0.20

    def test_symmetric_around_zero(self):
        """Positive and negative velocity of same magnitude should be symmetric around 0.50."""
        prob_up = self._kalman_sigmoid(50.0, 100.0)
        prob_down = self._kalman_sigmoid(-50.0, 100.0)
        assert prob_up + prob_down == pytest.approx(1.0, abs=1e-6)

    def test_small_velocity_near_neutral(self):
        """Tiny velocity (noise) should stay near 0.50."""
        prob = self._kalman_sigmoid(1.0, 100.0)  # 0.01 ATR
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
```

**Step 2: Run test to verify it fails (tests exist but code not yet changed)**

```bash
pytest tests/test_dynamic_probabilities.py::TestKalmanSigmoid -v
```

Expected: PASS (these test the reference implementation, not the actual code yet)

**Step 3: Commit**

```bash
git add tests/test_dynamic_probabilities.py
git commit -m "test: add Kalman sigmoid probability mapping tests"
```

---

### Task 3: Implement Kalman sigmoid in predict()

**Files:**
- Modify: `src/advanced_predictor.py:1097-1103`

**Step 1: Replace the Kalman probability mapping**

Replace lines 1097-1103:

```python
        # Kalman contribution — wider range for stronger signals
        kalman_prob = self.kalman.prob_sideways
        if kalman_result['trend'] == 'UP':
            kalman_prob = self.kalman.prob_up
        elif kalman_result['trend'] == 'DOWN':
            kalman_prob = self.kalman.prob_down
        prob_scores.append(('kalman', kalman_prob))
```

With:

```python
        # Kalman contribution — dynamic sigmoid on velocity
        if self.kalman_dynamic:
            velocity = kalman_result.get('velocity', 0.0)
            norm_velocity = velocity / (atr + 1e-10) if atr > 0 else 0.0
            kalman_prob = 1.0 / (1.0 + math.exp(-self.kalman_sigmoid_sensitivity * norm_velocity))
            kalman_prob = max(0.05, min(0.95, kalman_prob))
        else:
            kalman_prob = self.kalman.prob_sideways
            if kalman_result['trend'] == 'UP':
                kalman_prob = self.kalman.prob_up
            elif kalman_result['trend'] == 'DOWN':
                kalman_prob = self.kalman.prob_down
        prob_scores.append(('kalman', kalman_prob))
```

**Step 2: Run tests**

```bash
pytest tests/test_dynamic_probabilities.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add src/advanced_predictor.py
git commit -m "feat: implement Kalman sigmoid velocity-to-probability mapping"
```

---

### Task 4: Write tests for Fourier cosine mapping

**Files:**
- Modify: `tests/test_dynamic_probabilities.py`

**Step 1: Add Fourier tests to the test file**

```python
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
        # Max step should be small (no jumps)
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
```

**Step 2: Run tests**

```bash
pytest tests/test_dynamic_probabilities.py::TestFourierCosine -v
```

Expected: PASS (tests the reference math)

**Step 3: Commit**

```bash
git add tests/test_dynamic_probabilities.py
git commit -m "test: add Fourier cosine probability mapping tests"
```

---

### Task 5: Implement Fourier cosine in predict()

**Files:**
- Modify: `src/advanced_predictor.py:1089-1095`

**Step 1: Replace the Fourier probability mapping**

Replace lines 1089-1095:

```python
        # Fourier contribution — wider range for stronger signals
        fourier_prob = self.fourier.prob_neutral
        if fourier_result['signal'] == 'BULLISH':
            fourier_prob = self.fourier.prob_bullish
        elif fourier_result['signal'] == 'BEARISH':
            fourier_prob = self.fourier.prob_bearish
        prob_scores.append(('fourier', fourier_prob))
```

With:

```python
        # Fourier contribution — dynamic cosine on cycle phase
        if self.fourier_dynamic:
            cycle_phase = fourier_result.get('cycle_phase', 0.5)
            if not fourier_result.get('dominant_frequencies'):
                fourier_prob = 0.5
            else:
                fourier_prob = 0.5 + self.fourier_amplitude * math.cos(2 * math.pi * cycle_phase)
                fourier_prob = max(0.05, min(0.95, fourier_prob))
        else:
            fourier_prob = self.fourier.prob_neutral
            if fourier_result['signal'] == 'BULLISH':
                fourier_prob = self.fourier.prob_bullish
            elif fourier_result['signal'] == 'BEARISH':
                fourier_prob = self.fourier.prob_bearish
        prob_scores.append(('fourier', fourier_prob))
```

**Step 2: Run tests**

```bash
pytest tests/test_dynamic_probabilities.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add src/advanced_predictor.py
git commit -m "feat: implement Fourier cosine phase-to-probability mapping"
```

---

### Task 6: Write tests for Markov recentering and implement

**Files:**
- Modify: `tests/test_dynamic_probabilities.py`
- Modify: `src/advanced_predictor.py:485-496` (MarkovChain.analyze return)
- Modify: `src/advanced_predictor.py:549-557` (MarkovChain._default_result)
- Modify: `src/advanced_predictor.py:1105-1107` (ensemble markov_prob line)

**Step 1: Write Markov recentering tests**

Add to `tests/test_dynamic_probabilities.py`:

```python
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
```

**Step 2: Run tests to verify they pass (reference math)**

```bash
pytest tests/test_dynamic_probabilities.py::TestMarkovRecentering -v
```

**Step 3: Add `ensemble_prob` to MarkovChain.analyze() return**

In `src/advanced_predictor.py`, after line 484 (after computing dampened probs), add:

```python
            # Binary ensemble probability (centered at 0.50)
            ensemble_prob_val = prob_up / (prob_up + prob_down + 1e-10)
```

Then modify the return dict at lines 489-496 to include it:

```python
            return {
                'current_state': current_state,
                'prob_up': float(prob_up),
                'prob_down': float(prob_down),
                'prob_neutral': float(prob_neutral),
                'ensemble_prob': float(ensemble_prob_val),
                'transition_matrix': transition_matrix.tolist(),
                'steady_state': steady_state.tolist()
            }
```

Also update `_default_result` at lines 549-557 to include `ensemble_prob`:

```python
    def _default_result(self) -> Dict:
        return {
            'current_state': 'NEUTRAL',
            'prob_up': 0.33,
            'prob_down': 0.33,
            'prob_neutral': 0.34,
            'ensemble_prob': 0.50,
            'transition_matrix': [[0.33, 0.34, 0.33]] * 3,
            'steady_state': [0.33, 0.34, 0.33]
        }
```

**Step 4: Update ensemble calculation to use ensemble_prob**

Change line 1106 from:

```python
        markov_prob = markov_result['prob_up']
```

To:

```python
        markov_prob = markov_result.get('ensemble_prob', markov_result['prob_up'])
```

**Step 5: Run all tests**

```bash
pytest tests/test_dynamic_probabilities.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/advanced_predictor.py tests/test_dynamic_probabilities.py
git commit -m "feat: recenter Markov prob_up for ensemble use (fix bearish bias)"
```

---

### Task 7: Write tests for Entropy certainty scaling and implement

**Files:**
- Modify: `tests/test_dynamic_probabilities.py`
- Modify: `src/advanced_predictor.py:1109-1118`

**Step 1: Write Entropy tests**

Add to `tests/test_dynamic_probabilities.py`:

```python
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
        # certainty = 0.8, deviation = 0.10 * 0.8 * 0.8 = 0.064
        assert prob == pytest.approx(0.564, abs=0.01)

    def test_high_entropy_suppresses_kalman(self):
        """High entropy (chaotic market) should barely move from 0.50."""
        prob = self._entropy_certainty(kalman_prob=0.60, normalized_entropy=0.9)
        # certainty = 0.1, deviation = 0.10 * 0.1 * 0.8 = 0.008
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
        # certainty = 0.8, deviation = -0.10 * 0.8 * 0.8 = -0.064
        assert prob == pytest.approx(0.436, abs=0.01)
```

**Step 2: Run tests**

```bash
pytest tests/test_dynamic_probabilities.py::TestEntropyCertainty -v
```

Expected: PASS (reference math)

**Step 3: Replace Entropy probability mapping**

Replace lines 1109-1118:

```python
        # Entropy contribution — use in all regimes, not just TRENDING
        entropy_val = entropy_result.get('entropy', 0.5)
        entropy_prob = 0.5
        if entropy_result['regime'] == 'TRENDING':
            entropy_prob = self.entropy.prob_trending_up if kalman_result['trend'] == 'UP' else self.entropy.prob_trending_down
        elif entropy_result['regime'] in ('NORMAL', 'CHOPPY'):
            # Low entropy = more predictable, lean with Kalman trend
            if entropy_val < self.entropy.low_entropy_threshold:
                entropy_prob = self.entropy.prob_normal_up if kalman_result['trend'] == 'UP' else self.entropy.prob_normal_down
        # VOLATILE: keep at 0.5 (genuinely uncertain)
        prob_scores.append(('entropy', entropy_prob))
```

With:

```python
        # Entropy contribution — certainty-weighted scaling of Kalman
        if self.entropy_dynamic:
            normalized_entropy = entropy_result.get('normalized_entropy', 0.5)
            certainty = max(0.0, 1.0 - normalized_entropy)
            entropy_prob = 0.5 + (kalman_prob - 0.5) * certainty * self.entropy_certainty_scaling
            entropy_prob = max(0.05, min(0.95, entropy_prob))
        else:
            entropy_val = entropy_result.get('entropy', 0.5)
            entropy_prob = 0.5
            if entropy_result['regime'] == 'TRENDING':
                entropy_prob = self.entropy.prob_trending_up if kalman_result['trend'] == 'UP' else self.entropy.prob_trending_down
            elif entropy_result['regime'] in ('NORMAL', 'CHOPPY'):
                if entropy_val < self.entropy.low_entropy_threshold:
                    entropy_prob = self.entropy.prob_normal_up if kalman_result['trend'] == 'UP' else self.entropy.prob_normal_down
        prob_scores.append(('entropy', entropy_prob))
```

**Note:** This uses `kalman_prob` which was computed in the Kalman block above. The code order already has Kalman before Entropy, so this dependency is satisfied.

**Step 4: Run all tests**

```bash
pytest tests/test_dynamic_probabilities.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/advanced_predictor.py tests/test_dynamic_probabilities.py
git commit -m "feat: implement Entropy certainty-weighted Kalman scaling"
```

---

### Task 8: Write tests for symmetric confidence and implement

**Files:**
- Modify: `tests/test_dynamic_probabilities.py`
- Modify: `src/advanced_predictor.py:1147-1161`

**Step 1: Write symmetric confidence tests**

Add to `tests/test_dynamic_probabilities.py`:

```python
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
```

**Step 2: Run tests**

```bash
pytest tests/test_dynamic_probabilities.py::TestSymmetricConfidence -v
```

Expected: PASS

**Step 3: Implement symmetric confidence**

Replace lines 1147-1161:

```python
        # Determine direction and confidence using calibrated range mapping
        # Maps realistic ensemble range [floor, ceiling] → [0%, 100%]
        conf_range = self.confidence_ceiling - self.confidence_floor
        if ensemble_prob > self.buy_threshold:
            direction = "BUY"
            confidence = (ensemble_prob - self.confidence_floor) / conf_range if conf_range > 0 else 0.0
        elif ensemble_prob < self.sell_threshold:
            direction = "SELL"
            # Mirror: sell_prob = 1 - ensemble_prob, then same calibration
            sell_prob = 1.0 - ensemble_prob
            confidence = (sell_prob - self.confidence_floor) / conf_range if conf_range > 0 else 0.0
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
```

With:

```python
        # Determine direction and confidence
        conf_range = self.confidence_ceiling - self.confidence_floor
        if self.symmetric_confidence:
            # Symmetric: confidence = distance from 0.50, same for BUY and SELL
            if ensemble_prob > self.buy_threshold:
                direction = "BUY"
                confidence = (ensemble_prob - 0.5) / conf_range if conf_range > 0 else 0.0
            elif ensemble_prob < self.sell_threshold:
                direction = "SELL"
                confidence = (0.5 - ensemble_prob) / conf_range if conf_range > 0 else 0.0
            else:
                direction = "NEUTRAL"
                confidence = 0.0
        else:
            # Legacy: asymmetric mirror calculation
            if ensemble_prob > self.buy_threshold:
                direction = "BUY"
                confidence = (ensemble_prob - self.confidence_floor) / conf_range if conf_range > 0 else 0.0
            elif ensemble_prob < self.sell_threshold:
                direction = "SELL"
                sell_prob = 1.0 - ensemble_prob
                confidence = (sell_prob - self.confidence_floor) / conf_range if conf_range > 0 else 0.0
            else:
                direction = "NEUTRAL"
                confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
```

**Step 4: Run all tests**

```bash
pytest tests/test_dynamic_probabilities.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/advanced_predictor.py tests/test_dynamic_probabilities.py
git commit -m "feat: implement symmetric confidence calculation for BUY/SELL"
```

---

### Task 9: Add config values to config.yaml

**Files:**
- Modify: `config.yaml:137-155` (fourier, kalman, entropy sections)
- Modify: `config.yaml:187-195` (ensemble section)

**Step 1: Add dynamic_prob flags and params**

Under `prediction.fourier` (after line 146), add:

```yaml
    dynamic_prob: true
    amplitude: 0.15
```

Under `prediction.kalman` (after line 155), add:

```yaml
    dynamic_prob: true
    sigmoid_sensitivity: 2.0
```

Under `prediction.entropy` (after line 168), add:

```yaml
    dynamic_prob: true
    certainty_scaling: 0.8
```

Under `prediction.ensemble` (after line 195), add:

```yaml
    symmetric_confidence: true
```

**Step 2: Commit**

```bash
git add config.yaml
git commit -m "feat: add dynamic probability config flags to config.yaml"
```

---

### Task 10: Integration test — ensemble produces balanced output

**Files:**
- Modify: `tests/test_dynamic_probabilities.py`

**Step 1: Write integration test**

Add to `tests/test_dynamic_probabilities.py`:

```python
class TestEnsembleIntegration:
    """Integration test: full ensemble with dynamic probabilities produces balanced output."""

    def test_neutral_inputs_produce_neutral_ensemble(self):
        """When all algorithms output neutral, ensemble should be ~0.50."""
        weights = {'lstm': 0.35, 'fourier': 0.15, 'kalman': 0.20, 'markov': 0.15, 'entropy': 0.10, 'monte_carlo': 0.05}
        prob_scores = [
            ('lstm', 0.50),
            ('fourier', 0.50),     # phase=0.25 (peak) → cos(pi/2) ≈ 0 → 0.50
            ('kalman', 0.50),      # velocity=0 → sigmoid(0) = 0.50
            ('markov', 0.50),      # equal dampened → ensemble_prob = 0.50
            ('entropy', 0.50),     # kalman=0.50 → no amplification → 0.50
            ('monte_carlo', 0.50),
        ]
        ensemble = sum(weights.get(name, 0) * prob for name, prob in prob_scores)
        assert ensemble == pytest.approx(0.50, abs=0.01)

    def test_mild_downtrend_is_neutral_not_sell(self):
        """Mild downtrend should NOT produce SELL — should stay NEUTRAL."""
        # Kalman velocity = -0.1 ATR → sigmoid(-0.2) ≈ 0.45
        kalman_prob = 1.0 / (1.0 + math.exp(-2.0 * (-0.1)))
        # Entropy amplifies slightly: certainty=0.5 (entropy=0.5)
        entropy_prob = 0.5 + (kalman_prob - 0.5) * 0.5 * 0.8
        # Fourier neutral (phase=0.25)
        fourier_prob = 0.50
        # Markov neutral
        markov_prob = 0.50

        weights = {'lstm': 0.35, 'fourier': 0.15, 'kalman': 0.20, 'markov': 0.15, 'entropy': 0.10, 'monte_carlo': 0.05}
        ensemble = (weights['lstm'] * 0.50 + weights['fourier'] * fourier_prob +
                    weights['kalman'] * kalman_prob + weights['markov'] * markov_prob +
                    weights['entropy'] * entropy_prob + weights['monte_carlo'] * 0.50)

        # Should be in neutral zone [0.48, 0.52], NOT sell
        assert 0.47 < ensemble < 0.52, f"Mild downtrend ensemble={ensemble:.4f}, expected NEUTRAL zone"

    def test_strong_downtrend_produces_sell(self):
        """Strong downtrend SHOULD produce SELL signal."""
        kalman_prob = 1.0 / (1.0 + math.exp(-2.0 * (-0.5)))  # ~0.27
        entropy_prob = 0.5 + (kalman_prob - 0.5) * 0.8 * 0.8  # amplified
        markov_prob = 0.40  # slight bearish Markov

        weights = {'lstm': 0.35, 'fourier': 0.15, 'kalman': 0.20, 'markov': 0.15, 'entropy': 0.10, 'monte_carlo': 0.05}
        ensemble = (weights['lstm'] * 0.50 + weights['fourier'] * 0.40 +
                    weights['kalman'] * kalman_prob + weights['markov'] * markov_prob +
                    weights['entropy'] * entropy_prob + weights['monte_carlo'] * 0.48)

        assert ensemble < 0.48, f"Strong downtrend ensemble={ensemble:.4f}, expected < 0.48 (SELL)"

    def test_strong_uptrend_produces_buy(self):
        """Strong uptrend SHOULD produce BUY signal."""
        kalman_prob = 1.0 / (1.0 + math.exp(-2.0 * 0.5))  # ~0.73
        entropy_prob = 0.5 + (kalman_prob - 0.5) * 0.8 * 0.8
        markov_prob = 0.60

        weights = {'lstm': 0.35, 'fourier': 0.15, 'kalman': 0.20, 'markov': 0.15, 'entropy': 0.10, 'monte_carlo': 0.05}
        ensemble = (weights['lstm'] * 0.50 + weights['fourier'] * 0.60 +
                    weights['kalman'] * kalman_prob + weights['markov'] * markov_prob +
                    weights['entropy'] * entropy_prob + weights['monte_carlo'] * 0.52)

        assert ensemble > 0.52, f"Strong uptrend ensemble={ensemble:.4f}, expected > 0.52 (BUY)"
```

**Step 2: Run all tests**

```bash
pytest tests/test_dynamic_probabilities.py -v
```

Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_dynamic_probabilities.py
git commit -m "test: add ensemble integration tests for balanced signal distribution"
```

---

### Task 11: Run existing test suite and verify no regressions

**Files:** All test files

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | head -80
```

Expected: All existing tests PASS. The changes are backward-compatible (fallback flags exist).

**Step 2: If failures, fix and re-run**

Common issues to check:
- Tests that mock `MarkovChain.analyze()` return value — add `ensemble_prob` to mock
- Tests that assert specific ensemble_prob values — values will differ with dynamic mapping

**Step 3: Commit if fixes needed**

```bash
git add -A
git commit -m "fix: update tests for dynamic probability mapping compatibility"
```

---

### Task 12: Kill running bot, restart, and verify balanced signals

**Step 1: Stop the running bot**

```bash
kill $(pgrep -f "run_trading.py")
```

**Step 2: Restart the bot**

```bash
python3 run_trading.py 2>&1 | tee /tmp/trading_dynamic_test.log &
```

**Step 3: Wait for training + replay, then check signals**

After the bot starts producing signals (post-replay), check the signal distribution:

```bash
sleep 300 && grep "SIGNAL:" /tmp/trading_dynamic_test.log | awk '{print $7}' | sort | uniq -c
```

Expected: Roughly balanced BUY/SELL/NEUTRAL instead of 95% SELL.

**Step 4: Check database for signal balance**

```python
python3 -c "
import sqlite3
db = sqlite3.connect('data/trading.db')
rows = db.execute('SELECT signal_type, COUNT(*) FROM signals GROUP BY signal_type').fetchall()
for r in rows:
    print(f'{r[0]}: {r[1]}')
db.close()
"
```

Expected: BUY and SELL counts should be within 2:1 ratio of each other.
