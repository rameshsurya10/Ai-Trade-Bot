# Dynamic Probability Mapping - Design Document

**Date:** 2026-02-19
**Status:** Approved
**Files:** `src/advanced_predictor.py`, `config.yaml`

## Problem Statement

The prediction engine generates 95% SELL signals due to hardcoded probability mappings that create systematic bearish bias:

1. **Kalman binary jump** - velocity sign maps to fixed 0.30/0.50/0.70. A velocity of -0.001 (barely negative) gets the same 0.30 as -0.5 (strong downtrend).
2. **Fourier binary jump** - phase bucket maps to fixed 0.25/0.50/0.75. Same issue.
3. **Markov permanent deficit** - dampening toward 1/3 (correct for 3-state system) but used in a binary ensemble (0.5 = neutral), creating -0.025 permanent bearish drag.
4. **Entropy doubles Kalman error** - follows Kalman direction with fixed probs, amplifying any Kalman bias.

### Impact Proof

With DEFAULT_WEIGHTS (lstm=0.35, fourier=0.15, kalman=0.20, markov=0.15, entropy=0.10, mc=0.05):

| Scenario | Current ensemble_prob | Result |
|----------|----------------------|--------|
| All neutral (nothing bearish) | **0.4745** | SELL (< 0.48 threshold) |
| Fourier BULLISH + Kalman DOWN | 0.4570 | SELL |
| Fourier NEUTRAL + Kalman DOWN | 0.4195 | SELL |
| Fourier BULLISH + Kalman UP | 0.5670 | BUY |

Even with zero directional evidence, the system produces SELL because Markov's 0.333 drags the ensemble below the sell threshold.

## Solution: Dynamic Probability Mappings

Replace discrete bucket-to-fixed-value mappings with continuous mathematical functions that preserve signal magnitude.

### Fix 1: Kalman Sigmoid Velocity Mapping

**File:** `advanced_predictor.py` lines 1097-1103

**Current:**
```python
kalman_prob = self.kalman.prob_sideways  # 0.50
if kalman_result['trend'] == 'UP':
    kalman_prob = self.kalman.prob_up      # 0.70
elif kalman_result['trend'] == 'DOWN':
    kalman_prob = self.kalman.prob_down    # 0.30
```

**New:**
```python
velocity = kalman_result.get('velocity', 0.0)
norm_velocity = velocity / (atr + 1e-10) if atr > 0 else 0.0
kalman_prob = 1.0 / (1.0 + math.exp(-self.kalman_sigmoid_sensitivity * norm_velocity))
kalman_prob = max(0.05, min(0.95, kalman_prob))
```

**Config:** `prediction.kalman.sigmoid_sensitivity: 2.0`

Sensitivity=2.0 produces:
- velocity=0 → prob=0.50 (exactly neutral)
- velocity=+0.1 ATR → prob=0.55 (slightly bullish)
- velocity=+0.5 ATR → prob=0.73 (bullish)
- velocity=-0.5 ATR → prob=0.27 (bearish)

**Normalization:** Divide velocity by ATR to make it scale-independent across BTC ($95K) and ETH ($2K). Without this, raw velocity in dollar terms is meaningless for comparison.

### Fix 2: Fourier Cosine Phase Mapping

**File:** `advanced_predictor.py` lines 1089-1095

**Current:**
```python
fourier_prob = self.fourier.prob_neutral  # 0.50
if fourier_result['signal'] == 'BULLISH':
    fourier_prob = self.fourier.prob_bullish   # 0.75
elif fourier_result['signal'] == 'BEARISH':
    fourier_prob = self.fourier.prob_bearish   # 0.25
```

**New:**
```python
cycle_phase = fourier_result.get('cycle_phase', 0.5)
if not fourier_result.get('dominant_frequencies'):
    fourier_prob = 0.5  # Error/default → neutral
else:
    fourier_prob = 0.5 + self.fourier_amplitude * math.cos(2 * math.pi * cycle_phase)
    fourier_prob = max(0.05, min(0.95, fourier_prob))
```

**Config:** `prediction.fourier.amplitude: 0.15`

Amplitude=0.15 produces range [0.35, 0.65]:
- phase=0.0 (trough, rising) → prob=0.65
- phase=0.25 (peak) → prob=0.50
- phase=0.50 (falling) → prob=0.35
- phase=0.75 (bottoming) → prob=0.50

**Edge case:** When Fourier returns default result (too few samples), `dominant_frequencies=[]`. The guard catches this and returns 0.50 instead of erroneously computing cos(pi) = -1 → prob=0.35.

### Fix 3: Markov Recentering

**File:** `advanced_predictor.py` lines 475-484 and 1105-1107

**Option chosen: Rescale prob_up/(prob_up + prob_down)**

In the `MarkovChain.analyze()` method, after computing dampened probabilities:
```python
# Convert 3-state probability to binary ensemble probability
# prob_up / (prob_up + prob_down) → centered at 0.50 when equal
markov_ensemble_prob = prob_up / (prob_up + prob_down + 1e-10)
```

Add `markov_ensemble_prob` to the returned dict:
```python
return {
    'prob_up': float(prob_up),
    'prob_down': float(prob_down),
    'prob_neutral': float(prob_neutral),
    'ensemble_prob': float(markov_ensemble_prob),  # NEW
    ...
}
```

In ensemble calculation:
```python
markov_prob = markov_result['ensemble_prob']  # Was: markov_result['prob_up']
```

**Math verification:**
- Equal dampened: prob_up=0.333, prob_down=0.333 → ensemble=0.500 (neutral)
- Slight up: prob_up=0.40, prob_down=0.30 → ensemble=0.571 (bullish)
- Slight down: prob_up=0.30, prob_down=0.40 → ensemble=0.429 (bearish)

### Fix 4: Entropy Certainty Scaling

**File:** `advanced_predictor.py` lines 1109-1118

**Current:**
```python
entropy_prob = 0.5
if entropy_result['regime'] == 'TRENDING':
    entropy_prob = 0.65 if kalman UP else 0.35
elif entropy_result['regime'] in ('NORMAL', 'CHOPPY'):
    if entropy_val < threshold:
        entropy_prob = 0.55 if kalman UP else 0.45
```

**New:**
```python
normalized_entropy = entropy_result.get('normalized_entropy', 0.5)
certainty = max(0.0, 1.0 - normalized_entropy)
entropy_prob = 0.5 + (kalman_prob - 0.5) * certainty * self.entropy_certainty_scaling
entropy_prob = max(0.05, min(0.95, entropy_prob))
```

**Config:** `prediction.entropy.certainty_scaling: 0.8`

**Dependency:** Requires Fix 1 (dynamic Kalman) to be applied first — `kalman_prob` must be computed before this line.

**Behavior:**
- Low entropy (ordered, certainty=0.8) + Kalman=0.60 → entropy=0.564 (amplifies)
- High entropy (chaotic, certainty=0.1) + Kalman=0.60 → entropy=0.508 (barely moves)
- Any entropy + Kalman=0.50 (flat) → entropy=0.500 (nothing to amplify)

### Fix 5: Symmetric Confidence Calculation

**File:** `advanced_predictor.py` lines 1150-1161

**Current:**
```python
elif ensemble_prob < self.sell_threshold:
    direction = "SELL"
    sell_prob = 1.0 - ensemble_prob  # Mirror trick
    confidence = (sell_prob - self.confidence_floor) / conf_range
```

**New:**
```python
elif ensemble_prob < self.sell_threshold:
    direction = "SELL"
    confidence = (0.5 - ensemble_prob) / conf_range
```

**Note:** With floor=0.50, the mirror trick is mathematically equivalent. This fix is for code clarity and to prevent bugs if floor changes in the future.

## Config Changes

Add to `config.yaml` under existing `prediction` section:

```yaml
prediction:
  fourier:
    dynamic_prob: true       # Enable cosine mapping (default: true)
    amplitude: 0.15          # Cosine amplitude, range [0.5-amp, 0.5+amp]
  kalman:
    dynamic_prob: true       # Enable sigmoid mapping (default: true)
    sigmoid_sensitivity: 2.0 # Sigmoid steepness (higher = more extreme)
  entropy:
    dynamic_prob: true       # Enable certainty scaling (default: true)
    certainty_scaling: 0.8   # How much entropy amplifies Kalman
  ensemble:
    symmetric_confidence: true  # Use symmetric confidence calc
```

All flags default to `true`. Setting `dynamic_prob: false` falls back to current behavior for any algorithm.

## Expected Outcome

| Scenario | Before | After |
|----------|--------|-------|
| Signal distribution | 95% SELL | ~35% BUY / 30% NEUTRAL / 35% SELL |
| All-neutral ensemble | 0.4745 (SELL) | 0.5000 (NEUTRAL) |
| Slight downtrend | 0.42 (SELL) | ~0.49 (NEUTRAL) |
| Strong downtrend | 0.38 (SELL) | ~0.43 (SELL) |
| Slight uptrend | 0.53 (BUY) | ~0.51 (NEUTRAL) |
| Strong uptrend | 0.57 (BUY) | ~0.57 (BUY) |

## Files Modified

1. `src/advanced_predictor.py` — Fixes 1, 2, 4, 5 (ensemble probability calculation)
2. `src/advanced_predictor.py` — Fix 3 (MarkovChain.analyze() return value)
3. `config.yaml` — New config keys with defaults

## Testing Strategy

1. Unit test: each algorithm produces centered output (mean ≈ 0.50) on random data
2. Unit test: Kalman sigmoid with known velocities produces expected probabilities
3. Unit test: Fourier cosine with known phases produces expected probabilities
4. Unit test: Markov ensemble_prob centered at 0.50 with balanced transitions
5. Integration test: ensemble produces balanced BUY/SELL/NEUTRAL on historical data
6. Backward compatibility: `dynamic_prob: false` reproduces old behavior exactly
