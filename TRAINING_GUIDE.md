# Training Guide: How to Build a Profitable AI Trading Bot

> **Goal**: Build a system that is PROFITABLE, not one that has 80% accuracy.
> These are different things!

---

## The Truth About Win Rates

### Why 80% Win Rate is NOT the Goal

```
COMMON MISCONCEPTION:
"I need 80% accuracy to be profitable"

REALITY:
You can be VERY profitable with 52-55% accuracy if you manage risk properly.
```

### The Math That Matters

| Win Rate | Risk:Reward | Trades | Wins | Losses | Profit per Win | Loss per Trade | NET RESULT |
|----------|-------------|--------|------|--------|----------------|----------------|------------|
| 80% | 1:1 | 100 | 80 | 20 | $100 | $100 | +$6,000 |
| 55% | 1:2 | 100 | 55 | 45 | $200 | $100 | +$6,500 |
| 52% | 1:3 | 100 | 52 | 48 | $300 | $100 | +$10,800 |

**Key Insight**: 52% accuracy with 1:3 risk:reward BEATS 80% accuracy with 1:1!

---

## Step-by-Step Training Pipeline

### PHASE 1: Data Collection (Week 1-2)

#### Step 1.1: Choose Your Market
```python
# Recommended starting markets (easiest to most difficult):
MARKETS = {
    "crypto_spot": {
        "examples": ["BTC-USD", "ETH-USD"],
        "why": "24/7 trading, free data, high volatility",
        "difficulty": "Medium",
        "min_capital": "$100"
    },
    "forex": {
        "examples": ["EUR/USD", "GBP/USD"],
        "why": "High liquidity, clear trends",
        "difficulty": "Medium-Hard",
        "min_capital": "$1,000"
    },
    "stocks": {
        "examples": ["SPY", "AAPL", "MSFT"],
        "why": "Established patterns, good data",
        "difficulty": "Hard",
        "min_capital": "$25,000 (PDT rule)"
    }
}

# START WITH: BTC-USD on 1-hour timeframe
# WHY: Free data, 24/7, enough volatility to learn
```

#### Step 1.2: Download Historical Data
```python
"""
scripts/download_data.py
Download historical data for training
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def download_crypto_data(
    symbol: str = "BTC-USD",
    start_date: str = "2020-01-01",
    end_date: str = None,
    interval: str = "1h"
) -> pd.DataFrame:
    """
    Download historical OHLCV data.

    Args:
        symbol: Trading pair (e.g., "BTC-USD")
        start_date: Start date for data
        end_date: End date (default: today)
        interval: Candle interval (1m, 5m, 15m, 1h, 1d)

    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Downloading {symbol} data from {start_date} to {end_date}...")

    # Download data
    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=True
    )

    # Clean column names
    data.columns = [col.lower() for col in data.columns]

    # Add returns column
    data['returns'] = data['close'].pct_change()

    # Add target (next candle direction)
    # 1 = up, 0 = down
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    # Remove last row (no target) and first row (no return)
    data = data.iloc[1:-1]

    print(f"Downloaded {len(data)} candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Up candles: {data['target'].sum()} ({data['target'].mean()*100:.1f}%)")
    print(f"Down candles: {len(data) - data['target'].sum()} ({(1-data['target'].mean())*100:.1f}%)")

    return data


# Usage:
if __name__ == "__main__":
    # Download 4 years of hourly BTC data
    data = download_crypto_data(
        symbol="BTC-USD",
        start_date="2020-01-01",
        interval="1h"
    )

    # Save to CSV
    data.to_csv("data/btc_hourly.csv")
    print(f"Saved to data/btc_hourly.csv")
```

**Run this first:**
```bash
mkdir -p data
python scripts/download_data.py
```

---

### PHASE 2: Feature Engineering (Week 3-4)

#### Step 2.1: Create Technical Indicators
```python
"""
src/features/technical_indicators.py
Calculate technical indicators for prediction
"""

import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    """
    Calculate technical indicators and statistical features.

    These features help the model understand market conditions.
    """

    def __init__(self):
        self.feature_names = []

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for the dataset.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all features added
        """
        df = df.copy()

        # 1. TREND INDICATORS
        df = self._add_moving_averages(df)
        df = self._add_macd(df)

        # 2. MOMENTUM INDICATORS
        df = self._add_rsi(df)
        df = self._add_stochastic(df)

        # 3. VOLATILITY INDICATORS
        df = self._add_bollinger_bands(df)
        df = self._add_atr(df)

        # 4. VOLUME INDICATORS
        df = self._add_volume_features(df)

        # 5. STATISTICAL FEATURES
        df = self._add_statistical_features(df)

        # 6. LAG FEATURES (past candles)
        df = self._add_lag_features(df)

        # Remove NaN rows (from indicator calculations)
        df = df.dropna()

        print(f"Created {len(self.feature_names)} features")
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA and EMA indicators."""
        for period in [10, 20, 50, 100]:
            # Simple Moving Average
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            self.feature_names.append(f'sma_{period}')

            # Price relative to SMA (normalized)
            df[f'close_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            self.feature_names.append(f'close_sma_{period}_ratio')

        # EMA
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            self.feature_names.append(f'ema_{period}')

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator."""
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()

        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        self.feature_names.extend(['macd', 'macd_signal', 'macd_histogram'])
        return df

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator."""
        delta = df['close'].diff()

        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # RSI zones
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

        self.feature_names.extend(['rsi', 'rsi_oversold', 'rsi_overbought'])
        return df

    def _add_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        self.feature_names.extend(['stoch_k', 'stoch_d'])
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()

        df['bb_upper'] = sma + (2 * std)
        df['bb_lower'] = sma - (2 * std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma

        # Position within bands (0 = at lower, 1 = at upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        self.feature_names.extend(['bb_width', 'bb_position'])
        return df

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range (volatility)."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(period).mean()

        # ATR as percentage of price
        df['atr_percent'] = df['atr'] / df['close'] * 100

        self.feature_names.extend(['atr', 'atr_percent'])
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving average
        df['volume_sma_20'] = df['volume'].rolling(20).mean()

        # Volume relative to average
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Price-volume trend
        df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()

        self.feature_names.extend(['volume_ratio'])
        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        # Rolling statistics on returns
        for period in [10, 20, 50]:
            df[f'returns_mean_{period}'] = df['returns'].rolling(period).mean()
            df[f'returns_std_{period}'] = df['returns'].rolling(period).std()
            df[f'returns_skew_{period}'] = df['returns'].rolling(period).skew()

            self.feature_names.extend([
                f'returns_mean_{period}',
                f'returns_std_{period}',
                f'returns_skew_{period}'
            ])

        # Z-score of current return
        df['returns_zscore'] = (df['returns'] - df['returns'].rolling(20).mean()) / df['returns'].rolling(20).std()
        self.feature_names.append('returns_zscore')

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features (past candle info)."""
        # Past returns
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
            self.feature_names.append(f'return_lag_{lag}')

        # Consecutive up/down candles
        df['up_streak'] = df['returns'].apply(lambda x: 1 if x > 0 else 0)
        df['up_streak'] = df['up_streak'].groupby(
            (df['up_streak'] != df['up_streak'].shift()).cumsum()
        ).cumsum()
        self.feature_names.append('up_streak')

        return df

    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return self.feature_names


# Usage:
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/btc_hourly.csv", index_col=0, parse_dates=True)

    # Calculate features
    fe = FeatureEngineer()
    df_features = fe.calculate_all_features(df)

    # Save
    df_features.to_csv("data/btc_hourly_features.csv")
    print(f"Saved {len(df_features)} rows with {len(fe.feature_names)} features")
```

---

### PHASE 3: Model Training (Week 5-8)

#### Step 3.1: Prepare Data for Training
```python
"""
src/training/prepare_data.py
Prepare data for model training with proper splits
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader


class TradingDataset(Dataset):
    """PyTorch dataset for time series trading data."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 100
    ):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx + self.sequence_length]
        # Get target (next candle direction)
        y = self.targets[idx + self.sequence_length]

        return {
            'features': torch.FloatTensor(x),
            'target': torch.LongTensor([y])
        }


def prepare_data(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str = 'target',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    sequence_length: int = 100
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Prepare data with proper train/val/test splits.

    CRITICAL: We use TIME-BASED splits, not random splits!
    This prevents look-ahead bias.

    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names
        target_column: Name of target column
        train_ratio: Proportion for training (0.7 = 70%)
        val_ratio: Proportion for validation (0.15 = 15%)
        sequence_length: Number of past candles to use

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    # Extract features and target
    X = df[feature_columns].values
    y = df[target_column].values

    # Calculate split indices
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    print(f"Data split:")
    print(f"  Training:   0 to {train_end} ({train_ratio*100:.0f}%)")
    print(f"  Validation: {train_end} to {val_end} ({val_ratio*100:.0f}%)")
    print(f"  Testing:    {val_end} to {n} ({(1-train_ratio-val_ratio)*100:.0f}%)")

    # Split data (TIME-BASED - no shuffling!)
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    # Scale features (fit ONLY on training data!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)  # Use train statistics
    X_test_scaled = scaler.transform(X_test)  # Use train statistics

    # Create datasets
    train_dataset = TradingDataset(X_train_scaled, y_train, sequence_length)
    val_dataset = TradingDataset(X_val_scaled, y_val, sequence_length)
    test_dataset = TradingDataset(X_test_scaled, y_test, sequence_length)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False  # NEVER shuffle time series!
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"\nDataset sizes:")
    print(f"  Training:   {len(train_dataset)} sequences")
    print(f"  Validation: {len(val_dataset)} sequences")
    print(f"  Testing:    {len(test_dataset)} sequences")

    return train_loader, val_loader, test_loader, scaler
```

#### Step 3.2: Train the LSTM Model
```python
"""
src/training/train_model.py
Train the LSTM prediction model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import json


class TrendPredictor(nn.Module):
    """
    LSTM model for predicting price direction.

    This model predicts: Will the next candle go UP or DOWN?
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)  # 2 classes: down (0), up (1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Train the model with early stopping.

    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum training epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        device: Device to train on

    Returns:
        Training history dictionary
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'best_accuracy': 0
    }

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = batch['features'].to(device)
            targets = batch['target'].squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['target'].squeeze().to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.2%}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            history['best_accuracy'] = val_accuracy
            patience_counter = 0

            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_accuracy
            }, 'models/best_model.pt')
            print(f"  -> Saved best model (accuracy: {val_accuracy:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history


def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate model on test set.

    Returns detailed metrics.
    """
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].squeeze()

            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probabilities.extend(probs[:, 1].cpu().numpy())  # P(up)

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)

    # Calculate metrics
    accuracy = (predictions == targets).mean()

    # Precision, recall for "up" predictions
    true_positives = ((predictions == 1) & (targets == 1)).sum()
    false_positives = ((predictions == 1) & (targets == 0)).sum()
    false_negatives = ((predictions == 0) & (targets == 1)).sum()

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"\nPrediction distribution:")
    print(f"  Predicted UP:   {(predictions == 1).sum()} ({(predictions == 1).mean():.1%})")
    print(f"  Predicted DOWN: {(predictions == 0).sum()} ({(predictions == 0).mean():.1%})")
    print(f"\nActual distribution:")
    print(f"  Actual UP:   {(targets == 1).sum()} ({(targets == 1).mean():.1%})")
    print(f"  Actual DOWN: {(targets == 0).sum()} ({(targets == 0).mean():.1%})")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'predictions': predictions,
        'targets': targets,
        'probabilities': probabilities
    }
```

#### Step 3.3: Main Training Script
```python
"""
scripts/train.py
Main script to train the trading model
"""

import pandas as pd
import torch
import os

# Import our modules
from src.features.technical_indicators import FeatureEngineer
from src.training.prepare_data import prepare_data
from src.training.train_model import TrendPredictor, train_model, evaluate_model


def main():
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Load data
    print("Loading data...")
    df = pd.read_csv("data/btc_hourly.csv", index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows")

    # 2. Calculate features
    print("\nCalculating features...")
    fe = FeatureEngineer()
    df = fe.calculate_all_features(df)
    feature_columns = fe.get_feature_columns()
    print(f"Created {len(feature_columns)} features")

    # 3. Prepare data
    print("\nPreparing data...")
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df=df,
        feature_columns=feature_columns,
        target_column='target',
        train_ratio=0.7,
        val_ratio=0.15,
        sequence_length=100
    )

    # 4. Create model
    print("\nCreating model...")
    model = TrendPredictor(
        input_size=len(feature_columns),
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")

    # 5. Train model
    print("\nTraining model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        learning_rate=0.001,
        patience=15
    )

    # 6. Load best model and evaluate
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load("models/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nBest validation accuracy: {checkpoint['val_accuracy']:.2%}")

    # 7. Evaluate on test set
    results = evaluate_model(model, test_loader)

    # 8. Save scaler for inference
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")
    print("\nSaved scaler to models/scaler.pkl")

    # 9. Print final summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Test Accuracy: {results['accuracy']:.2%}")
    print(f"\nFiles saved:")
    print("  - models/best_model.pt (trained model)")
    print("  - models/scaler.pkl (feature scaler)")

    # Reality check
    print("\n" + "="*50)
    print("REALITY CHECK")
    print("="*50)
    if results['accuracy'] > 0.55:
        print("Your model is performing WELL (>55% accuracy).")
        print("This is a good foundation for a profitable system.")
    elif results['accuracy'] > 0.52:
        print("Your model is performing OKAY (52-55% accuracy).")
        print("With good risk management, this can be profitable.")
    else:
        print("Your model is performing at baseline (~50% accuracy).")
        print("Consider: more features, different timeframe, or more data.")

    print("\nREMEMBER: 52% accuracy + 1:2 risk:reward = PROFITABLE!")


if __name__ == "__main__":
    main()
```

---

### PHASE 4: Risk Management (CRITICAL!)

This is where profitability comes from, NOT accuracy.

#### Step 4.1: Implement Position Sizing
```python
"""
src/trading/risk_management.py
Risk management - THE KEY TO PROFITABILITY
"""


class RiskManager:
    """
    Risk management system.

    THIS IS MORE IMPORTANT THAN MODEL ACCURACY!

    Key principles:
    1. Never risk more than 2% of capital per trade
    2. Use stop losses on EVERY trade
    3. Target risk:reward of at least 1:2
    """

    def __init__(
        self,
        capital: float,
        risk_per_trade: float = 0.02,  # 2% risk per trade
        max_position_pct: float = 0.10  # Max 10% of capital per position
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float
    ) -> dict:
        """
        Calculate position size based on risk.

        Example:
            Capital: $10,000
            Risk per trade: 2% = $200
            Entry: $50,000
            Stop loss: $49,000 (2% below entry)

            Risk per unit = $50,000 - $49,000 = $1,000
            Position size = $200 / $1,000 = 0.2 BTC
            Position value = 0.2 × $50,000 = $10,000
        """
        # Calculate maximum risk in dollars
        max_risk_dollars = self.capital * self.risk_per_trade

        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit == 0:
            return {"error": "Stop loss cannot equal entry price"}

        # Calculate position size
        position_size = max_risk_dollars / risk_per_unit
        position_value = position_size * entry_price

        # Apply maximum position constraint
        max_position_value = self.capital * self.max_position_pct
        if position_value > max_position_value:
            position_value = max_position_value
            position_size = position_value / entry_price

        return {
            "position_size": position_size,
            "position_value": position_value,
            "risk_dollars": position_size * risk_per_unit,
            "risk_percent": (position_size * risk_per_unit) / self.capital * 100,
            "entry_price": entry_price,
            "stop_loss": stop_loss_price
        }

    def calculate_targets(
        self,
        entry_price: float,
        stop_loss_price: float,
        direction: str = "long",
        risk_reward_ratio: float = 2.0
    ) -> dict:
        """
        Calculate take profit based on risk:reward ratio.

        With 1:2 risk:reward:
        - Stop loss = 2% below entry
        - Take profit = 4% above entry

        This means even with 40% win rate, you can be profitable!
        """
        stop_distance = abs(entry_price - stop_loss_price)
        profit_distance = stop_distance * risk_reward_ratio

        if direction == "long":
            take_profit = entry_price + profit_distance
        else:
            take_profit = entry_price - profit_distance

        return {
            "entry": entry_price,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit,
            "risk_reward": risk_reward_ratio,
            "stop_distance_pct": stop_distance / entry_price * 100,
            "profit_distance_pct": profit_distance / entry_price * 100
        }


# Example usage:
if __name__ == "__main__":
    # Initialize with $10,000 capital
    rm = RiskManager(capital=10000, risk_per_trade=0.02)

    # Example trade
    entry = 50000  # BTC price
    stop = 49000   # 2% stop loss

    # Calculate position size
    position = rm.calculate_position_size(entry, stop)
    print("Position Size Calculation:")
    print(f"  Entry price: ${entry:,}")
    print(f"  Stop loss: ${stop:,}")
    print(f"  Position size: {position['position_size']:.4f} BTC")
    print(f"  Position value: ${position['position_value']:,.2f}")
    print(f"  Risk: ${position['risk_dollars']:.2f} ({position['risk_percent']:.1f}%)")

    # Calculate targets
    targets = rm.calculate_targets(entry, stop, "long", risk_reward_ratio=2.0)
    print(f"\nTrade Targets (1:2 Risk:Reward):")
    print(f"  Entry: ${targets['entry']:,}")
    print(f"  Stop Loss: ${targets['stop_loss']:,} (-{targets['stop_distance_pct']:.1f}%)")
    print(f"  Take Profit: ${targets['take_profit']:,} (+{targets['profit_distance_pct']:.1f}%)")
```

---

### PHASE 5: Backtesting (Week 9-12)

```python
"""
src/backtesting/backtest.py
Backtest the trading strategy
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    pnl: float
    pnl_percent: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal'


class Backtester:
    """
    Backtest trading strategies with realistic assumptions.

    Includes:
    - Transaction costs
    - Slippage
    - Proper position sizing
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        commission_pct: float = 0.001,  # 0.1% per trade
        slippage_pct: float = 0.0005,   # 0.05% slippage
        risk_per_trade: float = 0.02,
        risk_reward: float = 2.0
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk_per_trade = risk_per_trade
        self.risk_reward = risk_reward

    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        confidence_threshold: float = 0.55
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            predictions: Model predictions (0=down, 1=up) or probabilities
            confidence_threshold: Minimum confidence to trade

        Returns:
            Dictionary with backtest results
        """
        capital = self.initial_capital
        trades: List[Trade] = []
        equity_curve = [capital]

        in_position = False
        current_trade = None

        for i in range(len(df) - 1):
            current_price = df['close'].iloc[i]
            next_open = df['open'].iloc[i + 1]
            next_high = df['high'].iloc[i + 1]
            next_low = df['low'].iloc[i + 1]
            next_close = df['close'].iloc[i + 1]

            # If in position, check for exit
            if in_position:
                # Check stop loss
                if current_trade['direction'] == 'long':
                    if next_low <= current_trade['stop_loss']:
                        # Hit stop loss
                        exit_price = current_trade['stop_loss'] * (1 - self.slippage_pct)
                        pnl = (exit_price - current_trade['entry_price']) * current_trade['position_size']
                        pnl -= exit_price * current_trade['position_size'] * self.commission_pct

                        capital += pnl
                        trades.append(Trade(
                            entry_time=current_trade['entry_time'],
                            exit_time=df.index[i + 1],
                            direction='long',
                            entry_price=current_trade['entry_price'],
                            exit_price=exit_price,
                            stop_loss=current_trade['stop_loss'],
                            take_profit=current_trade['take_profit'],
                            position_size=current_trade['position_size'],
                            pnl=pnl,
                            pnl_percent=pnl / self.initial_capital * 100,
                            exit_reason='stop_loss'
                        ))
                        in_position = False

                    elif next_high >= current_trade['take_profit']:
                        # Hit take profit
                        exit_price = current_trade['take_profit'] * (1 - self.slippage_pct)
                        pnl = (exit_price - current_trade['entry_price']) * current_trade['position_size']
                        pnl -= exit_price * current_trade['position_size'] * self.commission_pct

                        capital += pnl
                        trades.append(Trade(
                            entry_time=current_trade['entry_time'],
                            exit_time=df.index[i + 1],
                            direction='long',
                            entry_price=current_trade['entry_price'],
                            exit_price=exit_price,
                            stop_loss=current_trade['stop_loss'],
                            take_profit=current_trade['take_profit'],
                            position_size=current_trade['position_size'],
                            pnl=pnl,
                            pnl_percent=pnl / self.initial_capital * 100,
                            exit_reason='take_profit'
                        ))
                        in_position = False

            # If not in position, check for entry
            if not in_position:
                prediction = predictions[i]

                # Check if we have a signal with enough confidence
                if isinstance(prediction, float):
                    # Probability output
                    if prediction > confidence_threshold:
                        signal = 'long'
                        confidence = prediction
                    elif prediction < (1 - confidence_threshold):
                        signal = 'short'
                        confidence = 1 - prediction
                    else:
                        signal = None
                else:
                    # Binary output
                    signal = 'long' if prediction == 1 else 'short'
                    confidence = 0.6  # Assume some confidence

                if signal == 'long':
                    # Calculate entry with slippage
                    entry_price = next_open * (1 + self.slippage_pct)

                    # Calculate stop loss (2% below)
                    stop_loss = entry_price * 0.98

                    # Calculate take profit (risk:reward)
                    stop_distance = entry_price - stop_loss
                    take_profit = entry_price + (stop_distance * self.risk_reward)

                    # Calculate position size
                    risk_dollars = capital * self.risk_per_trade
                    risk_per_unit = entry_price - stop_loss
                    position_size = risk_dollars / risk_per_unit
                    position_value = position_size * entry_price

                    # Check if we have enough capital
                    if position_value <= capital:
                        # Pay commission
                        commission = position_value * self.commission_pct
                        capital -= commission

                        current_trade = {
                            'entry_time': df.index[i + 1],
                            'direction': 'long',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'position_size': position_size
                        }
                        in_position = True

            equity_curve.append(capital)

        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve)

    def _calculate_metrics(self, trades: List[Trade], equity_curve: List[float]) -> Dict:
        """Calculate performance metrics."""
        if len(trades) == 0:
            return {"error": "No trades executed"}

        # Win/loss stats
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(trades)

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Returns
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital

        # Drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Exit reason breakdown
        stop_loss_exits = len([t for t in trades if t.exit_reason == 'stop_loss'])
        take_profit_exits = len([t for t in trades if t.exit_reason == 'take_profit'])

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "final_capital": equity_curve[-1],
            "max_drawdown": max_drawdown,
            "stop_loss_exits": stop_loss_exits,
            "take_profit_exits": take_profit_exits,
            "equity_curve": equity_curve,
            "trades": trades
        }


def print_backtest_results(results: Dict):
    """Print backtest results in a nice format."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    print(f"\nTrade Statistics:")
    print(f"  Total Trades:    {results['total_trades']}")
    print(f"  Winning Trades:  {results['winning_trades']}")
    print(f"  Losing Trades:   {results['losing_trades']}")
    print(f"  Win Rate:        {results['win_rate']:.1%}")

    print(f"\nProfitability:")
    print(f"  Total Return:    {results['total_return']:.1%}")
    print(f"  Final Capital:   ${results['final_capital']:,.2f}")
    print(f"  Profit Factor:   {results['profit_factor']:.2f}")
    print(f"  Max Drawdown:    {results['max_drawdown']:.1%}")

    print(f"\nAverage Trade:")
    print(f"  Avg Win:         ${results['avg_win']:.2f}")
    print(f"  Avg Loss:        ${results['avg_loss']:.2f}")

    print(f"\nExit Reasons:")
    print(f"  Stop Loss:       {results['stop_loss_exits']} ({results['stop_loss_exits']/results['total_trades']:.1%})")
    print(f"  Take Profit:     {results['take_profit_exits']} ({results['take_profit_exits']/results['total_trades']:.1%})")

    # Profitability assessment
    print("\n" + "="*60)
    if results['total_return'] > 0:
        print("RESULT: PROFITABLE STRATEGY!")
        print(f"With {results['win_rate']:.1%} win rate and {results['profit_factor']:.1f}x profit factor")
    else:
        print("RESULT: NOT PROFITABLE - needs improvement")
    print("="*60)
```

---

## Summary: Your Path to Profitability

### What You Actually Need:

| Factor | Target | Why It Matters |
|--------|--------|----------------|
| **Accuracy** | 52-58% | Higher is better but not critical |
| **Risk:Reward** | 1:2 or better | THIS is where profit comes from |
| **Risk per trade** | 2% max | Preserves capital during losing streaks |
| **Stop loss** | Always use | Limits losses to manageable size |
| **Consistency** | Follow system | Emotions destroy profits |

### The Formula for Profitability:

```
Profit = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

Example with 52% accuracy and 1:2 risk:reward:
  Win Rate: 52%
  Loss Rate: 48%
  Avg Win: $200 (2× risk)
  Avg Loss: $100 (1× risk)

  Profit per 100 trades = (52 × $200) - (48 × $100)
                        = $10,400 - $4,800
                        = $5,600 PROFIT!
```

### Run Order:

```bash
# 1. Download data
python scripts/download_data.py

# 2. Train model
python scripts/train.py

# 3. Backtest
python scripts/backtest.py

# 4. Paper trade for 4+ weeks before using real money!
```

---

**Remember**: 80% win rate is a fantasy. 52% win rate with proper risk management is REAL profitability.
