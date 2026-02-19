"""
Standalone LSTM Retraining Script
===================================
Retrains all LSTM models for configured symbols and intervals using
existing candle data from the SQLite database.

Replicates the exact training logic from runner.py _ensure_models_ready()
with the same hyperparameters, architecture, and training loop.

Usage:
    python3 retrain_now.py
    # or inside venv:
    source venv/bin/activate && python3 retrain_now.py
"""

import gc
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root to path so src.* imports work from any working directory
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.core.database import Database
from src.analysis_engine import FeatureCalculator, LSTMModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def retrain_all_models(config_path: str = "config.yaml") -> None:
    """
    Retrain LSTM models for every enabled symbol + interval pair.

    Mirrors _ensure_models_ready() in runner.py exactly:
    - Same LSTMModel class and hyperparameters
    - Same sliding-window feature construction
    - Same Adam + weight_decay + gradient clipping
    - Same early-stopping and LR scheduler logic
    - Same checkpoint format (torch.save dict)
    """
    logger.info("=" * 70)
    logger.info("STANDALONE LSTM RETRAINER")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config = Config.load(str(PROJECT_ROOT / config_path))
    raw = config.raw

    # ------------------------------------------------------------------
    # 2. Connect to database
    # ------------------------------------------------------------------
    db_path = str(PROJECT_ROOT / config.database.path)
    logger.info(f"Connecting to database: {db_path}")
    database = Database(db_path)

    # ------------------------------------------------------------------
    # 3. Read training hyper-params from config
    # ------------------------------------------------------------------
    auto_train_cfg = raw.get("auto_training", {})
    min_candles:       int   = auto_train_cfg.get("min_candles", 1000)
    training_candles:  int   = auto_train_cfg.get("training_candles", 5000)
    min_accuracy_req:  float = auto_train_cfg.get("min_accuracy_required", 0.58)
    target_accuracy:   float = auto_train_cfg.get("target_accuracy", 0.65)
    max_epochs:        int   = auto_train_cfg.get("max_epochs", 100)
    batch_size:        int   = auto_train_cfg.get("batch_size", 32)
    learning_rate:     float = auto_train_cfg.get("learning_rate", 0.001)
    patience:          int   = auto_train_cfg.get("patience", 10)

    # Model architecture from config
    hidden_size:   int   = config.model.hidden_size
    num_layers:    int   = config.model.num_layers
    dropout:       float = config.model.dropout
    models_dir = PROJECT_ROOT / config.model.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Scheduler
    scheduler_cfg = auto_train_cfg.get("scheduler", {})
    use_scheduler = scheduler_cfg.get("enabled", False)

    # ------------------------------------------------------------------
    # 4. Determine symbols and intervals
    # ------------------------------------------------------------------
    symbols: list = raw.get("symbols", [config.data.symbol])
    timeframe_cfg = raw.get("timeframes", {})
    training_intervals: list = []
    interval_seq_lengths: dict = {}

    if timeframe_cfg.get("enabled", False):
        for tf in timeframe_cfg.get("intervals", []):
            if tf.get("enabled", True) and tf.get("interval"):
                ivl = tf["interval"]
                training_intervals.append(ivl)
                interval_seq_lengths[ivl] = tf.get(
                    "sequence_length", config.model.sequence_length
                )

    if not training_intervals:
        training_intervals = [config.data.interval or "1h"]
        interval_seq_lengths[training_intervals[0]] = config.model.sequence_length

    logger.info(f"Symbols      : {symbols}")
    logger.info(f"Intervals    : {training_intervals}")
    logger.info(f"Models dir   : {models_dir}")
    logger.info(f"Min accuracy : {min_accuracy_req:.1%}")
    logger.info(f"Target acc   : {target_accuracy:.1%}")
    logger.info(f"Max epochs   : {max_epochs}")
    logger.info(f"Batch size   : {batch_size}")

    # Cache feature columns once — static across all symbols/intervals
    feature_columns = FeatureCalculator.get_feature_columns()

    # GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    # ------------------------------------------------------------------
    # 5. Training loop: symbol × interval
    # ------------------------------------------------------------------
    overall_results: dict = {}

    for symbol in symbols:
        symbol_ready = False
        symbol_best_acc = 0.0

        for train_interval in training_intervals:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"[{symbol} @ {train_interval}] Training starting")
            logger.info(f"{'=' * 60}")

            safe_symbol = symbol.replace("/", "_").replace("-", "_")
            model_path = models_dir / f"model_{safe_symbol}_{train_interval}.pt"
            sequence_length = interval_seq_lengths.get(
                train_interval, config.model.sequence_length
            )

            try:
                # --------------------------------------------------------
                # 5a. Fetch candle data
                # --------------------------------------------------------
                logger.info(
                    f"[{symbol} @ {train_interval}] Fetching up to "
                    f"{training_candles + 100} candles..."
                )
                candles_df = database.get_candles(
                    symbol=symbol,
                    interval=train_interval,
                    limit=training_candles + 100,  # Extra for indicator warmup
                )

                if candles_df is None or len(candles_df) < min_candles:
                    count = len(candles_df) if candles_df is not None else 0
                    logger.warning(
                        f"[{symbol} @ {train_interval}] Insufficient candles: "
                        f"{count} < {min_candles} — skipping interval."
                    )
                    continue

                logger.info(
                    f"[{symbol} @ {train_interval}] {len(candles_df)} candles fetched"
                )

                # --------------------------------------------------------
                # 5b. Feature engineering
                # --------------------------------------------------------
                df_features = FeatureCalculator.calculate_all(candles_df)
                features = df_features[feature_columns].values
                closes   = df_features["close"].values

                # Normalise features
                feature_means = np.nanmean(features, axis=0)
                feature_stds  = np.nanstd(features, axis=0)
                features = (features - feature_means) / (feature_stds + 1e-8)
                features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

                # --------------------------------------------------------
                # 5c. Build sliding-window sequences (vectorised, zero-copy)
                # shape: (num_samples, sequence_length, num_features)
                # --------------------------------------------------------
                num_sequences = len(features) - sequence_length - 1  # -1 for target

                X = (
                    np.lib.stride_tricks.sliding_window_view(
                        features, window_shape=sequence_length, axis=0
                    )[:num_sequences]
                    .transpose(0, 2, 1)
                    .copy()  # Own memory after transpose
                )

                current_closes = closes[sequence_length - 1 : sequence_length - 1 + num_sequences]
                next_closes    = closes[sequence_length     : sequence_length     + num_sequences]
                y = (next_closes > current_closes).astype(np.float64)

                # Drop any NaN rows
                valid = ~(np.isnan(y) | np.isnan(X).any(axis=(1, 2)))
                X, y  = X[valid], y[valid]

                if len(X) < min_candles:
                    logger.warning(
                        f"[{symbol} @ {train_interval}] Only {len(X)} sequences "
                        f"after filtering — skipping."
                    )
                    continue

                n_positive  = float(y.sum())
                label_ratio = n_positive / len(y) if len(y) > 0 else 0.5
                logger.info(
                    f"[{symbol} @ {train_interval}] {len(X)} sequences | "
                    f"labels: {label_ratio:.1%} UP / {1 - label_ratio:.1%} DOWN"
                )

                # --------------------------------------------------------
                # 5d. Train / validation split (chronological 80 / 20)
                # --------------------------------------------------------
                min_val_samples = 50
                split_idx = int(len(X) * 0.8)

                if len(X) - split_idx < min_val_samples:
                    if len(X) < min_val_samples * 2:
                        logger.warning(
                            f"[{symbol} @ {train_interval}] Not enough data for "
                            f"reliable train/val split — skipping."
                        )
                        continue
                    split_idx = len(X) - min_val_samples

                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                logger.info(
                    f"[{symbol} @ {train_interval}] "
                    f"Train: {len(X_train)}, Val: {len(X_val)}"
                )

                # --------------------------------------------------------
                # 5e. DataLoaders
                # --------------------------------------------------------
                train_ds = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_train), torch.FloatTensor(y_train)
                )
                val_ds = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_val), torch.FloatTensor(y_val)
                )
                train_loader = torch.utils.data.DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    val_ds, batch_size=batch_size
                )

                # --------------------------------------------------------
                # 5f. Build model
                # --------------------------------------------------------
                model = LSTMModel(
                    input_size=len(feature_columns),
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                ).to(device)

                criterion  = nn.BCELoss()
                optimizer  = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=1e-4,  # L2 regularisation
                )

                scheduler = None
                if use_scheduler:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        factor=scheduler_cfg.get("factor", 0.5),
                        patience=scheduler_cfg.get("patience", 5),
                        min_lr=scheduler_cfg.get("min_lr", 1e-5),
                    )

                # --------------------------------------------------------
                # 5g. Training loop with early stopping
                # --------------------------------------------------------
                best_val_acc    = 0.0
                best_state      = None
                patience_counter = 0

                logger.info(
                    f"[{symbol} @ {train_interval}] "
                    f"Training for up to {max_epochs} epochs..."
                )

                for epoch in range(max_epochs):
                    # -- Train --
                    model.train()
                    epoch_loss = 0.0
                    for X_batch, y_batch in train_loader:
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        optimizer.zero_grad()
                        outputs = model(X_batch).squeeze()
                        loss    = criterion(outputs, y_batch)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        epoch_loss += loss.item()

                    # -- Validate --
                    model.eval()
                    correct = 0
                    total   = 0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch = X_batch.to(device)
                            y_batch = y_batch.to(device)
                            outputs     = model(X_batch).squeeze()
                            predictions = (outputs > 0.5).float()
                            correct += (predictions == y_batch).sum().item()
                            total   += len(y_batch)

                    val_acc = correct / total if total > 0 else 0.0

                    if scheduler is not None:
                        scheduler.step(val_acc)

                    # Track best
                    if val_acc > best_val_acc:
                        best_val_acc  = val_acc
                        best_state    = {k: v.clone().detach() for k, v in model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Log every 10 epochs
                    if (epoch + 1) % 10 == 0:
                        avg_loss   = epoch_loss / len(train_loader)
                        current_lr = optimizer.param_groups[0]["lr"]
                        logger.info(
                            f"[{symbol} @ {train_interval}] "
                            f"Epoch {epoch + 1}/{max_epochs} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Val Acc: {val_acc:.2%} | "
                            f"LR: {current_lr:.6f}"
                        )

                    # Early stopping
                    if patience_counter >= patience:
                        logger.info(
                            f"[{symbol} @ {train_interval}] "
                            f"Early stopping at epoch {epoch + 1}"
                        )
                        break

                    # Target reached
                    if val_acc >= target_accuracy:
                        logger.info(
                            f"[{symbol} @ {train_interval}] "
                            f"[OK] Target accuracy reached: {val_acc:.2%}"
                        )
                        break

                # --------------------------------------------------------
                # 5h. Save best model checkpoint
                # --------------------------------------------------------
                if best_state is not None:
                    model.cpu()
                    model.load_state_dict(best_state)

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": {
                                "hidden_size":     hidden_size,
                                "num_layers":      num_layers,
                                "dropout":         dropout,
                                "sequence_length": sequence_length,
                                "input_size":      len(feature_columns),
                            },
                            "feature_means":        feature_means,
                            "feature_stds":         feature_stds,
                            "symbol":               symbol,
                            "interval":             train_interval,
                            "trained_at":           datetime.utcnow().isoformat(),
                            "samples_trained":      len(X_train),
                            "validation_accuracy":  best_val_acc,
                        },
                        model_path,
                    )

                    symbol_best_acc = max(symbol_best_acc, best_val_acc)

                    if best_val_acc >= min_accuracy_req:
                        symbol_ready = True
                        logger.info(
                            f"[{symbol} @ {train_interval}] [VALIDATED] MODEL READY\n"
                            f"    Accuracy : {best_val_acc:.2%} >= {min_accuracy_req:.2%} required\n"
                            f"    Samples  : {len(X_train)}\n"
                            f"    Saved to : {model_path}"
                        )
                    else:
                        logger.warning(
                            f"[{symbol} @ {train_interval}] [BELOW TARGET]\n"
                            f"    Accuracy : {best_val_acc:.2%} < {min_accuracy_req:.2%} required\n"
                            f"    Saved to : {model_path} (will still be used)"
                        )
                else:
                    logger.error(
                        f"[{symbol} @ {train_interval}] Training produced no valid model state"
                    )

            except Exception as exc:
                logger.error(
                    f"[{symbol} @ {train_interval}] Training failed: {exc}"
                )
                logger.debug(traceback.format_exc())

            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        overall_results[symbol] = {
            "ready":    symbol_ready,
            "best_acc": symbol_best_acc,
        }

    # ------------------------------------------------------------------
    # 6. Final summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RETRAINING COMPLETE — SUMMARY")
    logger.info("=" * 70)

    ready_count = 0
    for symbol, result in overall_results.items():
        status = "READY" if result["ready"] else "BELOW TARGET"
        acc    = result["best_acc"]
        logger.info(f"  {symbol:15s}  |  {status:12s}  |  Best val acc: {acc:.2%}")
        if result["ready"]:
            ready_count += 1

    logger.info(f"\n{ready_count}/{len(overall_results)} symbols passed validation threshold")
    logger.info("=" * 70)


if __name__ == "__main__":
    retrain_all_models()
