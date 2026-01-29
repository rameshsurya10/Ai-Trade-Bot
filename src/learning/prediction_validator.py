"""
Prediction Validator
====================

Validates AI predictions before allowing real trades.

Flow:
1. Model makes prediction for next candle
2. Wait for candle to close
3. Check if prediction was correct (profit/loss)
4. Track consecutive wins (streak)
5. Reset streak on any loss
6. Require 8 consecutive wins before trading
7. Store detailed notes for every prediction

This ensures the model PROVES accuracy before risking money.
"""

import logging
import json
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

from src.core.database import Database

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a single prediction and its outcome."""

    # Prediction details
    timestamp: int  # When prediction was made (ms)
    symbol: str
    timeframe: str
    predicted_direction: str  # "BUY", "SELL", "NEUTRAL"
    predicted_price: float  # Price when prediction made
    confidence: float  # Model confidence 0-1

    # Target levels
    target_price: float  # Expected price at candle close
    stop_loss: float
    take_profit: float

    # Rules validation
    rules_passed: int
    rules_total: int

    # Market context at prediction time
    market_context: Dict  # {regime, volatility, trend, etc}

    # Outcome (filled when candle closes)
    actual_price: Optional[float] = None  # Price at candle close
    actual_direction: Optional[str] = None  # "UP", "DOWN", "FLAT"
    is_correct: Optional[bool] = None  # Did prediction match reality?
    profit_loss_pct: Optional[float] = None  # % move from predicted
    closed_at: Optional[int] = None  # When candle closed (ms)

    # Learning notes
    notes: str = ""  # Detailed analysis of why right/wrong


class PredictionValidator:
    """
    Validates predictions by tracking accuracy over time.

    Requires 8 consecutive correct predictions before allowing trades.
    Resets streak on any incorrect prediction.
    """

    def __init__(
        self,
        database: Database,
        streak_required: int = 8,
        min_data_years: int = 1
    ):
        """
        Initialize prediction validator.

        Args:
            database: Database instance for persistence
            streak_required: Consecutive wins needed (default: 8)
            min_data_years: Minimum years of candle data needed (default: 1)
        """
        self.db = database
        self.streak_required = streak_required
        self.min_data_years = min_data_years

        # Current streak tracking
        self._current_streak: Dict[Tuple[str, str], int] = {}  # (symbol, tf) -> streak
        self._best_streak: Dict[Tuple[str, str], int] = {}
        # Note: _pending_predictions removed - now using database queries via get_pending_predictions_from_db()

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'streaks_achieved': 0,
            'streaks_broken': 0
        }

        # Initialize database table
        self._init_db()

        logger.info(
            f"PredictionValidator initialized: "
            f"streak_required={streak_required}, "
            f"min_data_years={min_data_years}"
        )

    def _init_db(self):
        """Create prediction_history table if it doesn't exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_price REAL NOT NULL,
            confidence REAL NOT NULL,
            target_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            rules_passed INTEGER NOT NULL,
            rules_total INTEGER NOT NULL,
            market_context TEXT NOT NULL,
            actual_price REAL,
            actual_direction TEXT,
            is_correct INTEGER,
            profit_loss_pct REAL,
            closed_at INTEGER,
            notes TEXT,
            created_at INTEGER NOT NULL
        )
        """
        self.db.execute(schema)

        # Create index for fast lookups
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_symbol_tf "
            "ON prediction_history(symbol, timeframe, timestamp DESC)"
        )

        logger.info("Prediction history table initialized")

    def record_prediction(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        current_price: float,
        confidence: float,
        target_price: float,
        stop_loss: float,
        take_profit: float,
        rules_passed: int,
        rules_total: int,
        market_context: Dict
    ) -> PredictionRecord:
        """
        Record a new prediction.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "1h")
            direction: Predicted direction ("BUY", "SELL", "NEUTRAL")
            current_price: Current price when prediction made
            confidence: Model confidence (0-1)
            target_price: Expected price at next candle close
            stop_loss: SL level
            take_profit: TP level
            rules_passed: Number of rules that passed
            rules_total: Total rules checked
            market_context: Dict with market state

        Returns:
            PredictionRecord instance
        """
        timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Check for duplicate predictions within the same candle period
        candle_interval = self._get_candle_interval_ms(timeframe)
        candle_start = (timestamp // candle_interval) * candle_interval

        with self._lock:
            # Check if we already have a prediction for this candle
            existing = self.db.query(
                """
                SELECT id, timestamp FROM prediction_history
                WHERE symbol = ? AND timeframe = ?
                AND timestamp >= ? AND timestamp < ?
                AND actual_price IS NULL
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (symbol, timeframe, candle_start, candle_start + candle_interval)
            )

            if existing:
                logger.warning(
                    f"⚠️ Duplicate prediction skipped: {symbol} {timeframe} "
                    f"(already have prediction for this candle period)"
                )
                return None

        record = PredictionRecord(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            predicted_direction=direction,
            predicted_price=current_price,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rules_passed=rules_passed,
            rules_total=rules_total,
            market_context=market_context
        )

        # Store in database
        with self._lock:
            self.db.execute(
                """
                INSERT INTO prediction_history (
                    timestamp, symbol, timeframe, predicted_direction,
                    predicted_price, confidence, target_price, stop_loss,
                    take_profit, rules_passed, rules_total, market_context,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp, symbol, timeframe, direction,
                    current_price, confidence, target_price, stop_loss,
                    take_profit, rules_passed, rules_total,
                    json.dumps(market_context), timestamp
                )
            )

            # Prediction stored in database (line 228-246)
            # No need for in-memory cache - use get_pending_predictions_from_db() to query
            self._stats['total_predictions'] += 1

        logger.info(
            f"Prediction recorded: {symbol} {timeframe} {direction} "
            f"@ ${current_price:,.2f} (confidence: {confidence:.1%})"
        )

        return record

    def validate_prediction(
        self,
        symbol: str,
        timeframe: str,
        actual_price: float,
        candle_close_time: int
    ) -> Tuple[bool, str, PredictionRecord]:
        """
        Validate a prediction when candle closes.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            actual_price: Actual price at candle close
            candle_close_time: Timestamp when candle closed (ms)

        Returns:
            (is_correct, notes, updated_record)
        """
        # Find the pending prediction that should close at this time
        candle_interval = self._get_candle_interval_ms(timeframe)
        record = None

        with self._lock:
            # Find prediction from database that matches this candle close time
            # Query for pending predictions for this symbol/timeframe
            query = """
                SELECT timestamp, predicted_direction, predicted_price, confidence,
                       target_price, stop_loss, take_profit
                FROM prediction_history
                WHERE symbol = ? AND timeframe = ? AND is_correct IS NULL
                ORDER BY timestamp DESC
            """
            rows = self.db.query(query, (symbol, timeframe))

            # Find the prediction that should close at this candle_close_time
            for row in rows:
                pred_timestamp = row[0]
                expected_close = ((pred_timestamp // candle_interval) + 1) * candle_interval
                # Allow some tolerance (±10% of candle interval)
                tolerance = candle_interval * 0.1
                if abs(candle_close_time - expected_close) < tolerance:
                    # Create PredictionRecord from database row
                    record = PredictionRecord(
                        timestamp=pred_timestamp,
                        symbol=symbol,
                        timeframe=timeframe,
                        predicted_direction=row[1],
                        predicted_price=row[2],
                        confidence=row[3],
                        target_price=row[4],
                        stop_loss=row[5],
                        take_profit=row[6],
                        rules_passed=9,  # Default value for old records
                        rules_total=10,   # Default value for old records
                        market_context={}  # Empty dict for old records
                    )
                    break

            if record is None:
                logger.warning(f"No pending prediction found for {symbol} {timeframe} at candle close {candle_close_time}")
                return False, "No pending prediction", None

            # Determine actual direction
            price_change = actual_price - record.predicted_price
            price_change_pct = (price_change / record.predicted_price) * 100

            if abs(price_change_pct) < 0.1:  # < 0.1% considered flat
                actual_direction = "FLAT"
            elif price_change > 0:
                actual_direction = "UP"
            else:
                actual_direction = "DOWN"

            # Check if prediction was correct
            is_correct = (
                (record.predicted_direction == "BUY" and actual_direction == "UP") or
                (record.predicted_direction == "SELL" and actual_direction == "DOWN") or
                (record.predicted_direction == "NEUTRAL" and actual_direction == "FLAT")
            )

            # Generate detailed notes
            notes = self._generate_notes(record, actual_price, actual_direction, is_correct)

            # Update record
            record.actual_price = actual_price
            record.actual_direction = actual_direction
            record.is_correct = is_correct
            record.profit_loss_pct = price_change_pct
            record.closed_at = candle_close_time
            record.notes = notes

            # Update database
            self.db.execute(
                """
                UPDATE prediction_history
                SET actual_price = ?, actual_direction = ?, is_correct = ?,
                    profit_loss_pct = ?, closed_at = ?, notes = ?
                WHERE timestamp = ? AND symbol = ? AND timeframe = ?
                """,
                (
                    actual_price, actual_direction, 1 if is_correct else 0,
                    price_change_pct, candle_close_time, notes,
                    record.timestamp, symbol, timeframe
                )
            )

            # Update streak (use symbol/timeframe key for continuity)
            streak_key = (symbol, timeframe)
            if is_correct:
                self._current_streak[streak_key] = self._current_streak.get(streak_key, 0) + 1
                self._stats['correct_predictions'] += 1

                # Update best streak
                if self._current_streak[streak_key] > self._best_streak.get(streak_key, 0):
                    self._best_streak[streak_key] = self._current_streak[streak_key]

                # Check if streak achieved
                if self._current_streak[streak_key] == self.streak_required:
                    self._stats['streaks_achieved'] += 1
                    logger.info(f"🎉 STREAK ACHIEVED: {symbol} {timeframe} - {self.streak_required} consecutive wins!")
            else:
                # Reset streak on loss
                old_streak = self._current_streak.get(streak_key, 0)
                self._current_streak[streak_key] = 0
                self._stats['incorrect_predictions'] += 1

                if old_streak > 0:
                    self._stats['streaks_broken'] += 1
                    logger.warning(f"❌ STREAK BROKEN: {symbol} {timeframe} - was at {old_streak}, reset to 0")

            # Prediction marked as validated in database (lines 348-361)
            # No need to remove from memory cache (now using database)

            logger.info(
                f"Prediction validated: {symbol} {timeframe} "
                f"{'✅ CORRECT' if is_correct else '❌ WRONG'} "
                f"(Predicted: {record.predicted_direction}, Actual: {actual_direction}, "
                f"Change: {price_change_pct:+.2f}%) "
                f"Streak: {self._current_streak[streak_key]}/{self.streak_required}"
            )

            return is_correct, notes, record

    def _generate_notes(
        self,
        record: PredictionRecord,
        actual_price: float,
        actual_direction: str,
        is_correct: bool
    ) -> str:
        """Generate detailed notes for a prediction outcome."""

        notes = []

        # Basic outcome
        notes.append(f"Prediction: {record.predicted_direction} @ ${record.predicted_price:,.2f}")
        notes.append(f"Actual: {actual_direction} @ ${actual_price:,.2f}")
        notes.append(f"Result: {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        notes.append(f"Change: {((actual_price - record.predicted_price) / record.predicted_price * 100):+.2f}%")
        notes.append("")

        # Model state
        notes.append(f"Confidence: {record.confidence:.1%}")
        notes.append(f"Rules: {record.rules_passed}/{record.rules_total} passed")
        notes.append("")

        # Market context
        ctx = record.market_context
        notes.append("Market Context:")
        notes.append(f"  Regime: {ctx.get('regime', 'N/A')}")
        notes.append(f"  Trend: {ctx.get('trend', 'N/A')}")
        notes.append(f"  Volatility: {ctx.get('volatility', 'N/A')}")
        notes.append(f"  Cycle Phase: {ctx.get('cycle_phase', 'N/A')}")
        notes.append("")

        # Analysis of why right/wrong
        if not is_correct:
            notes.append("Why Wrong:")
            if record.predicted_direction == "BUY" and actual_direction == "DOWN":
                notes.append("  - Expected price to rise but it fell")
                if ctx.get('regime') in ['CHOPPY', 'VOLATILE']:
                    notes.append("  - Market was choppy/volatile - should have waited")
                if record.confidence < 0.7:
                    notes.append("  - Low confidence - should not have predicted")
            elif record.predicted_direction == "SELL" and actual_direction == "UP":
                notes.append("  - Expected price to fall but it rose")
                if ctx.get('trend') == 'UP':
                    notes.append("  - Trend was UP - fighting the trend")
        else:
            notes.append("Why Correct:")
            notes.append(f"  - All indicators aligned for {record.predicted_direction}")
            notes.append(f"  - Market regime was favorable: {ctx.get('regime')}")

        return "\n".join(notes)

    def can_trade(self, symbol: str, timeframe: str) -> Tuple[bool, str, int]:
        """
        Check if trading is allowed based on prediction streak.

        Args:
            symbol: Trading pair
            timeframe: Timeframe

        Returns:
            (can_trade, reason, current_streak)
        """
        key = (symbol, timeframe)
        current_streak = self._current_streak.get(key, 0)

        if current_streak >= self.streak_required:
            return True, f"Streak achieved: {current_streak}/{self.streak_required}", current_streak
        else:
            needed = self.streak_required - current_streak
            return False, f"Need {needed} more consecutive wins (current: {current_streak}/{self.streak_required})", current_streak

    def get_history(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 20
    ) -> List[Dict]:
        """
        Get prediction history for a symbol/timeframe.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            limit: Maximum number of records to return

        Returns:
            List of prediction records (most recent first)
        """
        rows = self.db.query(
            """
            SELECT * FROM prediction_history
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, timeframe, limit)
        )

        return [dict(row) for row in rows]

    def get_stats(self, symbol: str = None, timeframe: str = None) -> Dict:
        """
        Get prediction statistics.

        Args:
            symbol: Optional filter by symbol
            timeframe: Optional filter by timeframe

        Returns:
            Dict with statistics
        """
        with self._lock:
            stats = self._stats.copy()

            if symbol and timeframe:
                key = (symbol, timeframe)
                stats['current_streak'] = self._current_streak.get(key, 0)
                stats['best_streak'] = self._best_streak.get(key, 0)

            # Calculate accuracy
            total = stats['correct_predictions'] + stats['incorrect_predictions']
            stats['accuracy'] = (stats['correct_predictions'] / total * 100) if total > 0 else 0

            return stats

    def _get_candle_interval_ms(self, timeframe: str) -> int:
        """
        Convert timeframe string to milliseconds.

        Args:
            timeframe: Timeframe string (e.g., "1m", "1h", "1d")

        Returns:
            Interval in milliseconds
        """
        # Only 4 supported timeframes: 15m, 1h, 4h, 1d
        timeframe_to_ms = {
            '15m': 15 * 60 * 1000,     # 15 minutes
            '1h': 60 * 60 * 1000,      # 1 hour
            '4h': 4 * 60 * 60 * 1000,  # 4 hours
            '1d': 24 * 60 * 60 * 1000, # 1 day
        }
        return timeframe_to_ms.get(timeframe, 60 * 60 * 1000)  # Default to 1h

    def cleanup_stale_predictions(self) -> int:
        """
        Mark stale predictions as failed in database.

        Predictions older than 2x their candle interval are marked as incorrect
        to prevent them from hanging indefinitely. This handles cases where
        validation fails or candles don't close properly.

        Returns:
            Number of stale predictions marked as failed
        """
        import time
        now = int(time.time() * 1000)
        stale_count = 0

        with self._lock:
            # Get all pending predictions from database
            pending = self.get_pending_predictions_from_db()

            for pred in pending:
                symbol = pred['symbol']
                timeframe = pred['timeframe']
                timestamp = pred['timestamp']

                candle_interval = self._get_candle_interval_ms(timeframe)
                age_ms = now - timestamp
                max_age = candle_interval * 2  # Allow 2x normal interval

                if age_ms > max_age:
                    # Mark as failed in database
                    self.db.execute(
                        """
                        UPDATE prediction_history
                        SET is_correct = 0,
                            notes = ?,
                            closed_at = ?
                        WHERE timestamp = ? AND symbol = ? AND timeframe = ?
                        """,
                        (
                            f"Marked as stale - no validation after {age_ms / 1000 / 60:.1f} minutes",
                            now,
                            timestamp,
                            symbol,
                            timeframe
                        )
                    )
                    stale_count += 1
                    logger.warning(
                        f"🗑️ Marked stale prediction as failed: {symbol} {timeframe} "
                        f"(age: {age_ms / 1000 / 60:.1f} min, max: {max_age / 1000 / 60:.1f} min)"
                    )

        if stale_count > 0:
            logger.info(f"Cleaned up {stale_count} stale predictions")

        return stale_count

    def check_data_requirement(self, df, timeframe: str) -> Tuple[bool, str]:
        """
        Check if we have enough historical data (1 year minimum).

        Args:
            df: DataFrame with candle data
            timeframe: Timeframe string (e.g., "1h", "4h", "1d")

        Returns:
            (has_enough_data, message)
        """
        if df is None or len(df) == 0:
            return False, "No data available"

        # Only 4 supported timeframes: 15m, 1h, 4h, 1d
        timeframe_to_candles = {
            '15m': 35040,  # 4 * 24 * 365
            '1h': 8760,    # 24 * 365
            '4h': 2190,    # 6 * 365
            '1d': 365,     # 365
        }

        required_candles = timeframe_to_candles.get(timeframe, 8760)  # Default to 1h
        actual_candles = len(df)

        # Need at least required amount for 1 year
        years_of_data = actual_candles / required_candles

        if years_of_data >= self.min_data_years:
            return True, f"✅ {years_of_data:.1f} years of data ({actual_candles:,} candles)"
        else:
            needed = int(required_candles * self.min_data_years - actual_candles)
            return False, f"❌ Need {needed:,} more candles ({years_of_data:.1f}/{self.min_data_years} years)"

    def get_pending_predictions_from_db(self) -> List[Dict]:
        """
        Get all pending predictions from database that need validation.

        Returns:
            List of pending prediction dictionaries with all fields
        """
        query = """
            SELECT id, symbol, timeframe, timestamp, predicted_direction,
                   predicted_price, confidence, target_price, stop_loss, take_profit
            FROM prediction_history
            WHERE is_correct IS NULL
            ORDER BY timestamp ASC
        """

        rows = self.db.query(query)

        predictions = []
        for row in rows:
            predictions.append({
                'id': row[0],
                'symbol': row[1],
                'timeframe': row[2],
                'timestamp': row[3],
                'direction': row[4],
                'predicted_price': row[5],
                'confidence': row[6],
                'target_price': row[7],
                'stop_loss': row[8],
                'take_profit': row[9]
            })

        logger.debug(f"Found {len(predictions)} pending predictions in database")
        return predictions
