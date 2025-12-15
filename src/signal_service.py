"""
Signal Service - Trading Signal Management
==========================================
Receives predictions from Analysis Engine.
Filters, validates, and forwards actionable signals.
NO AUTO-TRADING - just signals for YOUR manual trading.
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List

import yaml

logger = logging.getLogger(__name__)


class SignalService:
    """
    Signal management service.

    - Receives predictions from AnalysisEngine
    - Filters signals based on confidence thresholds
    - Stores signal history in database
    - Forwards actionable signals to Notifier
    - NO AUTO-TRADING - just alerts for you
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize signal service."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Signal thresholds (from config)
        self.min_confidence = self.config['analysis']['min_confidence']
        self.strong_signal = self.config['signals']['strong_signal']
        self.medium_signal = self.config['signals']['medium_signal']

        # Risk management info (for YOUR manual trading)
        self.risk_per_trade = self.config['signals']['risk_per_trade']
        self.risk_reward_ratio = self.config['signals']['risk_reward_ratio']

        # Cooldown from config (prevent spam)
        self._signal_cooldown_minutes = self.config['signals'].get('cooldown_minutes', 60)

        # Database
        self.db_path = Path(self.config['database']['path'])

        # State
        self._last_signal: Optional[dict] = None
        self._last_signal_time: Optional[datetime] = None
        self._callbacks: List[Callable] = []

        logger.info("SignalService initialized")

    def on_prediction(self, prediction: dict):
        """
        Process prediction from Analysis Engine.

        Args:
            prediction: Dict with signal, confidence, price, etc.
        """
        signal = prediction.get('signal', 'NEUTRAL')
        confidence = prediction.get('confidence', 0)
        price = prediction.get('price', 0)

        # Skip neutral/wait signals
        if signal in ['NEUTRAL', 'WAIT']:
            logger.debug(f"Skipping {signal} signal")
            return

        # Check confidence threshold
        if confidence < self.min_confidence:
            logger.debug(f"Low confidence ({confidence:.2%}), skipping")
            return

        # Check cooldown (prevent spam)
        if self._should_skip_cooldown(signal):
            logger.debug("Signal in cooldown period, skipping")
            return

        # Determine signal strength
        if confidence >= self.strong_signal:
            strength = 'STRONG'
        elif confidence >= self.medium_signal:
            strength = 'MEDIUM'
        else:
            strength = 'WEAK'

        # Build actionable signal
        actionable_signal = {
            'timestamp': prediction.get('timestamp', datetime.utcnow()),
            'signal': signal.replace('WEAK_', ''),  # Normalize to BUY/SELL
            'strength': strength,
            'confidence': confidence,
            'price': price,
            'stop_loss': prediction.get('stop_loss'),
            'take_profit': prediction.get('take_profit'),
            'atr': prediction.get('atr'),
            'risk_per_trade': self.risk_per_trade,
            'risk_reward_ratio': self.risk_reward_ratio,
            'indicators': {
                'rsi': prediction.get('rsi'),
                'macd_hist': prediction.get('macd_hist'),
                'bb_position': prediction.get('bb_position')
            }
        }

        # Store in database
        self._save_signal(actionable_signal)

        # Update state
        self._last_signal = actionable_signal
        self._last_signal_time = datetime.utcnow()

        # Log
        logger.info(
            f"🚨 {strength} {signal} SIGNAL @ ${price:.2f} "
            f"(Confidence: {confidence:.1%})"
        )

        # Forward to notifier
        for callback in self._callbacks:
            try:
                callback(actionable_signal)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _should_skip_cooldown(self, new_signal: str) -> bool:
        """Check if we should skip due to cooldown."""
        if self._last_signal is None:
            return False

        # Same direction signal?
        last_direction = self._last_signal.get('signal', '').replace('WEAK_', '')
        new_direction = new_signal.replace('WEAK_', '')

        if last_direction != new_direction:
            return False

        # Check time since last signal
        if self._last_signal_time:
            time_diff = (datetime.utcnow() - self._last_signal_time).total_seconds()
            if time_diff < self._signal_cooldown_minutes * 60:
                return True

        return False

    def _save_signal(self, signal: dict):
        """Save signal to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = signal.get('timestamp')
            if hasattr(timestamp, 'timestamp'):
                ts = int(timestamp.timestamp() * 1000)
                dt = timestamp.isoformat()
            else:
                ts = int(datetime.utcnow().timestamp() * 1000)
                dt = str(timestamp)

            cursor.execute('''
                INSERT INTO signals
                (timestamp, datetime, signal, confidence, price, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                ts,
                dt,
                f"{signal['strength']}_{signal['signal']}",
                signal['confidence'],
                signal['price'],
                signal.get('stop_loss'),
                signal.get('take_profit')
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving signal: {e}")

    def register_callback(self, callback: Callable):
        """Register callback for actionable signals."""
        self._callbacks.append(callback)

    def get_signal_history(self, limit: int = 50) -> List[dict]:
        """Get recent signal history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT datetime, signal, confidence, price, stop_loss, take_profit
                FROM signals
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'datetime': row[0],
                    'signal': row[1],
                    'confidence': row[2],
                    'price': row[3],
                    'stop_loss': row[4],
                    'take_profit': row[5]
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    def get_last_signal(self) -> Optional[dict]:
        """Get most recent signal."""
        return self._last_signal

    def format_signal_message(self, signal: dict) -> str:
        """
        Format signal as readable message.

        Returns message you can act on manually.
        """
        direction = signal['signal']
        strength = signal['strength']
        price = signal['price']
        confidence = signal['confidence']
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')

        # Emoji for strength
        emoji = '🔴' if direction == 'SELL' else '🟢'
        strength_emoji = '💪' if strength == 'STRONG' else '👍' if strength == 'MEDIUM' else '👌'

        message = f"""
{emoji} {strength} {direction} SIGNAL {strength_emoji}

📊 Entry Price: ${price:,.2f}
🎯 Confidence: {confidence:.1%}

💡 SUGGESTED LEVELS (for your manual trade):
   🛑 Stop Loss: ${stop_loss:,.2f}
   ✅ Take Profit: ${take_profit:,.2f}
   📏 Risk:Reward = 1:{self.risk_reward_ratio:.1f}

⚠️ REMINDER:
   • Risk only {self.risk_per_trade:.0%} of your capital
   • These are SUGGESTIONS - make your own decision
   • Never trade more than you can afford to lose

⏰ Time: {signal.get('timestamp', datetime.utcnow())}
"""
        return message.strip()

    def get_status(self) -> dict:
        """Get service status."""
        return {
            'last_signal': self._last_signal,
            'signal_cooldown_minutes': self._signal_cooldown_minutes,
            'min_confidence': self.min_confidence,
            'strong_threshold': self.strong_signal,
            'medium_threshold': self.medium_signal,
            'total_signals': len(self.get_signal_history(limit=1000))
        }


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    service = SignalService()

    # Test signal
    test_prediction = {
        'timestamp': datetime.utcnow(),
        'price': 50000,
        'signal': 'BUY',
        'confidence': 0.68,
        'stop_loss': 48000,
        'take_profit': 54000,
        'atr': 500,
        'rsi': 45,
        'macd_hist': 100,
        'bb_position': 0.4
    }

    service.on_prediction(test_prediction)

    print("\nFormatted Signal:")
    print(service.format_signal_message(service.get_last_signal()))
