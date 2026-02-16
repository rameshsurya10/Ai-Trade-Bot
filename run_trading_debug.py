#!/usr/bin/env python3
"""Debug runner - logs to file for monitoring."""
import sys
import logging

# Force all logging to a file AND stderr
log_file = 'data/trading_debug.log'
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

from src.live_trading.runner import LiveTradingRunner, TradingMode

def main():
    runner = LiveTradingRunner('config.yaml', mode=TradingMode.PAPER)
    runner.add_symbol('BTC/USDT', exchange='binance', interval='1h')
    runner.add_symbol('ETH/USDT', exchange='binance', interval='1h')

    print(f"Starting... logs going to {log_file}", flush=True)

    try:
        runner.start(blocking=True)
    except KeyboardInterrupt:
        runner.stop()
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
