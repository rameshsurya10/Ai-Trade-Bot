#!/usr/bin/env python3
"""
Download Historical Data for Training
=====================================

Downloads historical OHLCV data and saves to database.

Usage:
    python scripts/download_data.py

Options:
    --days 365      Number of days to download (default: 365)
    --symbol BTC-USD    Trading pair (default: from config)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

from src.data_service import DataService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Download historical data')
    parser.add_argument('--days', type=int, default=365,
                       help='Number of days to download')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("DOWNLOADING HISTORICAL DATA")
    logger.info("="*60)

    # Initialize data service
    service = DataService(args.config)

    logger.info(f"Symbol: {service.symbol}")
    logger.info(f"Interval: {service.interval}")
    logger.info(f"Days: {args.days}")

    # Download data
    df = service.fetch_historical_data(days=args.days)

    if df.empty:
        logger.error("No data downloaded!")
        sys.exit(1)

    # Save to database
    service.save_candles(df)

    # Verify
    saved = service.get_candles(limit=100000)

    logger.info("="*60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*60)
    logger.info(f"Total candles: {len(saved)}")
    logger.info(f"Date range: {saved['datetime'].min()} to {saved['datetime'].max()}")
    logger.info(f"Database: {service.db_path}")


if __name__ == "__main__":
    main()
