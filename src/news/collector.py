"""
News Collector
==============

Main service for collecting news from multiple sources.

Coordinates:
- NewsAPI
- Alpha Vantage
- Future sources (Reddit, Twitter, etc.)

Handles:
- Deduplication
- Database storage
- Scheduled fetching
- Statistics tracking
"""

import logging
import threading
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from src.core.database import Database
from .fetchers import NewsAPIFetcher, AlphaVantageFetcher
from .types import Article

logger = logging.getLogger(__name__)


class NewsCollector:
    """
    Collect news from multiple sources and store in database.

    Thread-safe: Runs in background thread
    Automatic: Fetches on schedule
    Deduplication: Uses content hash to avoid duplicates
    """

    def __init__(
        self,
        database: Database,
        config: dict = None
    ):
        """
        Initialize news collector.

        Args:
            database: Database instance
            config: Configuration dict from config.yaml['news']
        """
        self.db = database
        self.config = config or {}

        # Initialize fetchers
        self.fetchers: Dict[str, any] = {}

        newsapi_config = self.config.get('sources', {}).get('newsapi', {})
        if newsapi_config.get('enabled', True):
            self.fetchers['newsapi'] = NewsAPIFetcher(newsapi_config)

        alphavantage_config = self.config.get('sources', {}).get('alphavantage', {})
        if alphavantage_config.get('enabled', True):
            self.fetchers['alphavantage'] = AlphaVantageFetcher(alphavantage_config)

        # Scheduling
        self.enabled = self.config.get('enabled', True)
        self.fetch_interval = self.config.get('fetch_interval', 1800)  # 30 minutes
        self.lookback_hours = self.config.get('lookback_hours', 6)

        # Background thread
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Statistics
        self._stats_lock = threading.Lock()
        self._stats = {
            'total_articles_fetched': 0,
            'articles_stored': 0,
            'duplicates_skipped': 0,
            'fetch_cycles': 0,
            'errors': 0,
            'last_fetch_time': None
        }

        logger.info(
            f"NewsCollector initialized: "
            f"{len(self.fetchers)} fetchers, "
            f"interval={self.fetch_interval}s"
        )

    def start(self):
        """Start background news collection."""
        if not self.enabled:
            logger.info("News collection is disabled in config")
            return

        if self._running:
            logger.warning("NewsCollector already running")
            return

        self._running = True
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="NewsCollector"
        )
        self._thread.start()

        logger.info("NewsCollector started")

    def stop(self):
        """Stop background news collection."""
        if not self._running:
            return

        logger.info("Stopping NewsCollector...")
        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("NewsCollector stopped")

    def _collection_loop(self):
        """
        Main collection loop (runs in background thread).

        Fetches news on schedule and stores in database.
        """
        logger.info("News collection loop started")

        while self._running:
            try:
                # Fetch news from all sources
                self.fetch_and_store()

                # Wait for next cycle
                if self._stop_event.wait(timeout=self.fetch_interval):
                    # Stop event was set
                    break

            except Exception as e:
                logger.error(f"Error in collection loop: {e}", exc_info=True)
                with self._stats_lock:
                    self._stats['errors'] += 1

                # Wait a bit before retrying
                time.sleep(60)

        logger.info("News collection loop ended")

    def fetch_and_store(
        self,
        symbols: List[str] = None,
        lookback_hours: int = None
    ) -> int:
        """
        Fetch news from all sources and store in database.

        Args:
            symbols: List of symbols to fetch news for (None = from config)
            lookback_hours: Hours to look back (None = from config)

        Returns:
            Number of articles stored
        """
        if lookback_hours is None:
            lookback_hours = self.lookback_hours

        # Get symbols from config if not provided
        if symbols is None:
            symbols = self._get_symbols_from_config()

        if not symbols:
            logger.warning("No symbols configured for news collection")
            return 0

        logger.debug(f"Fetching news for symbols: {symbols}")

        all_articles = []
        fetcher_stats = {}

        # Fetch from all sources
        for source_name, fetcher in self.fetchers.items():
            try:
                articles = fetcher.fetch(
                    symbols=symbols,
                    lookback_hours=lookback_hours,
                    max_articles=100
                )

                all_articles.extend(articles)
                fetcher_stats[source_name] = len(articles)

                logger.debug(f"{source_name}: fetched {len(articles)} articles")

            except Exception as e:
                logger.error(f"Failed to fetch from {source_name}: {e}", exc_info=True)
                fetcher_stats[source_name] = 0

        logger.info(
            f"Fetched {len(all_articles)} articles total: {fetcher_stats}"
        )

        # Store in database (with deduplication)
        stored_count = self._store_articles(all_articles)

        # Update stats
        with self._stats_lock:
            self._stats['total_articles_fetched'] += len(all_articles)
            self._stats['articles_stored'] += stored_count
            self._stats['duplicates_skipped'] += (len(all_articles) - stored_count)
            self._stats['fetch_cycles'] += 1
            self._stats['last_fetch_time'] = datetime.utcnow().isoformat()

        return stored_count

    def _store_articles(self, articles: List[Article]) -> int:
        """
        Store articles in database with deduplication.

        Args:
            articles: List of Article objects

        Returns:
            Number of articles actually stored
        """
        stored_count = 0

        for article in articles:
            try:
                # Check for duplicate
                if self._is_duplicate(article):
                    logger.debug(f"Skipping duplicate: {article.title[:50]}...")
                    continue

                # Store in database (map Article fields to database method signature)
                self.db.save_news_article(
                    timestamp=article.timestamp,
                    source=article.source,
                    title=article.title,
                    description=article.description,
                    content=article.content,
                    url=article.url,
                    sentiment_score=article.sentiment_score,
                    sentiment_label=article.sentiment_label,
                    primary_symbol=article.primary_symbol,
                    content_hash=article.content_hash
                )
                stored_count += 1

            except Exception as e:
                logger.error(f"Failed to store article: {e}")
                continue

        logger.info(f"Stored {stored_count} new articles (skipped {len(articles) - stored_count} duplicates)")

        return stored_count

    def _is_duplicate(self, article: Article) -> bool:
        """
        Check if article is a duplicate based on content hash.

        Args:
            article: Article to check

        Returns:
            True if duplicate exists
        """
        if not article.content_hash:
            return False

        # Query database for existing article with same hash
        try:
            existing = self.db.get_news_by_hash(article.content_hash)
            return existing is not None
        except Exception as e:
            logger.warning(f"Failed to check for duplicate: {e}")
            return False

    def _get_symbols_from_config(self) -> List[str]:
        """
        Get symbols to track from config.

        Returns:
            List of symbols
        """
        # Try to get from live_trading config
        symbols = self.config.get('symbols', [])

        if not symbols:
            # Fallback to main config
            from src.core.config import load_config
            try:
                main_config = load_config()
                symbols = main_config.get('live_trading', {}).get('default_symbols', [])

                # Extract base symbols (remove /USDT, /USD, etc.)
                symbols = [s.split('/')[0] for s in symbols if '/' in s]
            except Exception as e:
                logger.warning(f"Failed to load symbols from config: {e}")

        return symbols

    def get_stats(self) -> dict:
        """Get news collector statistics (thread-safe)."""
        with self._stats_lock:
            stats = {
                **self._stats,
                'fetchers': {
                    name: fetcher.get_stats()
                    for name, fetcher in self.fetchers.items()
                },
                'running': self._running
            }

        return stats

    def get_recent_articles(
        self,
        symbol: str = None,
        hours: int = 24,
        limit: int = 100
    ) -> List[dict]:
        """
        Get recent articles from database.

        Args:
            symbol: Filter by symbol (None = all symbols)
            hours: Hours to look back
            limit: Maximum articles to return

        Returns:
            List of article dicts
        """
        try:
            since_timestamp = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())

            articles = self.db.get_news_articles(
                symbol=symbol,
                since_timestamp=since_timestamp,
                limit=limit,
                processed_only=False
            )

            return articles

        except Exception as e:
            logger.error(f"Failed to get recent articles: {e}")
            return []

    def manual_fetch(
        self,
        symbols: List[str],
        lookback_hours: int = 6
    ) -> int:
        """
        Manually trigger a fetch (for testing or on-demand updates).

        Args:
            symbols: List of symbols to fetch
            lookback_hours: Hours to look back

        Returns:
            Number of articles stored
        """
        logger.info(f"Manual fetch triggered for {symbols}")
        return self.fetch_and_store(symbols=symbols, lookback_hours=lookback_hours)
