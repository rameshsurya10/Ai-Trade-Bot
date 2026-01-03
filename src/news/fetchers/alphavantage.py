"""
Alpha Vantage News Fetcher
===========================

Fetch news from Alpha Vantage News & Sentiment API

Free tier: 25 requests/day
Strategy: Fetch every 2 hours (12 requests/day) or on-demand

API Documentation: https://www.alphavantage.co/documentation/#news-sentiment
"""

import logging
import os
import requests
from typing import List, Optional
from datetime import datetime, timedelta

from .base import BaseFetcher
from ..types import Article

logger = logging.getLogger(__name__)


class AlphaVantageFetcher(BaseFetcher):
    """
    Fetch news from Alpha Vantage News & Sentiment API.

    Free tier limitations:
    - 25 requests per day
    - 1000 news items per request max
    - Includes sentiment scores from Alpha Vantage

    Strategy:
    - Fetch every 2 hours (12 requests/day)
    - Use NEWS_SENTIMENT function
    - Filter by tickers
    """

    def __init__(self, config: dict = None):
        """
        Initialize Alpha Vantage fetcher.

        Args:
            config: Configuration dict from config.yaml

        Environment Variables:
            ALPHAVANTAGE_KEY: Alpha Vantage API key
        """
        super().__init__(config)

        # Get API key from environment
        self.api_key = os.getenv('ALPHAVANTAGE_KEY')
        if not self.api_key:
            logger.warning("ALPHAVANTAGE_KEY not set - Alpha Vantage fetcher will not work")

        # API endpoint
        self.base_url = "https://www.alphavantage.co/query"

        # Configuration
        self.enabled = self.config.get('enabled', True)

        # Rate limiting
        self.max_requests_per_day = self.config.get('max_requests_per_day', 25)
        self._requests_today = 0
        self._last_reset_date = datetime.utcnow().date()

    def fetch(
        self,
        symbols: List[str],
        lookback_hours: int = 6,
        max_articles: int = 50
    ) -> List[Article]:
        """
        Fetch news articles from Alpha Vantage.

        Args:
            symbols: List of symbols (e.g., ['CRYPTO:BTC', 'CRYPTO:ETH'])
            lookback_hours: Hours to look back
            max_articles: Maximum articles to return

        Returns:
            List of Article objects

        Raises:
            Exception: If API request fails
        """
        if not self.enabled:
            logger.debug("Alpha Vantage fetcher is disabled")
            return []

        if not self.api_key:
            logger.error("Alpha Vantage API key not configured")
            return []

        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Alpha Vantage daily rate limit reached")
            return []

        try:
            # Convert symbols to Alpha Vantage format
            tickers = self._convert_symbols_to_tickers(symbols)

            # Calculate time range
            from_time = datetime.utcnow() - timedelta(hours=lookback_hours)
            to_time = datetime.utcnow()

            # API parameters
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ','.join(tickers),
                'time_from': from_time.strftime('%Y%m%dT%H%M'),
                'time_to': to_time.strftime('%Y%m%dT%H%M'),
                'limit': min(max_articles, 1000),  # Max 1000 per request
                'apikey': self.api_key
            }

            logger.debug(f"Fetching Alpha Vantage: tickers={tickers}, lookback={lookback_hours}h")

            # Make request
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                raise Exception(f"Alpha Vantage error: {data['Error Message']}")

            if 'Note' in data:
                # Rate limit message
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return []

            feed = data.get('feed', [])
            logger.info(f"Alpha Vantage returned {len(feed)} articles")

            # Convert to Article objects
            parsed_articles = []
            for article_data in feed:
                try:
                    article = self._parse_article(article_data, symbols)
                    if article:
                        parsed_articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue

            # Update stats
            self._update_stats(len(parsed_articles))
            self._requests_today += 1

            return parsed_articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            self._update_stats(0, error=True)
            return []

        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed: {e}", exc_info=True)
            self._update_stats(0, error=True)
            return []

    def _convert_symbols_to_tickers(self, symbols: List[str]) -> List[str]:
        """
        Convert trading symbols to Alpha Vantage ticker format.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])

        Returns:
            List of Alpha Vantage tickers (e.g., ['CRYPTO:BTC', 'CRYPTO:ETH'])
        """
        tickers = []
        for symbol in symbols:
            # Remove /USDT, /USD etc.
            clean_symbol = symbol.split('/')[0] if '/' in symbol else symbol

            # Add CRYPTO: prefix
            ticker = f"CRYPTO:{clean_symbol}"
            tickers.append(ticker)

        return tickers

    def _parse_article(self, article_data: dict, symbols: List[str]) -> Optional[Article]:
        """
        Parse Alpha Vantage article data into Article object.

        Args:
            article_data: Raw article data from Alpha Vantage
            symbols: List of symbols to extract

        Returns:
            Article object or None if parsing fails
        """
        try:
            # Parse published date
            time_published = article_data.get('time_published')
            if time_published:
                # Format: YYYYMMDDTHHMMSS
                dt = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                timestamp = int(dt.timestamp())
                datetime_str = dt.isoformat()
            else:
                timestamp = int(datetime.utcnow().timestamp())
                datetime_str = datetime.utcnow().isoformat()

            # Extract basic info
            title = article_data.get('title', '')
            summary = article_data.get('summary', '')
            url = article_data.get('url', '')

            # Extract text for symbol matching
            full_text = f"{title} {summary}"

            # Find mentioned symbols
            mentioned_symbols = self._extract_symbols(full_text, symbols)

            # Also check ticker_sentiment
            ticker_sentiment = article_data.get('ticker_sentiment', [])
            for ticker_data in ticker_sentiment:
                ticker = ticker_data.get('ticker', '')
                # Remove CRYPTO: prefix
                if ticker.startswith('CRYPTO:'):
                    symbol = ticker.replace('CRYPTO:', '')
                    if symbol not in mentioned_symbols and symbol in symbols:
                        mentioned_symbols.append(symbol)

            # Skip if no relevant symbols
            if not mentioned_symbols:
                return None

            # Determine primary symbol (highest relevance or first mentioned)
            primary_symbol = self._get_primary_symbol(ticker_sentiment, mentioned_symbols)

            # Extract Alpha Vantage sentiment (if available)
            av_sentiment = None
            av_sentiment_label = None
            av_sentiment_score = None

            for ticker_data in ticker_sentiment:
                ticker = ticker_data.get('ticker', '')
                if ticker == f"CRYPTO:{primary_symbol}":
                    av_sentiment_label = ticker_data.get('ticker_sentiment_label')  # Bullish, Bearish, Neutral
                    av_sentiment_score = float(ticker_data.get('ticker_sentiment_score', 0.0))

                    # Convert to -1.0 to 1.0 scale
                    if av_sentiment_label == 'Bullish':
                        av_sentiment = abs(av_sentiment_score)
                    elif av_sentiment_label == 'Bearish':
                        av_sentiment = -abs(av_sentiment_score)
                    else:
                        av_sentiment = 0.0
                    break

            # Create Article
            article = Article(
                timestamp=timestamp,
                datetime=datetime_str,
                source='alphavantage',
                title=title,
                description=summary,
                content=None,  # Alpha Vantage doesn't provide full content
                url=url,
                symbols=mentioned_symbols,
                primary_symbol=primary_symbol,
                processed=True if av_sentiment is not None else False,  # Already has sentiment
                sentiment_score=av_sentiment,
                sentiment_label=av_sentiment_label,
                sentiment_compound=av_sentiment_score
            )

            # Calculate relevance score
            if primary_symbol:
                # Use Alpha Vantage relevance score if available
                for ticker_data in ticker_sentiment:
                    ticker = ticker_data.get('ticker', '')
                    if ticker == f"CRYPTO:{primary_symbol}":
                        relevance = float(ticker_data.get('relevance_score', 0.0))
                        article.relevance_score = relevance
                        break

                # Fallback to our own calculation
                if article.relevance_score is None:
                    article.relevance_score = self._calculate_relevance_score(article, primary_symbol)

            # Create content hash for deduplication
            article.content_hash = self._create_content_hash(article_data)

            return article

        except Exception as e:
            logger.warning(f"Failed to parse Alpha Vantage article: {e}")
            return None

    def _get_primary_symbol(
        self,
        ticker_sentiment: list,
        mentioned_symbols: List[str]
    ) -> Optional[str]:
        """
        Get primary symbol from ticker sentiment data.

        Args:
            ticker_sentiment: List of ticker sentiment dicts
            mentioned_symbols: List of mentioned symbols

        Returns:
            Primary symbol or None
        """
        if not mentioned_symbols:
            return None

        # Try to find symbol with highest relevance
        max_relevance = 0.0
        primary = mentioned_symbols[0]

        for ticker_data in ticker_sentiment:
            ticker = ticker_data.get('ticker', '')
            if ticker.startswith('CRYPTO:'):
                symbol = ticker.replace('CRYPTO:', '')
                if symbol in mentioned_symbols:
                    relevance = float(ticker_data.get('relevance_score', 0.0))
                    if relevance > max_relevance:
                        max_relevance = relevance
                        primary = symbol

        return primary

    def _check_rate_limit(self) -> bool:
        """
        Check if we've hit the daily rate limit.

        Returns:
            True if we can make more requests, False otherwise
        """
        # Reset counter if new day
        current_date = datetime.utcnow().date()
        if current_date > self._last_reset_date:
            self._requests_today = 0
            self._last_reset_date = current_date

        return self._requests_today < self.max_requests_per_day
