"""
NewsAPI Fetcher
===============

Fetch news from NewsAPI.org

Free tier: 100 requests/day
Strategy: Fetch every 30 minutes (48 requests/day) or on-demand

API Documentation: https://newsapi.org/docs
"""

import logging
import os
import requests
from typing import List, Optional
from datetime import datetime, timedelta

from .base import BaseFetcher
from ..types import Article

logger = logging.getLogger(__name__)


class NewsAPIFetcher(BaseFetcher):
    """
    Fetch news from NewsAPI.org

    Free tier limitations:
    - 100 requests per day
    - No historical data older than 30 days
    - 100 articles per request max

    Strategy:
    - Fetch every 30 minutes (48 requests/day)
    - Use 'everything' endpoint for crypto news
    - Filter by symbol keywords
    """

    def __init__(self, config: dict = None):
        """
        Initialize NewsAPI fetcher.

        Args:
            config: Configuration dict from config.yaml

        Environment Variables:
            NEWSAPI_KEY: NewsAPI API key
        """
        super().__init__(config)

        # Get API key from environment
        self.api_key = os.getenv('NEWSAPI_KEY')
        if not self.api_key:
            logger.warning("NEWSAPI_KEY not set - NewsAPI fetcher will not work")

        # API endpoints
        self.base_url = "https://newsapi.org/v2"
        self.everything_endpoint = f"{self.base_url}/everything"

        # Configuration
        self.enabled = self.config.get('enabled', True)
        self.language = self.config.get('language', 'en')
        self.sort_by = self.config.get('sort_by', 'publishedAt')  # publishedAt, relevancy, popularity

        # Rate limiting
        self.max_requests_per_day = self.config.get('max_requests_per_day', 100)
        self._requests_today = 0
        self._last_reset_date = datetime.utcnow().date()

    def fetch(
        self,
        symbols: List[str],
        lookback_hours: int = 6,
        max_articles: int = 100
    ) -> List[Article]:
        """
        Fetch news articles from NewsAPI.

        Args:
            symbols: List of symbols (e.g., ['BTC', 'ETH'])
            lookback_hours: Hours to look back (max 720 hours / 30 days for free tier)
            max_articles: Maximum articles to return

        Returns:
            List of Article objects

        Raises:
            Exception: If API request fails
        """
        if not self.enabled:
            logger.debug("NewsAPI fetcher is disabled")
            return []

        if not self.api_key:
            logger.error("NewsAPI key not configured")
            return []

        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("NewsAPI daily rate limit reached")
            return []

        try:
            # Build query from symbols
            query = self._build_query(symbols)

            # Calculate time range
            from_time = datetime.utcnow() - timedelta(hours=min(lookback_hours, 720))  # Max 30 days
            to_time = datetime.utcnow()

            # API parameters
            params = {
                'q': query,
                'from': from_time.isoformat(),
                'to': to_time.isoformat(),
                'language': self.language,
                'sortBy': self.sort_by,
                'pageSize': min(max_articles, 100),  # Max 100 per request
                'apiKey': self.api_key
            }

            logger.debug(f"Fetching NewsAPI: query={query}, lookback={lookback_hours}h")

            # Make request
            response = requests.get(self.everything_endpoint, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data['status'] != 'ok':
                raise Exception(f"NewsAPI error: {data.get('message', 'Unknown error')}")

            articles = data.get('articles', [])
            logger.info(f"NewsAPI returned {len(articles)} articles")

            # Convert to Article objects
            parsed_articles = []
            for article_data in articles:
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
            logger.error(f"NewsAPI request failed: {e}")
            self._update_stats(0, error=True)
            return []

        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}", exc_info=True)
            self._update_stats(0, error=True)
            return []

    def _build_query(self, symbols: List[str]) -> str:
        """
        Build search query from symbols.

        Args:
            symbols: List of symbols

        Returns:
            Query string for NewsAPI
        """
        # Get symbol variations
        query_terms = []
        for symbol in symbols:
            variations = self._get_symbol_variations(symbol)
            query_terms.extend(variations)

        # Join with OR operator
        query = ' OR '.join(query_terms)

        # Add crypto-related terms to improve relevance
        query = f"({query}) AND (cryptocurrency OR crypto OR trading)"

        return query

    def _parse_article(self, article_data: dict, symbols: List[str]) -> Optional[Article]:
        """
        Parse NewsAPI article data into Article object.

        Args:
            article_data: Raw article data from NewsAPI
            symbols: List of symbols to extract

        Returns:
            Article object or None if parsing fails
        """
        try:
            # Parse published date
            published_at = article_data.get('publishedAt')
            if published_at:
                dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                timestamp = int(dt.timestamp())
                datetime_str = dt.isoformat()
            else:
                timestamp = int(datetime.utcnow().timestamp())
                datetime_str = datetime.utcnow().isoformat()

            # Extract text for symbol matching
            title = article_data.get('title', '')
            description = article_data.get('description', '')
            content = article_data.get('content', '')
            full_text = f"{title} {description} {content}"

            # Find mentioned symbols
            mentioned_symbols = self._extract_symbols(full_text, symbols)

            # Skip if no relevant symbols
            if not mentioned_symbols:
                return None

            # Determine primary symbol (first mentioned or most relevant)
            primary_symbol = mentioned_symbols[0] if mentioned_symbols else None

            # Create Article
            article = Article(
                timestamp=timestamp,
                datetime=datetime_str,
                source='newsapi',
                title=title,
                description=description,
                content=content,
                url=article_data.get('url'),
                symbols=mentioned_symbols,
                primary_symbol=primary_symbol,
                processed=False
            )

            # Calculate relevance score
            if primary_symbol:
                article.relevance_score = self._calculate_relevance_score(article, primary_symbol)

            # Create content hash for deduplication
            article.content_hash = self._create_content_hash(article_data)

            return article

        except Exception as e:
            logger.warning(f"Failed to parse NewsAPI article: {e}")
            return None

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
