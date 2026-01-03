"""
News & Sentiment Analysis
=========================

Components for news collection and sentiment analysis.

- NewsCollector: Collect news from multiple sources
- SentimentAnalyzer: Analyze sentiment using VADER
- SentimentAggregator: Aggregate sentiment features for trading

Fetchers:
- NewsAPI: News from NewsAPI.org
- Alpha Vantage: News & Sentiment API
"""

from .types import Article, SentimentResult, SentimentFeatures
from .collector import NewsCollector
from .sentiment import SentimentAnalyzer
from .aggregator import SentimentAggregator
from .fetchers import NewsAPIFetcher, AlphaVantageFetcher, BaseFetcher

__all__ = [
    # Data types
    'Article',
    'SentimentResult',
    'SentimentFeatures',

    # Main services
    'NewsCollector',
    'SentimentAnalyzer',
    'SentimentAggregator',

    # Fetchers
    'NewsAPIFetcher',
    'AlphaVantageFetcher',
    'BaseFetcher'
]
