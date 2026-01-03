"""
News Fetchers
=============

News fetcher implementations for different sources.
"""

from .base import BaseFetcher
from .newsapi import NewsAPIFetcher
from .alphavantage import AlphaVantageFetcher

__all__ = [
    'BaseFetcher',
    'NewsAPIFetcher',
    'AlphaVantageFetcher'
]
