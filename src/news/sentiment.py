"""
Sentiment Analyzer
==================

Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

VADER is specifically designed for:
- Social media text
- Short texts (headlines, tweets)
- Mixed sentiment
- Emojis and slang

Perfect for crypto news which often includes:
- "🚀 Bitcoin to the moon!"
- "Bearish market dump"
- "HODL strong!"

Paper: Hutto & Gilbert (2014) - VADER: A Parsimonious Rule-based Model for
       Sentiment Analysis of Social Media Text
"""

import logging
from typing import Optional, Dict
from datetime import datetime

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning(
        "vaderSentiment not installed. "
        "Install with: pip install vaderSentiment"
    )

from .types import Article, SentimentResult

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyze sentiment of news articles using VADER.

    Features:
    - Crypto-specific lexicon (bullish, bearish, moon, dump, etc.)
    - Handles emojis and slang
    - Fast and efficient
    - No training required

    Thread-safe: Can be called from multiple threads
    """

    def __init__(self, config: dict = None):
        """
        Initialize sentiment analyzer.

        Args:
            config: Configuration dict from config.yaml['news']['sentiment']
        """
        self.config = config or {}

        if not VADER_AVAILABLE:
            raise ImportError(
                "vaderSentiment is required for sentiment analysis. "
                "Install with: pip install vaderSentiment"
            )

        # Initialize VADER
        self.analyzer = SentimentIntensityAnalyzer()

        # Add custom crypto lexicon
        self._add_crypto_lexicon()

        # Statistics
        self._stats = {
            'articles_analyzed': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'avg_confidence': 0.0
        }

        logger.info("SentimentAnalyzer initialized with VADER + crypto lexicon")

    def _add_crypto_lexicon(self):
        """
        Add crypto-specific terms to VADER lexicon.

        Extends VADER with cryptocurrency-specific sentiment words.
        """
        crypto_terms = self.config.get('custom_lexicon', {})

        # Default crypto lexicon if not in config
        if not crypto_terms:
            crypto_terms = {
                # Bullish terms
                'bullish': 0.8,
                'moon': 0.6,
                'mooning': 0.7,
                'hodl': 0.4,
                'buy': 0.5,
                'pump': 0.6,
                'rally': 0.7,
                'surge': 0.6,
                'breakout': 0.7,
                'uptrend': 0.6,
                'gainz': 0.7,
                'lambo': 0.5,
                'diamond hands': 0.6,

                # Bearish terms
                'bearish': -0.8,
                'dump': -0.7,
                'dumping': -0.7,
                'crash': -0.9,
                'sell': -0.5,
                'panic': -0.7,
                'fud': -0.6,
                'rekt': -0.8,
                'bag holder': -0.6,
                'downtrend': -0.6,
                'correction': -0.4,
                'drop': -0.5,
                'plunge': -0.7,
                'collapse': -0.9,
                'paper hands': -0.6,

                # Market terms (context-dependent)
                'volatile': -0.3,
                'volatility': -0.2,
                'uncertainty': -0.4,
                'regulation': -0.3,
                'ban': -0.7,
                'adoption': 0.6,
                'institutional': 0.5,
                'etf': 0.4,
                'halving': 0.5,
                'upgrade': 0.6
            }

        # Update VADER lexicon
        for term, score in crypto_terms.items():
            self.analyzer.lexicon[term.lower()] = score

        logger.debug(f"Added {len(crypto_terms)} crypto terms to VADER lexicon")

    def analyze(self, text: str) -> Optional[SentimentResult]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze (article title, description, or content)

        Returns:
            SentimentResult or None if analysis fails
        """
        if not text or not isinstance(text, str):
            return None

        try:
            # Get VADER scores
            scores = self.analyzer.polarity_scores(text)

            # Convert to SentimentResult
            result = SentimentResult.from_vader(scores)

            # Update statistics
            self._update_stats(result)

            return result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return None

    def analyze_article(self, article: Article) -> Article:
        """
        Analyze sentiment of an article and update its sentiment fields.

        Args:
            article: Article object

        Returns:
            Updated Article object with sentiment scores
        """
        # Combine title and description for sentiment analysis
        # Title is weighted more heavily in VADER
        text = ""
        if article.title:
            text += article.title
        if article.description:
            text += " " + article.description

        if not text.strip():
            logger.warning("Article has no text for sentiment analysis")
            return article

        # Analyze sentiment
        result = self.analyze(text)

        if result:
            # Update article with sentiment
            article.sentiment_score = result.score
            article.sentiment_label = result.label
            article.sentiment_compound = result.compound
            article.processed = True

            logger.debug(
                f"Sentiment: {result.label} ({result.score:.2f}) - "
                f"{article.title[:50]}..."
            )
        else:
            logger.warning(f"Failed to analyze sentiment for: {article.title[:50]}...")

        return article

    def batch_analyze(self, articles: list) -> list:
        """
        Analyze sentiment for multiple articles.

        Args:
            articles: List of Article objects

        Returns:
            List of updated Article objects
        """
        analyzed = []

        for article in articles:
            try:
                analyzed_article = self.analyze_article(article)
                analyzed.append(analyzed_article)
            except Exception as e:
                logger.error(f"Failed to analyze article: {e}")
                analyzed.append(article)  # Add unmodified

        logger.info(f"Analyzed sentiment for {len(analyzed)} articles")

        return analyzed

    def _update_stats(self, result: SentimentResult):
        """Update statistics."""
        self._stats['articles_analyzed'] += 1

        if result.label == 'positive':
            self._stats['positive_count'] += 1
        elif result.label == 'negative':
            self._stats['negative_count'] += 1
        else:
            self._stats['neutral_count'] += 1

        # Update average confidence (running average)
        n = self._stats['articles_analyzed']
        old_avg = self._stats['avg_confidence']
        self._stats['avg_confidence'] = (old_avg * (n - 1) + result.confidence) / n

    def get_stats(self) -> dict:
        """Get sentiment analyzer statistics."""
        total = self._stats['articles_analyzed']

        if total == 0:
            return {**self._stats, 'sentiment_distribution': {}}

        return {
            **self._stats,
            'sentiment_distribution': {
                'positive': self._stats['positive_count'] / total,
                'negative': self._stats['negative_count'] / total,
                'neutral': self._stats['neutral_count'] / total
            }
        }

    def get_market_sentiment(
        self,
        articles: list,
        time_weighted: bool = True,
        decay_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate overall market sentiment from articles.

        Args:
            articles: List of Article objects (with sentiment analyzed)
            time_weighted: Apply time decay (recent articles weighted more)
            decay_rate: Decay rate for time weighting (0.1 = 10% per hour)

        Returns:
            Dict with:
            - avg_sentiment: Average sentiment score
            - positive_ratio: Ratio of positive articles
            - confidence: Confidence in sentiment assessment
            - article_count: Number of articles analyzed
        """
        if not articles:
            return {
                'avg_sentiment': 0.0,
                'positive_ratio': 0.5,
                'confidence': 0.0,
                'article_count': 0
            }

        sentiments = []
        weights = []
        now = datetime.utcnow()

        for article in articles:
            if article.sentiment_score is None:
                continue

            sentiments.append(article.sentiment_score)

            # Calculate time weight if enabled
            if time_weighted:
                # Calculate hours since article
                article_time = datetime.fromisoformat(article.datetime)
                hours_old = (now - article_time).total_seconds() / 3600

                # Exponential decay: weight = e^(-decay_rate * hours)
                import math
                weight = math.exp(-decay_rate * hours_old)
                weights.append(weight)
            else:
                weights.append(1.0)

        if not sentiments:
            return {
                'avg_sentiment': 0.0,
                'positive_ratio': 0.5,
                'confidence': 0.0,
                'article_count': 0
            }

        # Calculate weighted average
        import numpy as np
        weights_array = np.array(weights)
        sentiments_array = np.array(sentiments)

        weighted_avg = np.average(sentiments_array, weights=weights_array)

        # Calculate positive ratio
        positive_count = sum(1 for s in sentiments if s > 0.05)
        positive_ratio = positive_count / len(sentiments)

        # Confidence is based on agreement (low std = high confidence)
        std_dev = np.std(sentiments_array)
        confidence = max(0.0, 1.0 - std_dev)  # Lower variance = higher confidence

        return {
            'avg_sentiment': float(weighted_avg),
            'positive_ratio': float(positive_ratio),
            'confidence': float(confidence),
            'article_count': len(sentiments)
        }
