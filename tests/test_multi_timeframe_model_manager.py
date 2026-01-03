"""
Test Multi-Timeframe Model Manager
===================================

Tests model loading, saving, caching, and multi-timeframe operations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import torch
import tempfile
import shutil
from datetime import datetime

from src.multi_timeframe.model_manager import (
    MultiTimeframeModelManager,
    ModelMetadata
)
from src.analysis_engine import LSTMModel, FeatureCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_model(input_size: int = 39) -> LSTMModel:
    """Create a test LSTM model."""
    model = LSTMModel(
        input_size=input_size,
        hidden_size=64,
        num_layers=1,
        dropout=0.1
    )
    return model


def test_model_path_generation():
    """Test model path generation."""
    logger.info("\nTest 1: Model Path Generation")

    manager = MultiTimeframeModelManager(models_dir="test_models")

    # Test various symbols and intervals
    tests = [
        ("BTC/USDT", "1h", "BTC_USDT_1h_model.pt"),
        ("ETH/USDT", "4h", "ETH_USDT_4h_model.pt"),
        ("BTC-USD", "1d", "BTC_USD_1d_model.pt"),
    ]

    for symbol, interval, expected_name in tests:
        path = manager.get_model_path(symbol, interval)
        assert path.name == expected_name, f"Expected {expected_name}, got {path.name}"
        logger.info(f"✓ {symbol} @ {interval} → {path.name}")

    logger.info("✓ All path generation tests passed")


def test_save_and_load_model():
    """Test saving and loading models."""
    logger.info("\nTest 2: Save and Load Model")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        manager = MultiTimeframeModelManager(models_dir=temp_dir)

        # Create test model
        feature_columns = FeatureCalculator.get_feature_columns()
        input_size = len(feature_columns)
        model = create_test_model(input_size)

        # Create metadata with correct config matching the model
        model_config = {
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'bidirectional': False
        }

        metadata = ModelMetadata(
            symbol="BTC/USDT",
            interval="1h",
            trained_at=datetime.utcnow(),
            samples_trained=5000,
            validation_accuracy=0.65,
            validation_confidence=0.82,
            version=1,
            config=model_config
        )

        # Create feature normalization tensors
        feature_means = torch.randn(input_size)
        feature_stds = torch.ones(input_size)

        # Save model
        manager.save_model(
            symbol="BTC/USDT",
            interval="1h",
            model=model,
            metadata=metadata,
            feature_means=feature_means,
            feature_stds=feature_stds
        )

        # Verify file exists
        model_path = manager.get_model_path("BTC/USDT", "1h")
        assert model_path.exists(), "Model file should exist"
        logger.info(f"✓ Model saved to {model_path}")

        # Clear cache
        manager.clear_cache()

        # Load model
        loaded_model = manager.load_model("BTC/USDT", "1h")
        assert loaded_model is not None, "Model should load successfully"
        logger.info("✓ Model loaded successfully")

        # Verify metadata
        loaded_metadata = manager.get_metadata("BTC/USDT", "1h")
        assert loaded_metadata is not None, "Metadata should exist"
        assert loaded_metadata.symbol == "BTC/USDT"
        assert loaded_metadata.interval == "1h"
        assert loaded_metadata.validation_accuracy == 0.65
        assert loaded_metadata.validation_confidence == 0.82
        logger.info(
            f"✓ Metadata verified: acc={loaded_metadata.validation_accuracy:.2%}, "
            f"conf={loaded_metadata.validation_confidence:.2%}"
        )

        # Verify model architecture
        assert isinstance(loaded_model, LSTMModel)
        logger.info("✓ Model architecture verified")

        logger.info("✓ All save/load tests passed")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_multi_interval_management():
    """Test managing multiple intervals for same symbol."""
    logger.info("\nTest 3: Multi-Interval Management")

    temp_dir = tempfile.mkdtemp()

    try:
        manager = MultiTimeframeModelManager(models_dir=temp_dir)

        feature_columns = FeatureCalculator.get_feature_columns()
        input_size = len(feature_columns)

        intervals = ["1h", "4h", "1d"]

        # Save models for each interval
        for interval in intervals:
            model = create_test_model(input_size)

            model_config = {
                'hidden_size': 64,
                'num_layers': 1,
                'dropout': 0.1,
                'bidirectional': False
            }

            metadata = ModelMetadata(
                symbol="BTC/USDT",
                interval=interval,
                trained_at=datetime.utcnow(),
                samples_trained=5000,
                validation_accuracy=0.60 + (0.05 * intervals.index(interval)),
                validation_confidence=0.75,
                version=1,
                config=model_config
            )

            manager.save_model(
                symbol="BTC/USDT",
                interval=interval,
                model=model,
                metadata=metadata
            )

            logger.info(f"✓ Saved model for BTC/USDT @ {interval}")

        # Verify all models exist
        for interval in intervals:
            assert manager.model_exists("BTC/USDT", interval), f"{interval} model should exist"

        logger.info("✓ All models saved successfully")

        # Get all intervals
        all_intervals = manager.get_all_intervals("BTC/USDT")
        assert set(all_intervals) == set(intervals), "All intervals should be listed"
        logger.info(f"✓ All intervals found: {all_intervals}")

        # Get all models for symbol
        all_models = manager.get_all_models("BTC/USDT")
        assert len(all_models) == len(intervals), "Should have all models cached"
        logger.info(f"✓ All models loaded: {list(all_models.keys())}")

        # List all models
        model_list = manager.list_all_models()
        assert len(model_list) == len(intervals), "Should list all models"
        logger.info(f"✓ Model list: {len(model_list)} models")

        for model_info in model_list:
            logger.info(
                f"  - {model_info['symbol']} @ {model_info['interval']}: "
                f"{model_info['size_mb']:.2f} MB"
            )

        logger.info("✓ All multi-interval tests passed")

    finally:
        shutil.rmtree(temp_dir)


def test_cache_functionality():
    """Test model caching."""
    logger.info("\nTest 4: Cache Functionality")

    temp_dir = tempfile.mkdtemp()

    try:
        manager = MultiTimeframeModelManager(models_dir=temp_dir)

        feature_columns = FeatureCalculator.get_feature_columns()
        input_size = len(feature_columns)

        # Save a model
        model = create_test_model(input_size)

        model_config = {
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'bidirectional': False
        }

        metadata = ModelMetadata(
            symbol="BTC/USDT",
            interval="1h",
            trained_at=datetime.utcnow(),
            samples_trained=5000,
            validation_accuracy=0.65,
            validation_confidence=0.82,
            version=1,
            config=model_config
        )

        manager.save_model("BTC/USDT", "1h", model, metadata)

        # Clear cache to force reload from disk
        manager.clear_cache("BTC/USDT", "1h")

        # Load first time (cache miss - needs to load from disk)
        model1 = manager.load_model("BTC/USDT", "1h")
        assert model1 is not None

        stats = manager.get_stats()
        logger.info(f"After first load: {stats}")
        assert stats['cache_misses'] >= 1, "Should have at least one cache miss"
        logger.info("✓ First load caused cache miss")

        # Load second time (cache hit)
        model2 = manager.load_model("BTC/USDT", "1h")
        assert model2 is not None
        assert model2 is model1, "Should return same cached instance"

        stats = manager.get_stats()
        logger.info(f"After second load: {stats}")
        assert stats['cache_hits'] == 1
        logger.info("✓ Second load caused cache hit")

        # Test cache hit rate
        cache_hit_rate = stats['cache_hit_rate']
        assert cache_hit_rate > 0, "Cache hit rate should be greater than 0"
        logger.info(f"✓ Cache hit rate: {cache_hit_rate:.1%}")

        # Test force reload
        model3 = manager.load_model("BTC/USDT", "1h", force_reload=True)
        assert model3 is not None

        stats = manager.get_stats()
        logger.info(f"After force reload: {stats}")
        assert stats['cache_misses'] == 2
        logger.info("✓ Force reload caused cache miss")

        # Clear cache
        manager.clear_cache("BTC/USDT", "1h")
        all_models = manager.get_all_models("BTC/USDT")
        assert len(all_models) == 0, "Cache should be empty"
        logger.info("✓ Cache cleared successfully")

        logger.info("✓ All cache tests passed")

    finally:
        shutil.rmtree(temp_dir)


def test_model_deletion():
    """Test model deletion."""
    logger.info("\nTest 5: Model Deletion")

    temp_dir = tempfile.mkdtemp()

    try:
        manager = MultiTimeframeModelManager(models_dir=temp_dir)

        feature_columns = FeatureCalculator.get_feature_columns()
        input_size = len(feature_columns)

        # Save a model
        model = create_test_model(input_size)

        model_config = {
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'bidirectional': False
        }

        metadata = ModelMetadata(
            symbol="BTC/USDT",
            interval="1h",
            trained_at=datetime.utcnow(),
            samples_trained=5000,
            validation_accuracy=0.65,
            validation_confidence=0.82,
            version=1,
            config=model_config
        )

        manager.save_model("BTC/USDT", "1h", model, metadata)

        # Verify exists
        assert manager.model_exists("BTC/USDT", "1h")
        logger.info("✓ Model exists before deletion")

        # Delete model
        manager.delete_model("BTC/USDT", "1h")

        # Verify deleted
        assert not manager.model_exists("BTC/USDT", "1h")
        logger.info("✓ Model deleted successfully")

        # Verify cache cleared
        all_models = manager.get_all_models("BTC/USDT")
        assert len(all_models) == 0
        logger.info("✓ Cache cleared after deletion")

        logger.info("✓ All deletion tests passed")

    finally:
        shutil.rmtree(temp_dir)


def test_multiple_symbols():
    """Test managing models for multiple symbols."""
    logger.info("\nTest 6: Multiple Symbols")

    temp_dir = tempfile.mkdtemp()

    try:
        manager = MultiTimeframeModelManager(models_dir=temp_dir)

        feature_columns = FeatureCalculator.get_feature_columns()
        input_size = len(feature_columns)

        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        intervals = ["1h", "4h"]

        # Save models for all combinations
        for symbol in symbols:
            for interval in intervals:
                model = create_test_model(input_size)

                model_config = {
                    'hidden_size': 64,
                    'num_layers': 1,
                    'dropout': 0.1,
                    'bidirectional': False
                }

                metadata = ModelMetadata(
                    symbol=symbol,
                    interval=interval,
                    trained_at=datetime.utcnow(),
                    samples_trained=5000,
                    validation_accuracy=0.65,
                    validation_confidence=0.82,
                    version=1,
                    config=model_config
                )

                manager.save_model(symbol, interval, model, metadata)

        logger.info(f"✓ Saved models for {len(symbols)} symbols × {len(intervals)} intervals")

        # List all models
        all_models = manager.list_all_models()
        assert len(all_models) == len(symbols) * len(intervals)
        logger.info(f"✓ Total models: {len(all_models)}")

        # Verify each symbol has correct intervals
        for symbol in symbols:
            intervals_for_symbol = manager.get_all_intervals(symbol)
            assert set(intervals_for_symbol) == set(intervals)
            logger.info(f"✓ {symbol}: {intervals_for_symbol}")

        # Clear cache for one symbol
        manager.clear_cache("BTC/USDT")
        btc_models = manager.get_all_models("BTC/USDT")
        assert len(btc_models) == 0
        logger.info("✓ Cleared cache for BTC/USDT")

        # Verify other symbols still cached
        eth_models = manager.get_all_models("ETH/USDT")
        assert len(eth_models) == len(intervals)
        logger.info("✓ Other symbols still cached")

        logger.info("✓ All multiple symbol tests passed")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    try:
        test_model_path_generation()
        test_save_and_load_model()
        test_multi_interval_management()
        test_cache_functionality()
        test_model_deletion()
        test_multiple_symbols()

        logger.info("\n" + "="*60)
        logger.info("All Multi-Timeframe Model Manager tests passed!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
