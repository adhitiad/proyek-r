from __future__ import annotations

from typing import Optional

from app.core.data_collector import DataSourceManager, NewsScraperV5
from app.core.sentiment_analysis import SentimentAnalyzerV5
from app.core.signal_generator import SignalGeneratorV5
from app.core.config import settings

_data_manager: Optional[DataSourceManager] = None
_news_scraper: Optional[NewsScraperV5] = None
_sentiment_analyzer: Optional[SentimentAnalyzerV5] = None
_signal_generator_v5: Optional[SignalGeneratorV5] = None


def get_data_manager() -> DataSourceManager:
    global _data_manager
    if _data_manager is None:
        _data_manager = DataSourceManager()
    return _data_manager


def get_news_scraper() -> NewsScraperV5:
    global _news_scraper
    if _news_scraper is None:
        _news_scraper = NewsScraperV5()
    return _news_scraper


def get_sentiment_analyzer() -> SentimentAnalyzerV5:
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzerV5(
            news_scraper=get_news_scraper(),
            hf_token=settings.HF_TOKEN if settings.HF_TOKEN else None
        )
    return _sentiment_analyzer


def get_signal_generator_v5() -> SignalGeneratorV5:
    global _signal_generator_v5
    if _signal_generator_v5 is None:
        _signal_generator_v5 = SignalGeneratorV5(sentiment=get_sentiment_analyzer())
    return _signal_generator_v5


def shutdown_singletons() -> None:
    global _data_manager
    if _data_manager is not None:
        _data_manager.close()
        _data_manager = None
