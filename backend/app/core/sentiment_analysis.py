import asyncio
import aiohttp
import json
import json
from groq import Groq
import numpy as np
from typing import List, Dict, Tuple
from transformers import pipeline
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import pandas as pd

from app.core.config import settings


import hashlib
import logging

from app.utils.redis import get_redis

logger = logging.getLogger(__name__)


def _json_serializable(obj):
    """Convert object to JSON-serializable dict"""
    import datetime as dt
    if hasattr(obj, '__dict__'):
        result = {}
        for k, v in obj.__dict__.items():
            if isinstance(v, pd.Timestamp):
                result[k] = v.isoformat()
            elif pd.isna(v):
                result[k] = None
            elif isinstance(v, (dt.datetime, dt.date)):
                result[k] = v.isoformat() if hasattr(v, 'isoformat') else str(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [_json_serializable(i) if hasattr(i, '__dict__') else (
                    i.isoformat() if isinstance(i, (dt.datetime, dt.date)) else i
                ) for i in v]
            elif hasattr(v, '__dict__'):
                result[k] = _json_serializable(v)
            elif v is None or isinstance(v, (str, int, float, bool)):
                result[k] = v
            else:
                result[k] = str(v)
        return result
    return obj

@dataclass
class SentimentResult:
    """Structured sentiment result"""
    score: float  # -1 to 1
    confidence: float
    sentiment: str  # positive, negative, neutral
    source: str
    impact: float  # 0-1 impact score
    key_points: List[str]

class EnsembleSentimentModel:
    """Ensemble of multiple sentiment models"""
    
    def __init__(self, hf_token: str = settings.HF_TOKEN):
        # Model khusus Indonesia
        self.indonesian_model = pipeline(
            "sentiment-analysis",
            model="w11wo/indonesian-roberta-base-sentiment-classifier",
            tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier",
            use_auth_token=hf_token
        )
        
        # Multilingual model
        self.multilingual_model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
            use_auth_token=hf_token
        )
        
        # FinBERT untuk financial sentiment
        self.financial_model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            use_auth_token=hf_token
        )
        
        self.weights = {
            'indonesian': 0.5,
            'multilingual': 0.2,
            'financial': 0.3
        }
    
    def analyze(self, text: str) -> SentimentResult:
        """Ensemble analysis dari semua model"""
        results = {}
        
        # Indonesian model
        try:
            indo_result = self.indonesian_model(text)[0]
            label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            results['indonesian'] = {
                'score': label_map.get(indo_result['label'], 0),
                'confidence': indo_result['score']
            }
        except:
            results['indonesian'] = {'score': 0, 'confidence': 0.5}
        
        # Multilingual model
        try:
            multi_result = self.multilingual_model(text)[0]
            score = (int(multi_result['label'].split()[0]) - 3) / 2
            results['multilingual'] = {
                'score': score,
                'confidence': multi_result['score']
            }
        except:
            results['multilingual'] = {'score': 0, 'confidence': 0.5}
        
        # Financial model
        try:
            fin_result = self.financial_model(text)[0]
            label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            results['financial'] = {
                'score': label_map.get(fin_result['label'], 0),
                'confidence': fin_result['score']
            }
        except:
            results['financial'] = {'score': 0, 'confidence': 0.5}
        
        # Weighted ensemble
        weighted_score = sum(
            results[model]['score'] * results[model]['confidence'] * self.weights[model]
            for model in self.weights.keys()
        )
        total_weight = sum(
            results[model]['confidence'] * self.weights[model]
            for model in self.weights.keys()
        )
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        
        return SentimentResult(
            score=final_score,
            confidence=total_weight,
            sentiment='positive' if final_score > 0.2 else 'negative' if final_score < -0.2 else 'neutral',
            source='ensemble',
            impact=abs(final_score),
            key_points=[]
        )

class GroqLLMAnalyzer:
    """Groq LLM untuk analisis sentimen mendalam"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    async def analyze(self, text: str) -> Dict:
        """Analyze text with Groq LLM"""
        import re
        
        # Initialize client in __init__ instead of per-call
        if not hasattr(self, '_client'):
            self._client = Groq(api_key=self.api_key)

        system_prompt = """Analisis sentimen berita berikut untuk pasar saham Indonesia.
Berikan analisis mendalam dengan format JSON:
{
    "sentiment": "positive/negative/neutral",
    "confidence": 0-1,
    "impact_score": 0-1,
    "key_points": [],
    "market_impact": "high/medium/low",
    "suggested_action": "buy/hold/sell"
}

Berita: """

        try:
            # Use asyncio.to_thread to avoid blocking the event loop
            completion = await asyncio.to_thread(
                self._client.chat.completions.create,
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=500
            )

            groq_jawab = completion.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', groq_jawab, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {'sentiment': 'neutral', 'confidence': 0.5, 'impact_score': 0.5}
        except Exception as e:
            logger.error(f"Groq analysis error: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'impact_score': 0.5}

class SentimentAnalyzerV5:
    """Level 5 Sentiment Analysis dengan multi-source dan ensemble"""
    
    def __init__(self, groq_api_key: str = None, news_scraper=None, hf_token: str = None):
        self.ensemble = EnsembleSentimentModel(hf_token)
        self.groq = GroqLLMAnalyzer(groq_api_key) if groq_api_key else None
        # Handle Redis connection failure gracefully
        try:
            self.cache = get_redis()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.cache = None
        self.news_scraper = news_scraper
        
    def _get_cache_key(self, symbol: str) -> str:
        return f"sentiment:{symbol}"
    
    async def analyze_news_batch(self, news_items: List[Dict]) -> List[SentimentResult]:
        """Batch analysis of multiple news items"""
        tasks = []
        for news in news_items:
            tasks.append(self.analyze_single_news(news['title']))
        results = await asyncio.gather(*tasks)
        return results
    
    async def analyze_single_news(self, text: str) -> SentimentResult:
        """Single news analysis with caching"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        try:
            if self.cache:
                cached = self.cache.get(cache_key)
            else:
                cached = None
        except Exception as e:
            logger.warning(f"Sentiment cache read failed: {e}")
            cached = None
        if cached:
            return SentimentResult(**json.loads(cached))
        
        # Ensemble model analysis
        ensemble_result = self.ensemble.analyze(text)
        
        # Groq analysis for deeper insight (if available)
        groq_result = None
        if self.groq:
            groq_result = await self.groq.analyze(text)
        
        # Combine results
        if groq_result:
            final_score = (ensemble_result.score + 
                          (1 if groq_result.get('sentiment') == 'positive' else 
                           -1 if groq_result.get('sentiment') == 'negative' else 0)) / 2
            confidence = (ensemble_result.confidence + groq_result.get('confidence', 0.5)) / 2
            impact = groq_result.get('impact_score', ensemble_result.impact)
        else:
            final_score = ensemble_result.score
            confidence = ensemble_result.confidence
            impact = ensemble_result.impact
        
        result = SentimentResult(
            score=final_score,
            confidence=confidence,
            sentiment='positive' if final_score > 0.2 else 'negative' if final_score < -0.2 else 'neutral',
            source='ensemble+groq' if groq_result else 'ensemble',
            impact=impact,
            key_points=groq_result.get('key_points', []) if groq_result else []
        )
        
        # Cache result (TTL 1 hour)
        try:
            if self.cache:
                self.cache.setex(cache_key, 3600, json.dumps(_json_serializable(result)))
        except Exception as e:
            logger.warning(f"Sentiment cache write failed: {e}")
        
        return result
    
    async def analyze_symbol(self, symbol: str) -> Dict:
        """Complete sentiment analysis for a symbol"""
        # Check cache
        cache_key = self._get_cache_key(symbol)
        try:
            if self.cache:
                cached = self.cache.get(cache_key)
            else:
                cached = None
        except Exception as e:
            logger.warning(f"Sentiment cache read failed: {e}")
            cached = None
        if cached:
            return json.loads(cached)
        
        # Get news from scraper
        if self.news_scraper:
            news_items = await self.news_scraper.get_news(symbol, max_articles=15)
        else:
            news_items = []
        
        if not news_items:
            return {
                'symbol': symbol,
                'avg_score': 0,
                'confidence': 0,
                'sentiment': 'neutral',
                'news_count': 0,
                'details': []
            }
        
        # Analyze all news
        try:
            results = await self.analyze_news_batch(news_items)
        except Exception as e:
            logger.warning(f"News batch analysis failed: {e}")
            results = []
        
        # Weighted average based on impact and recency
        total_weight = 0
        weighted_score = 0
        for i, (news, result) in enumerate(zip(news_items, results)):
            # Recent news have higher weight
            time_weight = 1.0  # Could be based on timestamp
            weight = result.impact * time_weight
            weighted_score += result.score * weight
            total_weight += weight
        
        avg_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Handle empty results
        if not results:
            return {
                'symbol': symbol,
                'avg_score': 0,
                'confidence': 0,
                'sentiment': 'neutral',
                'news_count': 0,
                'details': []
            }
        
        final_result = {
            'symbol': symbol,
            'avg_score': avg_score,
            'confidence': sum(r.confidence for r in results) / len(results) if results else 0,
            'sentiment': 'positive' if avg_score > 0.2 else 'negative' if avg_score < -0.2 else 'neutral',
            'news_count': len(news_items),
            'details': [
                {
                    'title': news['title'],
                    'sentiment': result.sentiment,
                    'score': result.score,
                    'confidence': result.confidence,
                    'key_points': result.key_points
                }
                for news, result in zip(news_items, results)
            ]
        }
        
        # Cache results (TTL 5 minutes)
        try:
            if self.cache:
                self.cache.setex(cache_key, 300, json.dumps(_json_serializable(final_result)))
        except Exception as e:
            logger.warning(f"Sentiment cache write failed: {e}")
        
        return final_result


class SentimentAnalyzer:
    """Lightweight sentiment analyzer for training features."""
    def analyze_news(self, symbol: str) -> float:
        # Placeholder: return neutral sentiment to avoid blocking training.
        return 0.0
