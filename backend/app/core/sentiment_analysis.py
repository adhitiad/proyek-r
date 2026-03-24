import asyncio
import aiohttp
import json
import numpy as np
from typing import List, Dict, Tuple
from transformers import pipeline
from dataclasses import dataclass
import torch
import torch.nn as nn
import redis
import hashlib
import logging

logger = logging.getLogger(__name__)

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
    
    def __init__(self):
        # Model khusus Indonesia
        self.indonesian_model = pipeline(
            "sentiment-analysis",
            model="w11wo/indonesian-roberta-base-sentiment-classifier",
            tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier"
        )
        
        # Multilingual model
        self.multilingual_model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
        # FinBERT untuk financial sentiment
        self.financial_model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
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
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
    async def analyze(self, text: str) -> Dict:
        """Analyze text with Groq LLM"""
        prompt = f"""
        Analisis sentimen berita berikut untuk pasar saham Indonesia.
        Berikan analisis mendalam dengan format JSON:
        {{
            "sentiment": "positive/negative/neutral",
            "confidence": 0-1,
            "impact_score": 0-1,
            "key_points": [],
            "market_impact": "high/medium/low",
            "suggested_action": "buy/hold/sell"
        }}
        
        Berita: "{text}"
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        # Extract JSON
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group())
            return {'sentiment': 'neutral', 'confidence': 0.5, 'impact_score': 0.5}
        except Exception as e:
            logger.error(f"Groq analysis error: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'impact_score': 0.5}

class SentimentAnalyzerV5:
    """Level 5 Sentiment Analysis dengan multi-source dan ensemble"""
    
    def __init__(self, groq_api_key: str = None):
        self.ensemble = EnsembleSentimentModel()
        self.groq = GroqLLMAnalyzer(groq_api_key) if groq_api_key else None
        self.cache = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
        self.news_scraper = None  # Will be injected
        
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
        cached = self.cache.get(cache_key)
        if cached:
            import json
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
        import json
        self.cache.setex(cache_key, 3600, json.dumps(result.__dict__))
        
        return result
    
    async def analyze_symbol(self, symbol: str) -> Dict:
        """Complete sentiment analysis for a symbol"""
        # Check cache
        cache_key = self._get_cache_key(symbol)
        cached = self.cache.get(cache_key)
        if cached:
            import json
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
        results = await self.analyze_news_batch(news_items)
        
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
        
        final_result = {
            'symbol': symbol,
            'avg_score': avg_score,
            'confidence': sum(r.confidence for r in results) / len(results),
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
        import json
        self.cache.setex(cache_key, 300, json.dumps(final_result))
        
        return final_result