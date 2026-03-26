import os
import pandas as pd
import numpy as np
import aiohttp
import asyncio
from app.utils.redis import get_redis
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from app.core.config import settings

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
class InstitutionalFlow:
    """Institutional order flow data"""
    net_flow: float
    buying_pressure: float
    selling_pressure: float
    large_orders: int
    source: str
    timestamp: str

class BandarDetectorV5:
    """Level 5 Bandar Detector dengan multiple data sources dan institutional flow"""
    
    def __init__(self):
        try:
            self.cache = get_redis()
        except Exception:
            self.cache = None
        self.sources = {
            'rti': {'url': 'https://rti.business/api/foreign/{symbol}', 'priority': 1},
            'idx': {'url': 'https://api.idx.co.id/v1/foreign/{symbol}', 'priority': 2},
            'bloomberg': {'url': None, 'priority': 3},  # Requires subscription
            'reuters': {'url': None, 'priority': 4}
        }
        
    def _get_cache_key(self, symbol: str) -> str:
        return f"bandar:{symbol}"
    
    async def fetch_rti_flow(self, symbol: str) -> Optional[Dict]:
        """Fetch foreign flow from RTI Business"""
        if settings.BANDAR_DISABLE_RTI:
            return None
        try:
            async with aiohttp.ClientSession() as session:
                url = self.sources['rti']['url'].format(symbol=symbol.replace('.JK', ''))
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'foreign_buy': data.get('foreign_buy', 0),
                            'foreign_sell': data.get('foreign_sell', 0),
                            'net_flow': data.get('foreign_buy', 0) - data.get('foreign_sell', 0),
                            'source': 'rti'
                        }
        except Exception as e:
            logger.warning(f"RTI fetch failed for {symbol}: {e}")
        return None
    
    async def fetch_idx_flow(self, symbol: str) -> Optional[Dict]:
        """Fetch foreign flow from IDX API"""
        try:
            async with aiohttp.ClientSession() as session:
                # IDX API endpoint (example)
                url = f"https://api.idx.co.id/v1/foreign/{symbol}"
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'foreign_buy': data.get('buy', 0),
                            'foreign_sell': data.get('sell', 0),
                            'net_flow': data.get('buy', 0) - data.get('sell', 0),
                            'source': 'idx'
                        }
        except Exception as e:
            logger.warning(f"IDX fetch failed for {symbol}: {e}")
        return None
    
    async def detect_institutional_flow(self, symbol: str, df: pd.DataFrame) -> InstitutionalFlow:
        """Detect institutional flow from multiple sources"""
        # Try real data sources first
        real_data = None
        
        # RTI
        rti_data = await self.fetch_rti_flow(symbol)
        if rti_data:
            real_data = rti_data
        
        # IDX if RTI fails
        if not real_data:
            idx_data = await self.fetch_idx_flow(symbol)
            if idx_data:
                real_data = idx_data
        
        if real_data:
            return InstitutionalFlow(
                net_flow=real_data['net_flow'],
                buying_pressure=real_data['foreign_buy'] / (real_data['foreign_buy'] + real_data['foreign_sell']) if real_data['foreign_buy'] + real_data['foreign_sell'] > 0 else 0,
                selling_pressure=real_data['foreign_sell'] / (real_data['foreign_buy'] + real_data['foreign_sell']) if real_data['foreign_buy'] + real_data['foreign_sell'] > 0 else 0,
                large_orders=0,
                source=real_data['source'],
                timestamp=pd.Timestamp.now().isoformat()
            )
        
        # Fallback: Proxy detection based on volume and price action
        return self._proxy_detection(df)
    
    def _proxy_detection(self, df: pd.DataFrame) -> InstitutionalFlow:
        """Proxy detection using volume and price action"""
        volume_avg = df['Volume'].rolling(20).mean()
        price_change = df['Close'].pct_change()
        
        # Detect accumulation: volume surge + price increase
        volume_surge = df['Volume'] > volume_avg * 1.5
        price_up = price_change > 0
        
        accumulation_days = (volume_surge & price_up).sum()
        total_days = len(df)
        
        net_flow = df['Volume'].diff().sum() if accumulation_days > total_days * 0.3 else 0
        buying_pressure = accumulation_days / total_days if total_days > 0 else 0
        
        return InstitutionalFlow(
            net_flow=net_flow,
            buying_pressure=buying_pressure,
            selling_pressure=1 - buying_pressure,
            large_orders=int(df['Volume'].max() / volume_avg.mean()) if volume_avg.mean() > 0 else 0,
            source='proxy',
            timestamp=pd.Timestamp.now().isoformat()
        )
    
    def detect_accumulation(self, flow: InstitutionalFlow, df: pd.DataFrame) -> Dict:
        """Detect accumulation pattern"""
        accumulation_score = 0
        
        # Net flow positive
        if flow.net_flow > 0:
            accumulation_score += 2
        
        # Buying pressure > 60%
        if flow.buying_pressure > 0.6:
            accumulation_score += 2
        elif flow.buying_pressure > 0.55:
            accumulation_score += 1
        
        # Large orders detected (proxy)
        if flow.large_orders > 10:
            accumulation_score += 1
        
        # Price action confirmation
        price_up_5d = (df['Close'].iloc[-1] > df['Close'].iloc[-5]) if len(df) >= 5 else False
        if price_up_5d:
            accumulation_score += 1
        
        # Volume trend
        volume_trend = df['Volume'].rolling(5).mean().iloc[-1] > df['Volume'].rolling(20).mean().iloc[-1]
        if volume_trend:
            accumulation_score += 1
        
        return {
            'is_accumulating': accumulation_score >= 3,
            'accumulation_score': accumulation_score,
            'strength': accumulation_score / 7,
            'net_flow': flow.net_flow,
            'buying_pressure': flow.buying_pressure,
            'source': flow.source,
            'confidence': 0.9 if flow.source != 'proxy' else 0.5
        }
    
    def detect_distribution(self, flow: InstitutionalFlow, df: pd.DataFrame) -> Dict:
        """Detect distribution pattern"""
        distribution_score = 0
        
        # Net flow negative
        if flow.net_flow < 0:
            distribution_score += 2
        
        # Selling pressure > 60%
        if flow.selling_pressure > 0.6:
            distribution_score += 2
        elif flow.selling_pressure > 0.55:
            distribution_score += 1
        
        # Price action confirmation
        price_down_5d = (df['Close'].iloc[-1] < df['Close'].iloc[-5]) if len(df) >= 5 else False
        if price_down_5d:
            distribution_score += 1
        
        return {
            'is_distributing': distribution_score >= 3,
            'distribution_score': distribution_score,
            'strength': distribution_score / 7,
            'net_flow': flow.net_flow,
            'selling_pressure': flow.selling_pressure,
            'source': flow.source
        }
    
    async def analyze(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Complete bandar analysis"""
        cache_key = self._get_cache_key(symbol)
        try:
            cached = self.cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Bandar cache read failed: {e}")
            cached = None
        if cached:
            return json.loads(cached)
        
        flow = await self.detect_institutional_flow(symbol, df)
        accumulation = self.detect_accumulation(flow, df)
        distribution = self.detect_distribution(flow, df)
        
        result = {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now().isoformat(),
            'institutional_flow': {
                'net_flow': flow.net_flow,
                'buying_pressure': flow.buying_pressure,
                'selling_pressure': flow.selling_pressure,
                'source': flow.source
            },
            'accumulation': accumulation,
            'distribution': distribution,
            'verdict': 'accumulating' if accumulation['is_accumulating'] else 'distributing' if distribution['is_distributing'] else 'neutral',
            'strength': max(accumulation['strength'], distribution['strength'])
        }
        
        # Cache for 5 minutes
        try:
            self.cache.setex(cache_key, 300, json.dumps(_json_serializable(result)))
        except Exception as e:
            logger.warning(f"Bandar cache write failed: {e}")
        
        return result
