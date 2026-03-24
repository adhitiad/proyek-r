import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import asyncio
import aiohttp
import backoff
import redis
import json
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis cache untuk data collection dengan TTL"""
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    
    def get(self, key):
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    def set(self, key, value, ttl=300):
        self.redis_client.setex(key, ttl, json.dumps(value))
    
    def generate_key(self, *args, **kwargs):
        key_str = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

class DataSourceManager:
    """Multi-source data dengan auto-fallback"""
    SOURCES = {
        'yfinance': {'priority': 1, 'timeout': 10},
        'alpha_vantage': {'priority': 2, 'timeout': 15, 'api_key': None},
        'polygon': {'priority': 3, 'timeout': 10, 'api_key': None},
        'yahoo_finance': {'priority': 4, 'timeout': 8}
    }
    
    def __init__(self):
        self.cache = RedisCache()
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def fetch_from_source(self, source: str, symbol: str, **kwargs):
        """Fetch data dari single source dengan retry"""
        if source == 'yfinance':
            return await self._fetch_yfinance(symbol, **kwargs)
        elif source == 'alpha_vantage':
            return await self._fetch_alpha_vantage(symbol, **kwargs)
        return None
    
    async def _fetch_yfinance(self, symbol: str, period: str = "3mo", interval: str = "1d"):
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            self.executor,
            lambda: yf.Ticker(symbol).history(period=period, interval=interval)
        )
        return df
    
    @staticmethod
    async def get_price_data(self, symbol: str, period: str = "3mo", interval: str = "1d", use_cache=True):
        """Get price data dengan multi-source dan caching"""
        try:
            cache_key = self.cache.generate_key('price', symbol, period, interval)
            
            if use_cache:
                cached = self.cache.get(cache_key)
                if cached:
                    return pd.DataFrame(cached)
            
            for source in sorted(self.SOURCES.keys(), key=lambda x: self.SOURCES[x]['priority']):
                try:
                    df = await self.fetch_from_source(source, symbol, period=period, interval=interval)
                    if df is not None and not df.empty:
                        # Cache hasil
                        self.cache.set(cache_key, df.to_dict('records'), ttl=300)
                        return df
                except Exception as e:
                    logger.warning(f"Source {source} failed for {symbol}: {e}")
                    continue
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return pd.DataFrame()


class NewsScraperV5:
    """Real-time news scraper dengan multiple sources dan rate limiting"""
    
    SOURCES = {
        'detik': {
            'url': 'https://www.detik.com/search/searchall?query={query}',
            'selector': '.media',
            'title_selector': '.media__title a',
            'date_selector': '.media__date'
        },
        'tempo': {
            'url': 'https://www.tempo.co/search?q={query}',
            'selector': '.article-list-item',
            'title_selector': 'h2 a',
            'date_selector': '.date'
        },
        'cnbc': {
            'url': 'https://www.cnbcindonesia.com/search?q={query}',
            'selector': '.search-list-item',
            'title_selector': 'h3 a',
            'date_selector': '.date'
        },
        'kontan': {
            'url': 'https://investasi.kontan.co.id/search?q={query}',
            'selector': '.article-item',
            'title_selector': 'h3 a',
            'date_selector': '.date'
        },
        'bisnis': {
            'url': 'https://market.bisnis.com/search?q={query}',
            'selector': '.article-item',
            'title_selector': 'h3 a',
            'date_selector': '.date'
        }
    }
    
    def __init__(self):
        self.cache = RedisCache()
        self.session = None
    
    async def _scrape_source(self, source: str, query: str, limit: int = 5):
        """Scrape single source dengan async"""
        config = self.SOURCES[source]
        url = config['url'].format(query=query.replace(' ', '%20'))
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status != 200:
                        return []
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    articles = []
                    for item in soup.select(config['selector'])[:limit]:
                        title_elem = item.select_one(config['title_selector'])
                        date_elem = item.select_one(config['date_selector'])
                        if title_elem:
                            articles.append({
                                'title': title_elem.text.strip(),
                                'url': title_elem.get('href', ''),
                                'date': date_elem.text.strip() if date_elem else datetime.now().isoformat(),
                                'source': source,
                                'scraped_at': datetime.now().isoformat()
                            })
                    return articles
        except Exception as e:
            logger.error(f"Error scraping {source}: {e}")
            return []
    
    async def get_news(self, symbol: str, max_articles: int = 20) -> List[Dict]:
        """Get news dari semua sumber secara paralel"""
        query = symbol.replace('.JK', '').lower()
        cache_key = self.cache.generate_key('news', symbol)
        
        # Check cache (TTL 5 menit untuk news)
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        tasks = [self._scrape_source(source, query, limit=5) for source in self.SOURCES.keys()]
        results = await asyncio.gather(*tasks)
        
        all_articles = []
        for articles in results:
            all_articles.extend(articles)
        
        # Sort by date
        all_articles.sort(key=lambda x: x['date'], reverse=True)
        all_articles = all_articles[:max_articles]
        
        # Cache results
        self.cache.set(cache_key, all_articles, ttl=300)
        
        return all_articles