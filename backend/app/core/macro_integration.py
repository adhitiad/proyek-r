import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple
import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)

class CrossAssetCorrelation:
    """Cross-asset correlation analysis"""
    
    ASSETS = {
        'stocks': ['^JKSE', '^GSPC', '^IXIC', '^FTSE'],
        'forex': ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X'],
        'commodities': ['GC=F', 'CL=F', 'SI=F', 'HG=F'],
        'bonds': ['^TNX', '^FVX', 'TLT'],
        'crypto': ['BTC-USD', 'ETH-USD']
    }
    
    def __init__(self):
        self.correlation_matrix = None
        self.rolling_correlations = {}
        
    async def fetch_asset_data(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch data for all assets"""
        data = {}
        for symbol in symbols:
            try:
                df = yf.download(symbol, period="1mo", interval="1d", progress=False)
                if not df.empty:
                    data[symbol] = df['Close']
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        return pd.DataFrame(data)
    
    async def calculate_correlations(self) -> Dict:
        """Calculate current cross-asset correlations"""
        all_symbols = []
        for assets in self.ASSETS.values():
            all_symbols.extend(assets)
        
        df = await self.fetch_asset_data(all_symbols)
        
        if df.empty:
            return {}
        
        # Calculate returns
        returns = df.pct_change().dropna()
        
        # Full correlation matrix
        self.correlation_matrix = returns.corr()
        
        # Rolling correlations (20-day window)
        self.rolling_correlations = returns.rolling(20).corr()
        
        # Identify high correlations
        high_correlations = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr = self.correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    high_correlations.append({
                        'asset1': self.correlation_matrix.columns[i],
                        'asset2': self.correlation_matrix.columns[j],
                        'correlation': corr
                    })
        
        return {
            'correlation_matrix': self.correlation_matrix.to_dict(),
            'high_correlations': high_correlations,
            'average_correlation': self.correlation_matrix.values[np.triu_indices_from(self.correlation_matrix.values, k=1)].mean()
        }
    
    def get_hedge_suggestion(self, symbol: str) -> List[Dict]:
        """Suggest hedges based on negative correlations"""
        if self.correlation_matrix is None or symbol not in self.correlation_matrix.columns:
            return []
        
        correlations = self.correlation_matrix[symbol].sort_values()
        # Find negatively correlated assets
        negative_corr = correlations[correlations < -0.5]
        
        return [
            {'hedge_asset': asset, 'correlation': corr}
            for asset, corr in negative_corr.items() if asset != symbol
        ]

class GlobalMacroIntegration:
    """Integrate global macroeconomic data"""
    
    MACRO_INDICATORS = {
        'US': ['GDP', 'CPI', 'UNEMPLOYMENT', 'FED_RATE'],
        'INDONESIA': ['GDP_ID', 'CPI_ID', 'BI_RATE'],
        'CHINA': ['GDP_CN', 'PMI_CN'],
        'GLOBAL': ['OIL_PRICE', 'GOLD_PRICE', 'VIX']
    }
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.macro_cache = {}
        
    async def fetch_macro_data(self) -> Dict:
        """Fetch macroeconomic data from various sources"""
        macro_data = {}
        
        # Fetch from FRED API (example)
        macro_data['US_FED_RATE'] = await self._fetch_fred_data('FEDFUNDS')
        
        # Fetch from BI (Bank Indonesia)
        macro_data['BI_RATE'] = await self._fetch_bi_rate()
        
        # Fetch from commodity markets
        macro_data['OIL_PRICE'] = await self._fetch_commodity('CL=F')
        macro_data['GOLD_PRICE'] = await self._fetch_commodity('GC=F')
        macro_data['VIX'] = await self._fetch_vix()
        
        return macro_data
    
    async def _fetch_fred_data(self, series_id: str) -> pd.Series:
        """Fetch data from FRED API"""
        # Implement actual FRED API call
        pass
    
    async def _fetch_bi_rate(self) -> float:
        """Fetch BI rate from Bank Indonesia"""
        # Implement BI rate scraping
        pass
    
    async def _fetch_commodity(self, symbol: str) -> float:
        """Fetch commodity price"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d")
            return df['Close'].iloc[-1] if not df.empty else None
        except:
            return None
    
    async def _fetch_vix(self) -> float:
        """Fetch VIX index"""
        try:
            ticker = yf.Ticker('^VIX')
            df = ticker.history(period="5d")
            return df['Close'].iloc[-1] if not df.empty else None
        except:
            return None
    
    async def analyze_macro_impact(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Analyze macro impact on specific symbol"""
        macro_data = await self.fetch_macro_data()
        
        # Use LLM to analyze impact
        impact = await self._analyze_with_llm(symbol, macro_data, df)
        
        return {
            'symbol': symbol,
            'macro_data': macro_data,
            'impact_score': impact.get('impact_score', 0),
            'risk_factors': impact.get('risk_factors', []),
            'opportunities': impact.get('opportunities', [])
        }
    
    async def _analyze_with_llm(self, symbol: str, macro_data: Dict, df: pd.DataFrame) -> Dict:
        """Analyze macro impact using LLM"""
        import aiohttp
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Analyze the impact of current macroeconomic conditions on {symbol}.
        
        Macro Data: {macro_data}
        Recent Price: {df['Close'].iloc[-1] if not df.empty else 'N/A'}
        
        Provide analysis in JSON format:
        {{
            "impact_score": -1 to 1,
            "risk_factors": [],
            "opportunities": [],
            "recommended_action": "buy/hold/sell",
            "confidence": 0-1
        }}
        """
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    import json
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"LLM macro analysis failed: {e}")
        
        return {'impact_score': 0, 'risk_factors': [], 'opportunities': []}