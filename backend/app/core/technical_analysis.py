import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketRegime:
    """Market regime detection"""
    type: str  # 'trending', 'ranging', 'volatile'
    strength: float
    direction: str  # 'up', 'down', 'neutral'

class AdvancedTechnicalV5:
    """Level 5 Technical Analysis dengan Multi-timeframe dan Order Flow"""
    
    def __init__(self):
        self.timeframes = ['1d', '4h', '1h']
        
    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Deteksi market regime menggunakan ADX, ATR, dan volatility"""
        # ADX untuk trend strength
        adx = self.adx(df)[-1]
        # ATR untuk volatility
        atr = self.atr(df)[-1]
        avg_atr = self.atr(df).rolling(20).mean()[-1]
        
        if adx > 25:
            regime_type = 'trending'
            strength = adx / 100
            # Deteksi arah trend
            ema_20 = df['Close'].rolling(20).mean()
            ema_50 = df['Close'].rolling(50).mean()
            direction = 'up' if ema_20.iloc[-1] > ema_50.iloc[-1] else 'down'
        elif atr > avg_atr * 1.5:
            regime_type = 'volatile'
            strength = atr / avg_atr
            direction = 'neutral'
        else:
            regime_type = 'ranging'
            strength = 1 - (adx / 100)
            direction = 'neutral'
        
        return MarketRegime(regime_type, strength, direction)
    
    def adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        
        return adx
    
    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def volume_profile(self, df: pd.DataFrame, num_bins: int = 30) -> Dict:
        """Volume Profile Analysis"""
        price_range = df['High'].max() - df['Low'].min()
        bin_width = price_range / num_bins
        bins = np.arange(df['Low'].min(), df['High'].max() + bin_width, bin_width)
        
        volume_profile = []
        for i in range(len(bins) - 1):
            mask = (df['Close'] >= bins[i]) & (df['Close'] < bins[i + 1])
            volume = df.loc[mask, 'Volume'].sum()
            volume_profile.append({
                'price_low': bins[i],
                'price_high': bins[i + 1],
                'volume': volume
            })
        
        # Find POC (Point of Control)
        poc = max(volume_profile, key=lambda x: x['volume'])
        
        return {
            'profile': volume_profile,
            'poc': poc,
            'value_area_high': bins[-10] if len(bins) > 10 else bins[-1],
            'value_area_low': bins[10] if len(bins) > 10 else bins[0]
        }
    
    def order_flow_imbalance(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """Order flow imbalance berdasarkan volume dan price change"""
        price_change = df['Close'].diff()
        volume_delta = df['Volume'] * np.sign(price_change)
        return volume_delta.rolling(window).sum()
    
    def multi_timeframe_analysis(self, dfs: Dict[str, pd.DataFrame]) -> Dict:
        """Analisis multi-timeframe"""
        results = {}
        for tf, df in dfs.items():
            regime = self.detect_market_regime(df)
            results[tf] = {
                'regime': regime,
                'trend': self.detect_trend(df),
                'support_resistance': self.find_support_resistance(df)
            }
        
        # Konfirmasi trend antar timeframe
        trend_alignment = all(
            results[tf]['trend'] == results['1d']['trend'] 
            for tf in results if tf != '1d'
        )
        
        return {
            'timeframes': results,
            'trend_alignment': trend_alignment,
            'strongest_trend': max(results.items(), key=lambda x: x[1]['regime'].strength)[0]
        }
    
    def detect_trend(self, df: pd.DataFrame) -> str:
        """Deteksi trend menggunakan multiple moving averages"""
        ema_20 = df['Close'].rolling(20).mean()
        ema_50 = df['Close'].rolling(50).mean()
        ema_200 = df['Close'].rolling(200).mean()
        
        if ema_20.iloc[-1] > ema_50.iloc[-1] > ema_200.iloc[-1]:
            return 'strong_bullish'
        elif ema_20.iloc[-1] > ema_50.iloc[-1]:
            return 'bullish'
        elif ema_20.iloc[-1] < ema_50.iloc[-1] < ema_200.iloc[-1]:
            return 'strong_bearish'
        elif ema_20.iloc[-1] < ema_50.iloc[-1]:
            return 'bearish'
        return 'neutral'
    
    def find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Find support and resistance levels"""
        # Pivot points
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(df) - window):
            if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
                pivot_highs.append(df['High'].iloc[i])
            if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
                pivot_lows.append(df['Low'].iloc[i])
        
        # Cluster similar levels
        resistance = self._cluster_levels(pivot_highs[-5:] if len(pivot_highs) > 5 else pivot_highs)
        support = self._cluster_levels(pivot_lows[-5:] if len(pivot_lows) > 5 else pivot_lows)
        
        return {
            'resistance': resistance,
            'support': support,
            'current_price': df['Close'].iloc[-1],
            'nearest_resistance': min(resistance, key=lambda x: abs(x - df['Close'].iloc[-1])) if resistance else None,
            'nearest_support': min(support, key=lambda x: abs(x - df['Close'].iloc[-1])) if support else None
        }
    
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.01) -> List[float]:
        """Cluster price levels yang berdekatan"""
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level / current_cluster[-1] - 1 < tolerance:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters