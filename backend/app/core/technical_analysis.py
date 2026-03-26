import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from app.core.config import settings

@dataclass
class MarketRegime:
    """Market regime detection"""
    type: str  # 'trending', 'ranging', 'volatile'
    strength: float
    direction: str  # 'up', 'down', 'neutral'

class AdvancedTechnicalV5:
    """Level 5 Technical Analysis dengan Multi-timeframe dan Order Flow"""
    
    def __init__(self):
        self.timeframes = settings.TECH_TIMEFRAMES
        
    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Deteksi market regime menggunakan ADX, ATR, dan volatility"""
        # Handle data insufficient untuk analisis
        if len(df) < 20:
            return MarketRegime(type='ranging', strength=0.5, direction='neutral')
        
        # ADX untuk trend strength
        adx_series = self.adx(df)
        if pd.isna(adx_series.iloc[-1]) or adx_series.iloc[-1] == 0:
            return MarketRegime(type='ranging', strength=0.5, direction='neutral')
        
        adx = adx_series[-1]
        # ATR untuk volatility
        atr = self.atr(df)[-1]
        avg_atr = self.atr(df).rolling(20).mean()
        if len(avg_atr) < 1 or pd.isna(avg_atr.iloc[-1]):
            avg_atr_val = atr  # fallback to current ATR
        else:
            avg_atr_val = avg_atr.iloc[-1]
        
        if adx > 25:
            regime_type = 'trending'
            strength = adx / 100
            # Deteksi arah trend
            ema_20 = df['Close'].rolling(20).mean()
            ema_50 = df['Close'].rolling(50).mean()
            # Handle NaN values
            if ema_20.iloc[-1] > ema_50.iloc[-1] if not pd.isna(ema_50.iloc[-1]) else False:
                direction = 'up'
            else:
                direction = 'down'
        elif avg_atr_val > 0 and atr > avg_atr_val * 1.5:
            regime_type = 'volatile'
            strength = min(atr / avg_atr_val, 1.0)
            direction = 'neutral'
        else:
            regime_type = 'ranging'
            strength = 1 - (adx / 100) if adx > 0 else 0.5
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
        try:
            price_range = df['High'].max() - df['Low'].min()
            if price_range == 0 or pd.isna(price_range):
                return {
                    'profile': [],
                    'poc': {'price_low': 0, 'price_high': 0, 'volume': 0},
                    'value_area_high': 0,
                    'value_area_low': 0
                }
            
            bin_width = price_range / num_bins
            bins = np.arange(df['Low'].min(), df['High'].max() + bin_width, bin_width)
            
            volume_profile = []
            for i in range(len(bins) - 1):
                try:
                    mask = (df['Close'] >= bins[i]) & (df['Close'] < bins[i + 1])
                    volume_col = df.loc[mask, 'Volume']
                    # Ensure Volume is numeric
                    if pd.api.types.is_numeric_dtype(volume_col):
                        volume = volume_col.sum()
                    else:
                        volume = 0
                except Exception:
                    volume = 0
                volume_profile.append({
                    'price_low': bins[i],
                    'price_high': bins[i + 1],
                    'volume': volume
                })
            
            # Find POC (Point of Control)
            if volume_profile:
                poc = max(volume_profile, key=lambda x: x['volume'])
            else:
                poc = {'price_low': 0, 'price_high': 0, 'volume': 0}
            
            return {
                'profile': volume_profile,
                'poc': poc,
                'value_area_high': bins[-10] if len(bins) > 10 else bins[-1],
                'value_area_low': bins[10] if len(bins) > 10 else bins[0]
            }
        except Exception as e:
            return {
                'profile': [],
                'poc': {'price_low': 0, 'price_high': 0, 'volume': 0},
                'value_area_high': 0,
                'value_area_low': 0
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
        
        if not results:
            return {
                'timeframes': {},
                'trend_alignment': True,
                'strongest_trend': None
            }

        # Konfirmasi trend antar timeframe
        base_tf = '1d' if '1d' in results else next(iter(results.keys()))
        if len(results) <= 1:
            trend_alignment = True
        else:
            trend_alignment = all(
                results[tf]['trend'] == results[base_tf]['trend']
                for tf in results if tf != base_tf
            )
        
        return {
            'timeframes': results,
            'trend_alignment': trend_alignment,
            'strongest_trend': max(results.items(), key=lambda x: x[1]['regime'].strength)[0]
        }
    
    def detect_trend(self, df: pd.DataFrame) -> str:
        """Deteksi trend menggunakan multiple moving averages"""
        # Handle insufficient data
        if len(df) < 50:
            return 'neutral'
        
        ema_20 = df['Close'].rolling(20).mean()
        ema_50 = df['Close'].rolling(50).mean()
        ema_200 = df['Close'].rolling(200).mean()
        
        # Handle NaN values
        try:
            val_20 = ema_20.iloc[-1]
            val_50 = ema_50.iloc[-1]
            val_200 = ema_200.iloc[-1]
            
            if pd.isna(val_20) or pd.isna(val_50):
                return 'neutral'
            
            if val_20 > val_50:
                if not pd.isna(val_200) and val_50 > val_200:
                    return 'strong_bullish'
                return 'bullish'
            elif val_20 < val_50:
                if not pd.isna(val_200) and val_50 < val_200:
                    return 'strong_bearish'
                return 'bearish'
        except Exception:
            pass
        
        return 'neutral'
    
    def find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Find support and resistance levels"""
        try:
            # Handle insufficient data
            if len(df) < window * 2:
                return {
                    'resistance': [],
                    'support': [],
                    'current_price': df['Close'].iloc[-1] if len(df) > 0 else 0,
                    'nearest_resistance': None,
                    'nearest_support': None
                }
            
            # Pivot points
            pivot_highs = []
            pivot_lows = []
            
            for i in range(window, len(df) - window):
                try:
                    if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
                        pivot_highs.append(float(df['High'].iloc[i]))
                    if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
                        pivot_lows.append(float(df['Low'].iloc[i]))
                except Exception:
                    continue
            
            # Cluster similar levels
            resistance = self._cluster_levels(pivot_highs[-5:] if len(pivot_highs) > 5 else pivot_highs)
            support = self._cluster_levels(pivot_lows[-5:] if len(pivot_lows) > 5 else pivot_lows)
            
            current_price = float(df['Close'].iloc[-1]) if len(df) > 0 else 0
            
            return {
                'resistance': resistance,
                'support': support,
                'current_price': current_price,
                'nearest_resistance': min(resistance, key=lambda x: abs(x - current_price)) if resistance else None,
                'nearest_support': min(support, key=lambda x: abs(x - current_price)) if support else None
            }
        except Exception as e:
            return {
                'resistance': [],
                'support': [],
                'current_price': 0,
                'nearest_resistance': None,
                'nearest_support': None
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


class TechnicalAnalysis:
    """Lightweight technical indicators used by ModelTrainer."""
    def rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    def bollinger_bands(self, series: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series]:
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
