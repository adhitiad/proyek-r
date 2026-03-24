import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import asyncio
from dataclasses import dataclass
from app.core.technical_analysis import AdvancedTechnicalV5, MarketRegime
from app.core.sentiment_analysis import SentimentAnalyzerV5
from app.core.bandar_detector import BandarDetectorV5

@dataclass
class Signal:
    """Structured trading signal"""
    symbol: str
    action: str  # buy, sell, hold
    bias: str  # bullish, bearish, neutral
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-1
    risk_reward: float
    time_horizon: str  # intraday, swing, position
    reasoning: Dict
    timestamp: str

class SignalGeneratorV5:
    """Level 5 Signal Generator dengan ensemble dan adaptive weighting"""
    
    def __init__(
        self,
        technical: Optional[AdvancedTechnicalV5] = None,
        sentiment: Optional[SentimentAnalyzerV5] = None,
        bandar: Optional[BandarDetectorV5] = None
    ):
        self.technical = technical or AdvancedTechnicalV5()
        self.sentiment = sentiment or SentimentAnalyzerV5()
        self.bandar = bandar or BandarDetectorV5()
        
        # Adaptive weights based on market regime
        self.base_weights = {
            'technical': 0.4,
            'sentiment': 0.3,
            'institutional': 0.3
        }
    
    def _adjust_weights(self, regime: MarketRegime) -> Dict:
        """Adjust weights based on market regime"""
        weights = self.base_weights.copy()
        
        if regime.type == 'trending':
            weights['technical'] += 0.1
            weights['institutional'] -= 0.05
            weights['sentiment'] -= 0.05
        elif regime.type == 'volatile':
            weights['sentiment'] += 0.1
            weights['technical'] -= 0.05
            weights['institutional'] -= 0.05
        elif regime.type == 'ranging':
            weights['institutional'] += 0.1
            weights['technical'] -= 0.05
            weights['sentiment'] -= 0.05
        
        return weights
    
    async def _analyze_technical(self, df: pd.DataFrame) -> Dict:
        """Comprehensive technical analysis"""
        regime = self.technical.detect_market_regime(df)
        trend = self.technical.detect_trend(df)
        sr_levels = self.technical.find_support_resistance(df)
        volume_profile = self.technical.volume_profile(df)
        
        # Score calculation
        bullish_score = 0
        bearish_score = 0
        
        # Trend score
        if 'bullish' in trend:
            bullish_score += 2
        elif 'bearish' in trend:
            bearish_score += 2
        
        # Support/Resistance
        if sr_levels['nearest_support'] and df['Close'].iloc[-1] < sr_levels['nearest_support'] * 1.02:
            bullish_score += 1
        if sr_levels['nearest_resistance'] and df['Close'].iloc[-1] > sr_levels['nearest_resistance'] * 0.98:
            bearish_score += 1
        
        # Volume Profile
        current_price = df['Close'].iloc[-1]
        if current_price < volume_profile['poc']['price_low']:
            bullish_score += 1
        elif current_price > volume_profile['poc']['price_high']:
            bearish_score += 1
        
        return {
            'regime': regime,
            'trend': trend,
            'support_resistance': sr_levels,
            'volume_profile': volume_profile,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'signal': 'bullish' if bullish_score > bearish_score else 'bearish' if bearish_score > bullish_score else 'neutral',
            'strength': max(bullish_score, bearish_score) / 5
        }
    
    async def _analyze_sentiment(self, symbol: str) -> Dict:
        """Comprehensive sentiment analysis"""
        sentiment_data = await self.sentiment.analyze_symbol(symbol)
        
        return {
            'score': sentiment_data['avg_score'],
            'confidence': sentiment_data['confidence'],
            'sentiment': sentiment_data['sentiment'],
            'news_count': sentiment_data['news_count'],
            'details': sentiment_data.get('details', [])
        }
    
    async def _analyze_institutional(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Comprehensive institutional analysis"""
        bandar_data = await self.bandar.analyze(symbol, df)
        
        return {
            'verdict': bandar_data['verdict'],
            'strength': bandar_data['strength'],
            'accumulation': bandar_data['accumulation']['is_accumulating'],
            'distribution': bandar_data['distribution']['is_distributing'],
            'flow': bandar_data['institutional_flow']
        }
    
    def _calculate_entry_exit(self, df: pd.DataFrame, technical: Dict, sentiment: Dict, 
                              institutional: Dict, action: str) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and take profit levels"""
        current_price = df['Close'].iloc[-1]
        atr = self.technical.atr(df).iloc[-1]
        
        if action == 'buy':
            # Entry: nearest support or FVG
            entry = technical['support_resistance']['nearest_support'] or current_price
            entry = min(entry, current_price * 1.01)  # Don't chase too high
            
            # Stop Loss: 1.5x ATR below entry
            stop_loss = entry - (atr * 1.5)
            
            # Take Profit: 2.5x risk
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
            
        elif action == 'sell':
            # Entry: nearest resistance
            entry = technical['support_resistance']['nearest_resistance'] or current_price
            entry = max(entry, current_price * 0.99)  # Don't short too low
            
            # Stop Loss: 1.5x ATR above entry
            stop_loss = entry + (atr * 1.5)
            
            # Take Profit: 2.5x risk
            risk = stop_loss - entry
            take_profit = entry - (risk * 2.5)
        else:
            entry = current_price
            stop_loss = entry * 0.98 if current_price > 0 else entry * 1.02
            take_profit = entry * 1.04 if current_price > 0 else entry * 0.96
        
        return entry, stop_loss, take_profit
    
    async def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """Generate comprehensive trading signal"""
        # Analyze all components in parallel
        technical_task = self._analyze_technical(df)
        sentiment_task = self._analyze_sentiment(symbol)
        institutional_task = self._analyze_institutional(symbol, df)
        
        technical, sentiment, institutional = await asyncio.gather(
            technical_task, sentiment_task, institutional_task
        )
        
        # Adjust weights based on market regime
        weights = self._adjust_weights(technical['regime'])
        
        # Calculate combined score
        # Technical score
        tech_score = 1 if technical['signal'] == 'bullish' else -1 if technical['signal'] == 'bearish' else 0
        tech_score *= technical['strength']
        
        # Sentiment score
        sent_score = sentiment['score']
        
        # Institutional score
        inst_score = 1 if institutional['verdict'] == 'accumulating' else -1 if institutional['verdict'] == 'distributing' else 0
        inst_score *= institutional['strength']
        
        # Weighted final score
        final_score = (
            tech_score * weights['technical'] +
            sent_score * weights['sentiment'] +
            inst_score * weights['institutional']
        )
        
        # Determine action
        if final_score > 0.3:
            action = 'buy'
            bias = 'bullish'
        elif final_score < -0.3:
            action = 'sell'
            bias = 'bearish'
        else:
            action = 'hold'
            bias = 'neutral'
        
        # Calculate confidence
        confidence = abs(final_score)
        
        # Calculate entry, SL, TP
        entry, stop_loss, take_profit = self._calculate_entry_exit(
            df, technical, sentiment, institutional, action
        )
        
        # Risk reward ratio
        if action == 'buy':
            risk = entry - stop_loss
            reward = take_profit - entry
        elif action == 'sell':
            risk = stop_loss - entry
            reward = entry - take_profit
        else:
            risk = reward = 1
        
        risk_reward = reward / risk if risk > 0 else 0
        
        # Determine time horizon
        if technical['regime'].type == 'trending' and technical['regime'].strength > 0.7:
            time_horizon = 'position'  # weeks to months
        elif technical['regime'].type == 'volatile':
            time_horizon = 'intraday'  # hours to days
        else:
            time_horizon = 'swing'  # days to weeks
        
        return Signal(
            symbol=symbol,
            action=action,
            bias=bias,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            risk_reward=risk_reward,
            time_horizon=time_horizon,
            reasoning={
                'technical': {
                    'regime': technical['regime'].type,
                    'trend': technical['trend'],
                    'strength': technical['strength']
                },
                'sentiment': {
                    'score': sentiment['score'],
                    'confidence': sentiment['confidence'],
                    'news_count': sentiment['news_count']
                },
                'institutional': {
                    'verdict': institutional['verdict'],
                    'strength': institutional['strength'],
                    'flow': institutional['flow']
                },
                'weights': weights,
                'final_score': final_score
            },
            timestamp=pd.Timestamp.now().isoformat()
        )


def signal_to_legacy_dict(signal: Signal) -> Dict:
    """Map Signal (V5) to legacy dict format expected by older modules."""
    reasoning = signal.reasoning or {}
    technical = reasoning.get('technical', {})
    sentiment = reasoning.get('sentiment', {})
    institutional = reasoning.get('institutional', {})

    notes_parts = []
    if technical:
        notes_parts.append(f"Trend {technical.get('trend', 'n/a')}")
        regime = technical.get('regime')
        if regime:
            notes_parts.append(f"Regime {regime}")
    if sentiment:
        notes_parts.append(f"Sentiment score {sentiment.get('score', 0):.2f}")
    if institutional:
        notes_parts.append(f"Institutional {institutional.get('verdict', 'n/a')}")

    notes = " | ".join(notes_parts) if notes_parts else "Auto-generated signal."

    return {
        'symbol': signal.symbol,
        'bias': signal.bias,
        'action': signal.action,
        'action_type': 'market',
        'entry_zone': float(signal.entry_price),
        'stop_loss_1': float(signal.stop_loss) if signal.stop_loss is not None else None,
        'stop_loss_2': None,
        'take_profit_1': float(signal.take_profit) if signal.take_profit is not None else None,
        'take_profit_2': None,
        'risk_reward': f"{signal.risk_reward:.2f}",
        'probability': int(round(signal.confidence * 100)),
        'notes': notes
    }


class SignalGenerator:
    """Legacy-compatible signal generator wrapper around V5."""
    def __init__(
        self,
        capital: float = 100000000,
        params: Optional[Dict] = None,
        v5: Optional[SignalGeneratorV5] = None
    ):
        self.capital = capital
        self.params = params or {}
        self._v5 = v5 or SignalGeneratorV5()

    @property
    def sentiment(self):
        return self._v5.sentiment

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Dict:
        signal = asyncio.run(self._v5.generate_signal(symbol, df))
        return signal_to_legacy_dict(signal)
