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
        try:
            regime = self.technical.detect_market_regime(df)
        except Exception as e:
            regime = None
        
        try:
            trend = self.technical.detect_trend(df)
        except Exception as e:
            trend = 'neutral'
        
        try:
            sr_levels = self.technical.find_support_resistance(df)
        except Exception as e:
            sr_levels = {'nearest_support': None, 'nearest_resistance': None}
        
        try:
            volume_profile = self.technical.volume_profile(df)
        except Exception as e:
            volume_profile = {'poc': {'price_low': 0, 'price_high': 0}, 'profile': []}
        
        # Score calculation
        bullish_score = 0
        bearish_score = 0
        
        # Trend score
        if 'bullish' in trend:
            bullish_score += 2
        elif 'bearish' in trend:
            bearish_score += 2
        
        # Support/Resistance
        try:
            current_price = df['Close'].iloc[-1]
            if sr_levels.get('nearest_support') and current_price < sr_levels['nearest_support'] * 1.02:
                bullish_score += 1
            if sr_levels.get('nearest_resistance') and current_price > sr_levels['nearest_resistance'] * 0.98:
                bearish_score += 1
            
            # Volume Profile
            poc = volume_profile.get('poc', {})
            if poc and current_price < poc.get('price_low', 0):
                bullish_score += 1
            elif poc and current_price > poc.get('price_high', 0):
                bearish_score += 1
        except Exception:
            pass  # Skip if current_price access fails
        
        return {
            'regime': regime,
            'trend': trend,
            'support_resistance': sr_levels,
            'volume_profile': volume_profile,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'signal': 'bullish' if bullish_score > bearish_score else 'bearish' if bearish_score > bullish_score else 'neutral',
            'strength': max(bullish_score, bearish_score) / 5 if max(bullish_score, bearish_score) > 0 else 0.1
        }
    
    async def _analyze_sentiment(self, symbol: str) -> Dict:
        """Comprehensive sentiment analysis"""
        try:
            sentiment_data = await self.sentiment.analyze_symbol(symbol)
            return {
                'score': sentiment_data.get('avg_score', 0),
                'confidence': sentiment_data.get('confidence', 0),
                'sentiment': sentiment_data.get('sentiment', 'neutral'),
                'news_count': sentiment_data.get('news_count', 0),
                'details': sentiment_data.get('details', [])
            }
        except Exception as e:
            return {
                'score': 0,
                'confidence': 0,
                'sentiment': 'neutral',
                'news_count': 0,
                'details': []
            }
    
    async def _analyze_institutional(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Comprehensive institutional analysis"""
        try:
            bander_data = await self.bandar.analyze(symbol, df)
            return {
                'verdict': bander_data.get('verdict', 'unknown'),
                'strength': bander_data.get('strength', 0),
                'accumulation': bander_data.get('accumulation', {}).get('is_accumulating', False),
                'distribution': bander_data.get('distribution', {}).get('is_distributing', False),
                'flow': bander_data.get('institutional_flow', 'unknown')
            }
        except Exception as e:
            return {
                'verdict': 'unknown',
                'strength': 0,
                'accumulation': False,
                'distribution': False,
                'flow': 'unknown'
            }
    
    def _calculate_entry_exit(self, df: pd.DataFrame, technical: Dict, sentiment: Dict, 
                              institutional: Dict, action: str) -> Tuple[float, float, float]:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            current_price = float(df['Close'].iloc[-1])
        except Exception:
            current_price = 0
        
        try:
            atr_series = self.technical.atr(df)
            atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else current_price * 0.02
        except Exception:
            atr = current_price * 0.02
        
        if action == 'buy':
            # Entry: nearest support or FVG
            sr = technical.get('support_resistance', {})
            entry = sr.get('nearest_support') or current_price
            if entry is None:
                entry = current_price
            entry = min(float(entry), current_price * 1.01)  # Don't chase too high
            
            # Stop Loss: 1.5x ATR below entry
            stop_loss = entry - (atr * 1.5)
            
            # Take Profit: 2.5x risk
            risk = entry - stop_loss
            take_profit = entry + (risk * 2.5)
            
        elif action == 'sell':
            # Entry: nearest resistance
            sr = technical.get('support_resistance', {})
            entry = sr.get('nearest_resistance') or current_price
            if entry is None:
                entry = current_price
            entry = max(float(entry), current_price * 0.99)  # Don't short too low
            
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
        
        try:
            technical, sentiment, institutional = await asyncio.gather(
                technical_task, sentiment_task, institutional_task
            )
        except Exception as e:
            # Return neutral signal on error
            return Signal(
                symbol=symbol,
                action='hold',
                bias='neutral',
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                confidence=0,
                risk_reward=0,
                time_horizon='swing',
                reasoning={'error': str(e)},
                timestamp=pd.Timestamp.now().isoformat()
            )
        
        # Adjust weights based on market regime
        regime = technical.get('regime')
        if regime is not None:
            weights = self._adjust_weights(regime)
        else:
            weights = self._adjust_weights(MarketRegime(type='ranging', strength=0.5, direction='neutral'))
        
        # Calculate combined score
        # Technical score
        tech_score = 1 if technical['signal'] == 'bullish' else -1 if technical['signal'] == 'bearish' else 0
        tech_score *= technical.get('strength', 0.1)
        
        # Sentiment score
        sent_score = sentiment.get('score', 0)
        
        # Institutional score
        inst_score = 1 if institutional.get('verdict') == 'accumulating' else -1 if institutional.get('verdict') == 'distributing' else 0
        inst_score *= institutional.get('strength', 0.1)
        
        # Weighted final score
        final_score = (
            tech_score * weights.get('technical', 0.4) +
            sent_score * weights.get('sentiment', 0.3) +
            inst_score * weights.get('institutional', 0.3)
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
        if regime is not None and regime.type == 'trending' and regime.strength > 0.7:
            time_horizon = 'position'  # weeks to months
        elif regime is not None and regime.type == 'volatile':
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
                    'regime': regime.type if regime else 'unknown',
                    'trend': technical.get('trend', 'neutral'),
                    'strength': technical.get('strength', 0)
                },
                'sentiment': {
                    'score': sentiment.get('score', 0),
                    'confidence': sentiment.get('confidence', 0),
                    'news_count': sentiment.get('news_count', 0)
                },
                'institutional': {
                    'verdict': institutional.get('verdict', 'unknown'),
                    'strength': institutional.get('strength', 0),
                    'flow': institutional.get('flow', 'unknown')
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
