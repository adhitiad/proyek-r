import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from app.ml.gans import MarketScenarioGenerator
from app.ml.trading_agent import DQNTradingAgent, TradingEnvironment
from app.ml.detector import SelfHealingSystem, SystemHealthMonitor
from app.core.macro_integration import CrossAssetCorrelation, GlobalMacroIntegration
from app.core.signal_generator import SignalGeneratorV5, Signal

@dataclass
class Level6Signal(Signal):
    """Level 6 signal dengan generative & predictive capabilities"""
    scenario_analysis: Dict  # GAN-generated scenarios
    rl_optimized_params: Dict  # DRL optimized parameters
    macro_impact: Dict  # Global macro integration
    cross_asset_hedge: List[Dict]  # Cross-asset hedge suggestions
    confidence_distribution: Dict  # Confidence across scenarios

class SignalGeneratorV6:
    """Level 6 – Predictive & Generative Signal Generator"""
    
    def __init__(self, groq_api_key: str):
        self.v5_generator = SignalGeneratorV5()
        self.scenario_gen = MarketScenarioGenerator(groq_api_key)
        self.rl_agent = DQNTradingAgent(state_dim=1)  # Will be reinitialized
        self.self_healing = SelfHealingSystem()
        self.cross_asset = CrossAssetCorrelation()
        self.macro = GlobalMacroIntegration(groq_api_key)
        
        # Register monitors
        self._setup_healing_monitors()
    
    def _setup_healing_monitors(self):
        """Setup self-healing monitors"""
        monitor = SystemHealthMonitor()
        self.self_healing.register_monitor('price_data', monitor)
    
    async def generate_scenario_analysis(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Generate market scenarios using GAN + LLM"""
        scenarios = []
        
        # Generate different market scenarios
        scenario_prompts = [
            "Generate a bullish market scenario with high volume",
            "Generate a bearish market scenario with increasing volatility",
            "Generate a ranging market scenario with low volatility",
            "Generate a crisis scenario with extreme volatility",
            "Generate a recovery scenario after a downturn"
        ]
        
        for prompt in scenario_prompts[:3]:  # Limit to 3 for performance
            scenario = await self.scenario_gen.generate_scenario(prompt, df)
            scenarios.append(scenario)
        
        # Calculate confidence across scenarios
        bullish_count = sum(1 for s in scenarios if s['parameters'].get('trend') == 'bullish')
        bearish_count = sum(1 for s in scenarios if s['parameters'].get('trend') == 'bearish')
        
        return {
            'scenarios': scenarios,
            'bullish_probability': bullish_count / len(scenarios),
            'bearish_probability': bearish_count / len(scenarios),
            'worst_case': min(scenarios, key=lambda x: x['portfolio_impact']['expected_return']),
            'best_case': max(scenarios, key=lambda x: x['portfolio_impact']['expected_return'])
        }
    
    async def optimize_with_rl(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Optimize strategy using Deep Reinforcement Learning"""
        env = TradingEnvironment(df)
        self.rl_agent = DQNTradingAgent(state_dim=len(env.reset()))
        result = self.rl_agent.optimize_strategy(df)
        return result
    
    async def analyze_macro_impact(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Analyze global macro impact"""
        return await self.macro.analyze_macro_impact(symbol, df)
    
    async def get_hedge_suggestions(self, symbol: str) -> List[Dict]:
        """Get cross-asset hedge suggestions"""
        await self.cross_asset.calculate_correlations()
        return self.cross_asset.get_hedge_suggestion(symbol)
    
    async def generate_signal(self, symbol: str, df: pd.DataFrame) -> Level6Signal:
        """Generate Level 6 signal with generative and predictive capabilities"""
        
        # 1. Self-healing: Clean anomalies
        healed_df = await self.self_healing.check_and_heal({symbol: df})
        clean_df = healed_df.get(symbol, df)
        
        # 2. Generate Level 5 base signal
        base_signal = await self.v5_generator.generate_signal(symbol, clean_df)
        
        # 3. Run all Level 6 analyses in parallel
        scenario_task = self.generate_scenario_analysis(symbol, clean_df)
        rl_task = self.optimize_with_rl(symbol, clean_df)
        macro_task = self.analyze_macro_impact(symbol, clean_df)
        hedge_task = self.get_hedge_suggestions(symbol)
        
        scenario_analysis, rl_optimized, macro_impact, cross_asset_hedge = await asyncio.gather(
            scenario_task, rl_task, macro_task, hedge_task
        )
        
        # 4. Adjust signal based on scenario analysis
        if scenario_analysis['bullish_probability'] > 0.6 and base_signal.action != 'buy':
            base_signal.action = 'buy'
            base_signal.confidence *= 1.2
        elif scenario_analysis['bearish_probability'] > 0.6 and base_signal.action != 'sell':
            base_signal.action = 'sell'
            base_signal.confidence *= 1.2
        
        # 5. Adjust based on RL optimization
        if rl_optimized['performance']['sharpe_ratio'] > 1:
            base_signal.confidence *= 1.1
        
        # 6. Adjust based on macro impact
        base_signal.confidence *= (1 + macro_impact.get('impact_score', 0) * 0.2)
        
        # 7. Cap confidence
        base_signal.confidence = min(base_signal.confidence, 0.95)
        
        # 8. Create Level 6 signal
        return Level6Signal(
            symbol=base_signal.symbol,
            action=base_signal.action,
            bias=base_signal.bias,
            entry_price=base_signal.entry_price,
            stop_loss=base_signal.stop_loss,
            take_profit=base_signal.take_profit,
            confidence=base_signal.confidence,
            risk_reward=base_signal.risk_reward,
            time_horizon=base_signal.time_horizon,
            reasoning=base_signal.reasoning,
            timestamp=base_signal.timestamp,
            scenario_analysis=scenario_analysis,
            rl_optimized_params=rl_optimized,
            macro_impact=macro_impact,
            cross_asset_hedge=cross_asset_hedge,
            confidence_distribution={
                'base': base_signal.confidence,
                'scenario_boost': scenario_analysis['bullish_probability'] if base_signal.action == 'buy' else scenario_analysis['bearish_probability'],
                'macro_adjustment': macro_impact.get('impact_score', 0),
                'rl_enhancement': rl_optimized['performance']['sharpe_ratio'] if rl_optimized['performance'] else 0
            }
        )