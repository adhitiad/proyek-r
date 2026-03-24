import itertools
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.backtester import Backtester
from app.core.signal_generator import SignalGenerator

class ParameterOptimizer:
    def __init__(self, symbol, start_date, end_date, initial_capital=100000000, 
                 commission=0.001, slippage=0.0005):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run_backtest(self, params):
        try:
            signal_gen = SignalGenerator(capital=self.initial_capital, params=params)
            bt = Backtester(
                self.symbol, self.start_date, self.end_date,
                initial_capital=self.initial_capital,
                risk_per_trade=params['risk_per_trade'],
                commission=self.commission,
                slippage=self.slippage,
                signal_generator=signal_gen
            )
            result = bt.run()
            return {'params': params, **result}
        except Exception as e:
            return {'params': params, 'error': str(e)}

    def optimize(self, param_grid, metric='sharpe_ratio', max_workers=4, 
                 min_win_rate=0.4, max_drawdown_limit=-0.3, min_trades=10):
        """
        Optimasi parameter dengan filter metrik.
        
        Args:
            param_grid: dict parameter yang akan diuji
            metric: metrik utama untuk optimasi ('sharpe_ratio', 'total_return', 'profit_factor', 
                    'win_rate', 'expectancy')
            max_workers: jumlah thread paralel
            min_win_rate: minimum win rate yang diterima
            max_drawdown_limit: batas drawdown maksimum (negatif)
            min_trades: minimum jumlah trade
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.run_backtest, combo): combo for combo in combinations}
            for future in as_completed(futures):
                result = future.result()
                # Filter berdasarkan kriteria minimum
                if (result.get('win_rate', 0) >= min_win_rate and 
                    result.get('max_drawdown', 0) >= max_drawdown_limit and
                    result.get('num_trades', 0) >= min_trades):
                    results.append(result)
        
        # Urutkan berdasarkan metrik yang dipilih
        results.sort(key=lambda x: x.get(metric, -np.inf), reverse=True)
        best = results[0] if results else None
        
        # Generate summary of top 10 parameter sets
        top_10 = results[:10] if results else []
        
        return {
            'best_params': best['params'] if best else None,
            'best_metrics': {
                'total_return': best.get('total_return', 0),
                'sharpe_ratio': best.get('sharpe_ratio', 0),
                'win_rate': best.get('win_rate', 0),
                'max_drawdown': best.get('max_drawdown', 0),
                'profit_factor': best.get('profit_factor', 0),
                'expectancy': best.get('expectancy', 0),
                'num_trades': best.get('num_trades', 0)
            } if best else None,
            'top_10_summary': [
                {
                    'params': r['params'],
                    'total_return': r.get('total_return', 0),
                    'sharpe_ratio': r.get('sharpe_ratio', 0),
                    'win_rate': r.get('win_rate', 0),
                    'max_drawdown': r.get('max_drawdown', 0),
                    'profit_factor': r.get('profit_factor', 0),
                    'expectancy': r.get('expectancy', 0),
                    'num_trades': r.get('num_trades', 0)
                } for r in top_10
            ],
            'total_combinations': len(combinations),
            'valid_combinations': len(results)
        }