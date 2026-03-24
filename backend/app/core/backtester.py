import pandas as pd
import numpy as np
from datetime import datetime

from app.core.data_collector import DataSourceManager
from app.core.signal_generator import SignalGeneratorV5
from app.core.risk_manager import RiskManager

class Backtester:
    def __init__(self, symbol, start_date, end_date, initial_capital=100000000,
                 risk_per_trade=0.02, commission=0.001, slippage=0.0005, signal_generator=None):
        self.symbol = symbol
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.risk_mgr = RiskManager(capital=initial_capital, risk_per_trade=risk_per_trade)
        self.signal_generator = signal_generator
        self.trades = []  # list of trade dicts
        self.daily_equity = []
        self.current_position = None
        self.df = None
        self.equity_curve = []  # untuk drawdown tracking

    def load_data(self):
        self.df = DataSourceManager().get_price_data(self.symbol, period="3mo", interval="1d")
        if self.df.empty:
            raise ValueError("No data retrieved")
        self.df = self.df[(self.df.index >= self.start_date) & (self.df.index <= self.end_date)]
        if len(self.df) == 0:
            raise ValueError("No data within date range")
        # Generate signals for each bar
        self.signals = []
        for i in range(20, len(self.df)):
            data_slice = self.df.iloc[:i+1]
            signal = self.signal_generator.generate_signal(self.symbol, data_slice)
            self.signals.append((self.df.index[i], signal))

    def run(self):
        self.load_data()
        for idx, (date, signal) in enumerate(self.signals):
            price = self.df.loc[date, 'Close']
            # Jika ada posisi terbuka, cek apakah stop loss atau take profit tercapai
            if self.current_position:
                self.check_exit(date, price)

            # Jika tidak ada posisi, cek apakah ada sinyal entry
            if not self.current_position and signal['action'] != 'hold':
                self.enter_position(date, signal, price)

            # Catat equity harian
            equity = self.capital
            if self.current_position:
                position_value = self.current_position['shares'] * price
                equity = self.capital + (position_value - self.current_position['shares'] * self.current_position['entry_price'])
            self.daily_equity.append((date, equity))
            self.equity_curve.append(equity)

        # Tutup posisi di akhir periode
        if self.current_position:
            self.close_position(self.df.index[-1], self.df.loc[self.df.index[-1], 'Close'], reason="end_of_period")

        return self.calculate_metrics()

    def enter_position(self, date, signal, current_price):
        action = signal['action']
        entry_zone = signal['entry_zone']
        stop_loss = signal['stop_loss_1']  # gunakan SL pertama
        take_profit = signal['take_profit_1']

        # Tentukan harga eksekusi berdasarkan action_type
        exec_price = current_price
        if signal['action_type'] == 'limit':
            if action == 'buy':
                if current_price <= entry_zone:
                    exec_price = entry_zone
                else:
                    return
            else:  # sell
                if current_price >= entry_zone:
                    exec_price = entry_zone
                else:
                    return
        elif signal['action_type'] == 'stop':
            if action == 'buy':
                if current_price >= entry_zone:
                    exec_price = entry_zone
                else:
                    return
            else:
                if current_price <= entry_zone:
                    exec_price = entry_zone
                else:
                    return

        # Tambahkan slippage
        if action == 'buy':
            exec_price = exec_price * (1 + self.slippage)
        else:
            exec_price = exec_price * (1 - self.slippage)

        # Hitung jumlah saham/unit berdasarkan risk
        risk_amount = self.capital * self.risk_per_trade
        if stop_loss is not None:
            risk_per_share = abs(exec_price - stop_loss)
        else:
            risk_per_share = exec_price * 0.02  # default 2%
        if risk_per_share <= 0:
            return
        shares = risk_amount / risk_per_share
        shares = int(shares)
        if shares == 0:
            return

        # Komisi
        commission_cost = shares * exec_price * self.commission
        if commission_cost > self.capital:
            return
        self.capital -= commission_cost
        self.capital -= shares * exec_price

        self.current_position = {
            'symbol': self.symbol,
            'action': action,
            'entry_date': date,
            'entry_price': exec_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_zone': entry_zone,
            'action_type': signal['action_type']
        }

    def check_exit(self, date, current_price):
        pos = self.current_position
        if pos['action'] == 'buy':
            if pos['stop_loss'] is not None and current_price <= pos['stop_loss']:
                self.close_position(date, current_price, reason="stop_loss")
            elif pos['take_profit'] is not None and current_price >= pos['take_profit']:
                self.close_position(date, current_price, reason="take_profit")
        else:  # sell
            if pos['stop_loss'] is not None and current_price >= pos['stop_loss']:
                self.close_position(date, current_price, reason="stop_loss")
            elif pos['take_profit'] is not None and current_price <= pos['take_profit']:
                self.close_position(date, current_price, reason="take_profit")

    def close_position(self, date, price, reason):
        pos = self.current_position
        if not pos:
            return
        # Slippage saat exit
        if pos['action'] == 'buy':
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)
        gross_proceeds = pos['shares'] * exit_price
        commission_cost = gross_proceeds * self.commission
        net_proceeds = gross_proceeds - commission_cost
        # Hitung profit
        if pos['action'] == 'buy':
            profit = net_proceeds - (pos['shares'] * pos['entry_price'])
        else:
            profit = (pos['shares'] * pos['entry_price']) - net_proceeds
        self.capital += net_proceeds
        trade_record = {
            'symbol': self.symbol,
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'action': pos['action'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'shares': pos['shares'],
            'profit': profit,
            'profit_pct': profit / (pos['shares'] * pos['entry_price']),
            'exit_reason': reason,
            'action_type': pos['action_type']
        }
        self.trades.append(trade_record)
        self.current_position = None

    def calculate_metrics(self):
        df_equity = pd.DataFrame(self.daily_equity, columns=['date', 'equity'])
        df_equity.set_index('date', inplace=True)
        returns = df_equity['equity'].pct_change().dropna()
        total_return = (df_equity['equity'].iloc[-1] / self.initial_capital - 1)
        
        # Annualized return (assuming 252 trading days)
        days = (df_equity.index[-1] - df_equity.index[0]).days
        if days > 0:
            cagr = (df_equity['equity'].iloc[-1] / self.initial_capital) ** (252 / days) - 1
        else:
            cagr = 0
            
        # Sharpe ratio (risk-free rate = 0)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        # --- DRAWNDOWN DETAIL ---
        cumulative = df_equity['equity']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional drawdown metrics
        drawdown_duration = self.calculate_drawdown_duration(drawdown, df_equity.index)
        avg_drawdown = drawdown[drawdown < 0].mean() if any(drawdown < 0) else 0
        avg_drawdown_duration = drawdown_duration['avg_duration']
        max_drawdown_duration = drawdown_duration['max_duration']
        
        # --- WIN RATE DETAIL ---
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            winners = df_trades[df_trades['profit'] > 0]
            losers = df_trades[df_trades['profit'] < 0]
            break_even = df_trades[df_trades['profit'] == 0]
            
            win_rate = len(winners) / len(df_trades) if len(df_trades) > 0 else 0
            loss_rate = len(losers) / len(df_trades) if len(df_trades) > 0 else 0
            break_even_rate = len(break_even) / len(df_trades) if len(df_trades) > 0 else 0
            
            total_profit = winners['profit'].sum() if len(winners) > 0 else 0
            total_loss = abs(losers['profit'].sum()) if len(losers) > 0 else 0
            profit_factor = total_profit / total_loss if total_loss != 0 else 0
            
            avg_win = winners['profit'].mean() if len(winners) > 0 else 0
            avg_loss = losers['profit'].mean() if len(losers) > 0 else 0
            avg_win_pct = winners['profit_pct'].mean() if len(winners) > 0 else 0
            avg_loss_pct = losers['profit_pct'].mean() if len(losers) > 0 else 0
            
            # Win/Loss Ratio
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) + (loss_rate * avg_loss) if avg_loss != 0 else 0
            
            # Consecutive wins/losses
            consecutive_wins = self.calculate_consecutive_wins_losses(df_trades, 'profit')
            max_consecutive_wins = consecutive_wins['max_wins']
            max_consecutive_losses = consecutive_wins['max_losses']
            
            # Recovery factor
            recovery_factor = total_profit / abs(max_drawdown * self.initial_capital) if max_drawdown != 0 else 0
            
        else:
            win_rate = loss_rate = break_even_rate = profit_factor = 0
            avg_win = avg_loss = avg_win_pct = avg_loss_pct = 0
            win_loss_ratio = expectancy = 0
            max_consecutive_wins = max_consecutive_losses = 0
            recovery_factor = 0

        return {
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_capital': df_equity['equity'].iloc[-1],
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            
            # Drawdown Metrics
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration_days': max_drawdown_duration,
            'avg_drawdown_duration_days': avg_drawdown_duration,
            
            # Win Rate Metrics
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'break_even_rate': break_even_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_percent': avg_win_pct,
            'avg_loss_percent': avg_loss_pct,
            'win_loss_ratio': win_loss_ratio,
            'expectancy': expectancy,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'recovery_factor': recovery_factor,
            
            # Raw Data
            'trades': self.trades,
            'daily_equity': self.daily_equity
        }

    def calculate_drawdown_duration(self, drawdown_series, dates):
        """Hitung durasi drawdown dalam hari."""
        in_drawdown = False
        start_idx = None
        durations = []
        
        for i, (dd, date) in enumerate(zip(drawdown_series, dates)):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                durations.append((dates[i] - dates[start_idx]).days)
        
        if in_drawdown:
            durations.append((dates[-1] - dates[start_idx]).days)
        
        if durations:
            return {
                'avg_duration': np.mean(durations),
                'max_duration': max(durations)
            }
        return {'avg_duration': 0, 'max_duration': 0}

    def calculate_consecutive_wins_losses(self, df_trades, profit_col):
        """Hitung consecutive wins dan losses."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for profit in df_trades[profit_col]:
            if profit > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif profit < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:  # break even
                current_wins = 0
                current_losses = 0
        
        return {'max_wins': max_wins, 'max_losses': max_losses}

import pandas as pd
from app.core.backtester import Backtester

# Ambil data historis
symbol = "BBCA.JK"
start = "2024-01-01"
end = "2024-12-31"

bt = Backtester(symbol, start, end, initial_capital=100000000, risk_per_trade=0.02, commission=0.001, slippage=0.0005)
result = bt.run()

# Cetak metrik
print(f"Total Return: {result['total_return']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result['max_drawdown']:.2%}")
print(f"Win Rate: {result['win_rate']:.2%}")
print(f"Profit Factor: {result['profit_factor']:.2f}")
print(f"Number of Trades: {result['num_trades']}")