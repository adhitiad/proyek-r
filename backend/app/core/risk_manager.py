class RiskManager:
    def __init__(self, capital: float, risk_per_trade: float = 0.02):
        self.capital = capital
        self.risk_per_trade = risk_per_trade

    def position_size(self, entry: float, stop_loss: float):
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(entry - stop_loss)
        if risk_per_share == 0:
            return 0
        shares = risk_amount / risk_per_share
        return int(shares)

    def risk_reward_ratio(self, entry, stop_loss, take_profit):
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        return round(reward / risk, 2)
    
    def trailing_stop(self, entry_price, current_price, stop_loss, trail_percent=0.02):
        """Naikkan stop loss jika profit sudah mencapai trail_percent dari entry."""
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct >= trail_percent:
            new_stop = entry_price + (entry_price * (profit_pct - trail_percent))
            return max(stop_loss, new_stop)  # untuk buy
        return stop_loss

    def dynamic_position_size(self, capital, risk_per_trade, atr, entry_price):
        """Gunakan ATR untuk menentukan posisi size."""
        risk_amount = capital * risk_per_trade
        # Stop loss ditempatkan di 1.5 * ATR dari entry
        stop_distance = 1.5 * atr
        position_size = risk_amount / stop_distance
        return int(position_size)