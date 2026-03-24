import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

class AdvancedTechnical:
    @staticmethod
    def find_swing_points(df, window=5):
        """
        Menemukan swing high dan low dengan rolling window.
        Returns:
            df dengan kolom 'swing_high', 'swing_low' (boolean)
        """
        df = df.copy()
        high = df['High'].values
        low = df['Low'].values
        swing_high = np.zeros(len(df), dtype=bool)
        swing_low = np.zeros(len(df), dtype=bool)
        for i in range(window, len(df)-window):
            if high[i] == max(high[i-window:i+window+1]):
                swing_high[i] = True
            if low[i] == min(low[i-window:i+window+1]):
                swing_low[i] = True
        df['swing_high'] = swing_high
        df['swing_low'] = swing_low
        return df

    @staticmethod
    def detect_order_blocks(df, lookback=20, volume_mult=1.5, zone_width=0.005):
        """
        Mendeteksi supply (bearish) dan demand (bullish) zone.
        zone_width: persentase dari harga untuk menentukan batas zona.
        Returns:
            supply_zone: dict {'price_high', 'price_low', 'strength', 'index'} or None
            demand_zone: dict similarly
        """
        df = AdvancedTechnical.find_swing_points(df)
        df['volume_avg'] = df['Volume'].rolling(lookback).mean()
        supply_zones = []
        demand_zones = []
        for i in range(len(df)):
            if df['swing_high'].iloc[i]:
                if df['Volume'].iloc[i] > df['volume_avg'].iloc[i] * volume_mult:
                    zone_high = df['High'].iloc[i]
                    zone_low = zone_high * (1 - zone_width)
                    supply_zones.append({
                        'price_high': zone_high,
                        'price_low': zone_low,
                        'type': 'supply',
                        'strength': df['Volume'].iloc[i] / df['volume_avg'].iloc[i],
                        'index': i
                    })
            if df['swing_low'].iloc[i]:
                if df['Volume'].iloc[i] > df['volume_avg'].iloc[i] * volume_mult:
                    zone_low = df['Low'].iloc[i]
                    zone_high = zone_low * (1 + zone_width)
                    demand_zones.append({
                        'price_low': zone_low,
                        'price_high': zone_high,
                        'type': 'demand',
                        'strength': df['Volume'].iloc[i] / df['volume_avg'].iloc[i],
                        'index': i
                    })
        last_supply = supply_zones[-1] if supply_zones else None
        last_demand = demand_zones[-1] if demand_zones else None
        return last_supply, last_demand

    @staticmethod
    def detect_liquidity_grab(df, window=5, lookback=10, volume_mult=1.5):
        """
        Deteksi false breakout pada level swing high/low.
        Returns:
            (is_grab, level, direction)
        """
        df = AdvancedTechnical.find_swing_points(df, window=window)
        df['volume_avg'] = df['Volume'].rolling(lookback).mean()
        # Cari swing high dan low terakhir
        swing_highs = df[df['swing_high']].index.tolist()
        swing_lows = df[df['swing_low']].index.tolist()
        if not swing_highs or not swing_lows:
            return False, None, None
        last_swing_high = swing_highs[-1]
        last_swing_low = swing_lows[-1]
        last_candle = df.iloc[-1]
        # Liquidity grab pada resistance (bearish)
        if (last_candle['High'] > df.loc[last_swing_high, 'High'] and
            last_candle['Close'] < df.loc[last_swing_high, 'High']):
            if last_candle['Volume'] > df['volume_avg'].iloc[-1] * volume_mult:
                return True, df.loc[last_swing_high, 'High'], 'bearish'
        # Liquidity grab pada support (bullish)
        if (last_candle['Low'] < df.loc[last_swing_low, 'Low'] and
            last_candle['Close'] > df.loc[last_swing_low, 'Low']):
            if last_candle['Volume'] > df['volume_avg'].iloc[-1] * volume_mult:
                return True, df.loc[last_swing_low, 'Low'], 'bullish'
        return False, None, None

    @staticmethod
    def detect_fair_value_gap(df):
        """
        Mendeteksi Fair Value Gap dan mengembalikan yang aktif.
        Returns:
            list of active FVGs, last FVG dict
        """
        df = df.copy()
        df['fvg_bull'] = False
        df['fvg_bear'] = False
        df['fvg_bull_top'] = np.nan
        df['fvg_bull_bottom'] = np.nan
        df['fvg_bear_top'] = np.nan
        df['fvg_bear_bottom'] = np.nan

        for i in range(2, len(df)):
            # Bullish gap: low[i] > high[i-1]
            if df['Low'].iloc[i] > df['High'].iloc[i-1]:
                df.loc[df.index[i], 'fvg_bull'] = True
                df.loc[df.index[i], 'fvg_bull_top'] = df['Low'].iloc[i]
                df.loc[df.index[i], 'fvg_bull_bottom'] = df['High'].iloc[i-1]
            # Bearish gap: high[i] < low[i-1]
            if df['High'].iloc[i] < df['Low'].iloc[i-1]:
                df.loc[df.index[i], 'fvg_bear'] = True
                df.loc[df.index[i], 'fvg_bear_top'] = df['Low'].iloc[i-1]
                df.loc[df.index[i], 'fvg_bear_bottom'] = df['High'].iloc[i]

        active_fvg = []
        # Hanya gap yang belum ditembus (harga belum kembali ke area gap)
        for idx in df.index:
            if df.loc[idx, 'fvg_bull']:
                mask = df.index > idx
                later_lows = df.loc[mask, 'Low']
                later_highs = df.loc[mask, 'High']
                top = df.loc[idx, 'fvg_bull_top']
                bottom = df.loc[idx, 'fvg_bull_bottom']
                # Jika tidak ada candle yang menyentuh area gap, aktif
                if not any((later_lows <= top) & (later_highs >= bottom)):
                    active_fvg.append({
                        'type': 'bullish',
                        'top': top,
                        'bottom': bottom,
                        'index': idx
                    })
            if df.loc[idx, 'fvg_bear']:
                mask = df.index > idx
                later_lows = df.loc[mask, 'Low']
                later_highs = df.loc[mask, 'High']
                top = df.loc[idx, 'fvg_bear_top']
                bottom = df.loc[idx, 'fvg_bear_bottom']
                if not any((later_lows <= top) & (later_highs >= bottom)):
                    active_fvg.append({
                        'type': 'bearish',
                        'top': top,
                        'bottom': bottom,
                        'index': idx
                    })
        last_fvg = active_fvg[-1] if active_fvg else None
        return active_fvg, last_fvg