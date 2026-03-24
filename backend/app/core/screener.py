from app.core.backtester import Backtester
from app.core.database import db
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from datetime import datetime, timedelta

class Screener:
    def __init__(self, symbols, start_date, end_date, use_cache=True, **backtest_kwargs):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.use_cache = use_cache
        self.backtest_kwargs = backtest_kwargs

    def _get_cache_key(self, symbol):
        key_data = {
            'symbol': symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'params': self.backtest_kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, symbol):
        if not self.use_cache:
            return None
        cache_key = self._get_cache_key(symbol)
        cached = db.screening_cache.find_one({
            'cache_key': cache_key,
            'expires_at': {'$gt': datetime.now()}
        })
        if cached:
            return cached['result']
        return None

    def _cache_result(self, symbol, result):
        cache_key = self._get_cache_key(symbol)
        db.screening_cache.update_one(
            {'cache_key': cache_key},
            {
                '$set': {
                    'result': result,
                    'expires_at': datetime.now() + timedelta(hours=24),
                    'updated_at': datetime.now()
                }
            },
            upsert=True
        )

    def run_single(self, symbol):
        try:
            cached = self._get_cached_result(symbol)
            if cached:
                return cached
            bt = Backtester(symbol, self.start_date, self.end_date, **self.backtest_kwargs)
            result = bt.run()
            self._cache_result(symbol, result)
            return result
        except Exception as e:
            return {'symbol': symbol, 'error': str(e)}

    def run(self, max_workers=10):
        """Screening paralel dengan caching."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.run_single, sym): sym for sym in self.symbols}
            for future in as_completed(futures):
                results.append(future.result())
        # Urutkan berdasarkan win rate atau profit factor
        results.sort(key=lambda x: x.get('win_rate', 0), reverse=True)
        return results