from app.celery_app import celery_app
from app.core.singletons import get_data_manager, get_news_scraper, get_signal_generator_v5
from app.ml.trainer import ModelTrainer
from app.ml.automl import AutoMLSelector
from app.services.telegram_bot import send_telegram_message
from app.core.database import db
from app.core.config import settings
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3)
def scan_signals(self):
    """Scan all instruments and update signals"""
    try:
        instruments = list(db.instruments.find({}, {"symbol": 1}))
        data_manager = get_data_manager()
        news_scraper = get_news_scraper()
        signal_gen = get_signal_generator_v5()
        
        # Inject dependencies
        signal_gen.sentiment.news_scraper = news_scraper
        
        results = []
        for inst in instruments:
            symbol = inst['symbol']
            try:
                df = asyncio.run(data_manager.get_price_data(symbol))
                if df.empty:
                    continue
                
                signal = asyncio.run(signal_gen.generate_signal(symbol, df))
                
                # Send notification if signal changed (compare before update)
                old_signal = db.signals.find_one({"symbol": symbol})
                if old_signal and old_signal.get('action') != signal.action:
                    send_telegram_message(f"🔄 Signal Changed: {symbol} -> {signal.action.upper()}")

                # Save to database
                db.signals.update_one(
                    {"symbol": symbol},
                    {"$set": signal.__dict__},
                    upsert=True
                )

                results.append(signal)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        return {'scanned': len(results), 'timestamp': datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Scan task failed: {e}")
        raise self.retry(exc=e, countdown=60)

@celery_app.task(bind=True, max_retries=2)
def retrain_model(self):
    """Weekly model retraining"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        symbols = ["BBCA.JK", "BBRI.JK", "ASII.JK", "TLKM.JK", "UNVR.JK"]
        
        if settings.ENABLE_AUTOML:
            selector = AutoMLSelector(symbols, start_date, end_date, target_days=5)
            metadata, accuracy, _ = selector.run()
        else:
            trainer = ModelTrainer(symbols, start_date, end_date)
            metadata, accuracy = trainer.train(epochs=100)
        
        # Compare with active model
        active_config = db.config.find_one({"key": "active_model"})
        if active_config:
            active_meta = db.model_metadata.find_one({"model_path": active_config["value"]})
            if active_meta and metadata['accuracy'] > active_meta['accuracy']:
                db.config.update_one(
                    {"key": "active_model"},
                    {"$set": {"value": metadata['model_path'], "updated_at": datetime.now()}}
                )
                send_telegram_message(f"✅ New model activated! Accuracy: {accuracy:.2%}")
        
        return {
            'accuracy': accuracy,
            'model_path': metadata['model_path'],
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Retrain failed: {e}")
        raise self.retry(exc=e, countdown=300)

@celery_app.task
def daily_report():
    """Generate and send daily performance report"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        active_signals = list(db.signals.find({"action": {"$in": ["buy", "sell"]}}))
        
        report = f"📊 **Daily Report - {today}**\n\n"
        report += f"Active Signals: {len(active_signals)}\n"
        report += f"Buy: {len([s for s in active_signals if s['action'] == 'buy'])}\n"
        report += f"Sell: {len([s for s in active_signals if s['action'] == 'sell'])}\n\n"
        
        # Top signals
        report += "**Top 5 Signals:**\n"
        for sig in sorted(active_signals, key=lambda x: x.get('confidence', 0), reverse=True)[:5]:
            report += f"{sig['symbol']}: {sig['action'].upper()} (conf: {sig.get('confidence', 0):.1%})\n"
        
        send_telegram_message(report)
        return {'sent': True, 'timestamp': datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Daily report failed: {e}")
        return {'sent': False, 'error': str(e)}

@celery_app.task
def cleanup_data():
    """Clean up old data from database"""
    try:
        # Delete old signals (> 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        db.signals.delete_many({"timestamp": {"$lt": cutoff.isoformat()}})
        
        # Archive old trades
        old_trades = list(db.trades_collection.find({"exit_date": {"$lt": cutoff}}))
        if old_trades:
            db.trades_archive.insert_many(old_trades)
            db.trades_collection.delete_many({"exit_date": {"$lt": cutoff}})
        
        return {'cleaned': len(old_trades), 'timestamp': datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {'error': str(e)}
