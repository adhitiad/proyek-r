from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from app.core.signal_generator import SignalGenerator
from app.core.singletons import get_signal_generator_v5
from app.core.data_collector import DataCollector
from app.core.database import db
from app.services.telegram_bot import send_telegram_message
from app.services.websocket_manager import manager
from app.ml.trainer import ModelTrainer
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()

def scan_and_update_signals():
    """Scanning periodik dengan error handling dan caching."""
    instruments = list(db.instruments.find({}, {"symbol": 1})) or [
        {"symbol": "BBCA.JK"}, {"symbol": "BBRI.JK"}, {"symbol": "ASII.JK"},
        {"symbol": "TLKM.JK"}, {"symbol": "UNVR.JK"}, {"symbol": "GOTO.JK"},
        {"symbol": "BBNI.JK"}, {"symbol": "BMRI.JK"}, {"symbol": "ADRO.JK"},
        {"symbol": "EURUSD=X"}, {"symbol": "XAUUSD=X"}
    ]
    gen = SignalGenerator(v5=get_signal_generator_v5())
    for inst in instruments:
        symbol = inst['symbol']
        try:
            df = DataCollector.get_price_data(symbol, period="1mo", interval="1d")
            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            signal = gen.generate_signal(symbol, df)

            existing = db.signals.find_one({"symbol": symbol})
            if existing:
                # Cek apakah ada perubahan sinyal
                if existing.get("action") != signal["action"]:
                    msg = f"🔄 **Sinyal Baru**\n📊 {symbol}\n🎯 {signal['action'].upper()} @ {signal['entry_zone']:,.0f}\n📈 Prob: {signal['probability']}%\n📝 {signal['notes'][:100]}"
                    send_telegram_message(msg)
                    # Broadcast via WebSocket
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        asyncio.create_task(manager.broadcast(signal))
                    else:
                        asyncio.run(manager.broadcast(signal))
                db.signals.update_one({"symbol": symbol}, {"$set": signal})
            else:
                db.signals.insert_one(signal)
                msg = f"🔔 **Sinyal Awal**\n📊 {symbol}\n🎯 {signal['action'].upper()} @ {signal['entry_zone']:,.0f}\n📈 Prob: {signal['probability']}%"
                send_telegram_message(msg)
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    asyncio.create_task(manager.broadcast(signal))
                else:
                    asyncio.run(manager.broadcast(signal))

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")

def retrain_model():
    """Retraining mingguan dengan data terbaru."""
    logger.info("Starting weekly model retraining...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    symbols = ["BBCA.JK", "BBRI.JK", "ASII.JK", "TLKM.JK", "UNVR.JK"]

    trainer = ModelTrainer(symbols, start_date.isoformat(), end_date.isoformat(), target_days=5)
    try:
        metadata, accuracy = trainer.train(epochs=100)
        msg = f"🤖 **Model Retraining Complete**\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n🎯 Akurasi: {accuracy:.2%}\n📁 Model: {metadata['model_path']}"
        send_telegram_message(msg)

        # Bandingkan dengan model aktif
        active_config = db.config.find_one({"key": "active_model"})
        if active_config:
            active_meta = db.model_metadata.find_one({"model_path": active_config["value"]})
            if active_meta and metadata["accuracy"] > active_meta["accuracy"]:
                db.config.update_one(
                    {"key": "active_model"},
                    {"$set": {"value": metadata["model_path"], "updated_at": datetime.now()}}
                )
                send_telegram_message(f"✅ **Model Baru Diaktifkan!**\nAkurasi: {accuracy:.2%} (sebelumnya {active_meta['accuracy']:.2%})")
        else:
            db.config.insert_one({"key": "active_model", "value": metadata["model_path"], "updated_at": datetime.now()})
            send_telegram_message(f"✅ **Model Pertama Diaktifkan!**\nAkurasi: {accuracy:.2%}")

        db.jobs.insert_one({
            "type": "retrain",
            "timestamp": datetime.now(),
            "accuracy": accuracy,
            "model_path": metadata["model_path"]
        })
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        send_telegram_message(f"❌ **Retraining Error**: {str(e)}")

def start_scheduler():
    """Start all scheduled jobs."""
    # Scanning setiap 60 menit
    scheduler.add_job(
        scan_and_update_signals,
        trigger=IntervalTrigger(minutes=60),
        id='scan_signals',
        replace_existing=True
    )
    # Retraining setiap Minggu jam 02:00
    scheduler.add_job(
        retrain_model,
        trigger=CronTrigger(day_of_week='sun', hour=2, minute=0),
        id='retrain_model',
        replace_existing=True
    )
    # Daily report (opsional)
    scheduler.add_job(
        send_daily_report,
        trigger=CronTrigger(hour=18, minute=0),
        id='daily_report',
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduler started: scanning every 60min, retraining Sunday 02:00")

def send_daily_report():
    """Kirim laporan harian performa sinyal."""
    today = datetime.now().strftime('%Y-%m-%d')
    active_signals = list(db.signals.find({"action": {"$in": ["buy", "sell"]}}, {"_id": 0}))
    if active_signals:
        msg = f"📊 **Daily Signal Report - {today}**\n\n"
        for sig in active_signals[:10]:
            msg += f"{sig['symbol']}: {sig['action'].upper()} @ {sig['entry_zone']:,.0f} (prob {sig['probability']}%)\n"
        send_telegram_message(msg)
