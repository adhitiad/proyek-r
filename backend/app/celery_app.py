from celery import Celery
from celery.schedules import crontab
import os

# Redis sebagai broker dan backend
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

celery_app = Celery(
    'trading_signal',
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Jakarta',
    enable_utc=False,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Scheduled tasks
celery_app.conf.beat_schedule = {
    'scan-signals-every-5-min': {
        'task': 'app.tasks.scan_signals',
        'schedule': 300.0,  # 5 minutes
    },
    'retrain-model-weekly': {
        'task': 'app.tasks.retrain_model',
        'schedule': crontab(day_of_week='sunday', hour=2, minute=0),
    },
    'generate-daily-report': {
        'task': 'app.tasks.daily_report',
        'schedule': crontab(hour=18, minute=0),
    },
    'cleanup-old-data': {
        'task': 'app.tasks.cleanup_data',
        'schedule': crontab(hour=3, minute=0),
    },
}