import os


os.environ.setdefault("DISABLE_SCHEDULER", "1")
os.environ.setdefault("ENABLE_AUTOML", "false")

try:
    from app.core.config import settings
    settings.DISABLE_SCHEDULER = True
    settings.ENABLE_AUTOML = False
except Exception:
    # Avoid hard failure if config import triggers other errors during collection.
    pass
