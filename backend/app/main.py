from fastapi import FastAPI


from app.api import telegram
from app.api import sentiment
from app.api import signals, backtest, evaluation
from app.scheduler import start_scheduler
from app.api import optimize
from backend.app.api import websocket
from app.api import model
from app.api import v6
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


# Metrics
signal_generated = Counter('signals_generated_total', 'Total signals generated', ['action'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
backtest_duration = Histogram('backtest_duration_seconds', 'Backtest duration')

app = FastAPI(title="Forex & IDX Signal Generator")

# Include routers
app.include_router(optimize.router)
app.include_router(signals.router)
app.include_router(backtest.router)
app.include_router(evaluation.router)
app.include_router(sentiment.router)
app.include_router(model.router)
app.include_router(v6.router)
app.include_router(telegram.router)
app.include_router(websocket.router)


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
@app.get("/")
async def root():
    return {"message": "Signal Generator API is running"}


@app.on_event("startup")
async def startup_event():
    start_scheduler()