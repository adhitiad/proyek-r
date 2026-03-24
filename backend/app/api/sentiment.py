from fastapi import APIRouter, HTTPException
from app.core.sentiment_analysis import SentimentAnalyzerV5

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

@router.get("/{symbol}")
async def get_sentiment(symbol: str):
    analyzer = SentimentAnalyzerV5()
    result = analyzer.analyze_symbol(symbol)
    return result