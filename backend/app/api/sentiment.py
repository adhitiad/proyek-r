from fastapi import APIRouter, Depends
from app.core.singletons import get_sentiment_analyzer
from app.core.security import require_api_key

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

@router.get("/{symbol}", dependencies=[Depends(require_api_key)])
async def get_sentiment(symbol: str):
    analyzer = get_sentiment_analyzer()
    result = await analyzer.analyze_symbol(symbol)
    return result
