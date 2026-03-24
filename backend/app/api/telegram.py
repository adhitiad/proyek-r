from fastapi import APIRouter, Request
from app.services.telegram_bot import handle_telegram_update

router = APIRouter(prefix="/telegram", tags=["telegram"])

@router.post("/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    await handle_telegram_update(update)
    return {"ok": True}