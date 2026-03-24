import requests
import os
import logging
from fastapi import Request

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # bisa string atau list

def send_telegram_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram token or chat ID not set")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

async def handle_telegram_update(update: dict):
    """Proses update dari webhook Telegram."""
    if "message" in update:
        chat_id = update["message"]["chat"]["id"]
        text = update["message"].get("text", "")
        if text == "/status":
            # Kirim status sistem
            send_telegram_message("Sistem sinyal aktif", chat_id=chat_id)
        elif text.startswith("/signal"):
            parts = text.split()
            if len(parts) > 1:
                symbol = parts[1]
                # Ambil sinyal terbaru dari database
                # ...
                send_telegram_message(f"Sinyal untuk {symbol}: ...", chat_id=chat_id)