import requests
import os
import logging

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # bisa string atau list (comma-separated)

def _normalize_chat_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]
        return [value.strip()]
    return [str(value)]


def send_telegram_message(text, chat_id=None):
    if not BOT_TOKEN:
        logger.warning("Telegram token not set")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    targets = _normalize_chat_ids(chat_id or CHAT_ID)
    if not targets:
        logger.warning("Telegram chat ID not set")
        return
    for target in targets:
        payload = {"chat_id": target, "text": text}
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
