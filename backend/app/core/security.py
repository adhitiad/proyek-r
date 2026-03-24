from typing import Optional

from fastapi import Header, HTTPException
from app.core.config import settings


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """
    Optional API key guard.
    If API_KEY env var is set, require matching X-API-Key header.
    If not set, allow all requests (backwards-compatible).
    """
    if not settings.API_KEY:
        return
    if not x_api_key or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
