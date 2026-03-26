from datetime import datetime, timedelta
import os

import jwt
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.security import require_api_key

router = APIRouter(prefix="/auth", tags=["auth"])

JWT_SECRET = os.getenv("JWT_SECRET", "dev-jwt-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_HOURS = 1


class JwtRequest(BaseModel):
    sub: str
    role: str = "user"


@router.post("/jwt", dependencies=[Depends(require_api_key)])
async def create_jwt(payload: JwtRequest):
    now = datetime.utcnow()
    token = jwt.encode(
        {
            "sub": payload.sub,
            "role": payload.role,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=JWT_EXPIRES_HOURS)).timestamp()),
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )
    return {"token": token}
