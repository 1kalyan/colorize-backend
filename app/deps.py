from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt

from .config import settings
from .db import get_db


async def get_database():
    return get_db()

def get_current_user_claims(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        if payload.get("exp") and payload["exp"] < int(datetime.now(timezone.utc).timestamp()):
            raise JWTError("Token expired")
        return payload
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

def bearer_token(authorization: str | None = None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    return authorization.split(" ", 1)[1]
