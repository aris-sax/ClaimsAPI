from fastapi import HTTPException, Header
from typing import Optional
from app.config import settings


async def get_query_token(x_token: Optional[str] = Header(None)):
    if x_token != settings.OUR_SYSTEM_SECRET_KEY_TOKEN:
        raise HTTPException(status_code=400, detail="No valid secret key token provided")
