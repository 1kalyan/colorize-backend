# app/schemas/image.py
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, HttpUrl


class ImageRecordOut(BaseModel):
    id: str
    original_url: HttpUrl
    colorized_url: Optional[HttpUrl] = None
    output_url: Optional[HttpUrl] = None   # <-- required for alias
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
