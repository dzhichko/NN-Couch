from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ItemModel(BaseModel):
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ItemResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime