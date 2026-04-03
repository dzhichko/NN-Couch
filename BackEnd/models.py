from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionCreate(BaseModel):
    user_id: Optional[str] = "anonymous"


class SessionResponse(BaseModel):
    id: str
    user_id: str
    created_at: datetime
    message_count: int


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
