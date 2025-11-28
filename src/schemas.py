from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    role: str
    content: str


class GenerationRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class GenerationResponse(BaseModel):
    content: str
