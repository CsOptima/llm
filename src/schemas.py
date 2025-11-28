from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    role: str
    content: str


class GenerationRequest(BaseModel):
    message: str


class GenerationResponse(BaseModel):
    queries: list[str]
