from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        return value.strip()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1, max_length=8)


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False
