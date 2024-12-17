from enum import Enum

from pydantic import BaseModel, Field, field_serializer, field_validator

from .contents import MessageContent


class Role(str, Enum):
    """Role enumeration for message roles."""

    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """
    Model for a message in the conversation.

    A message can contain either a string or a list of content blocks.
    """

    role: Role = Field(..., description="Role of the message sender")
    content: str | list[MessageContent] = Field(
        ...,
        description="Content of the message. Can be either a string or a list of content blocks",
    )

    @field_validator("role", mode="before")
    def validate_role(cls, value):
        if not isinstance(value, Role):
            try:
                return Role(value)
            except ValueError:
                raise ValueError(f"Invalid role value: {value}")
        return value

    @field_serializer("role")
    def serialize_role(self, value: Role) -> str:
        return value.value
