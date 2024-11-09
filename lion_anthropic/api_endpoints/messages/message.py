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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"role": "user", "content": "Hello, Claude!"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "base64_encoded_image_data...",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                            "name": "get_stock_price",
                            "input": {"ticker": "AAPL"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                            "content": [
                                {"type": "text", "text": "Current price: $150.25"}
                            ],
                        }
                    ],
                },
            ]
        }
    }

    @field_validator("role", mode="before")
    def _validate_role(cls, value):
        if not isinstance(value, Role):
            try:
                return Role(value)
            except ValueError:
                raise ValueError(f"Invalid role value: {value}")
        return value

    @field_serializer("role")
    def _validate_role(self, value: Role) -> str:
        return value.value
