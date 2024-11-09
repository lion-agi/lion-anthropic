from typing import Literal

from pydantic import BaseModel, Field

from .base_content import ContentBase
from .image import ImageSource
from .types import ContentTypes


class ToolResultContentImage(BaseModel):
    """Model for image content within tool results."""

    type: Literal[ContentTypes.IMAGE]
    source: ImageSource


class ToolResultContentText(BaseModel):
    """Model for text content within tool results."""

    type: Literal[ContentTypes.TEXT]
    text: str


class ToolResultContent(ContentBase):
    """Model for tool result content."""

    type: Literal[ContentTypes.TOOL_RESULT]
    tool_use_id: str = Field(..., description="ID of the tool use this result is for")
    is_error: bool | None = Field(
        None, description="Whether this result represents an error"
    )
    content: str | list[ToolResultContentText | ToolResultContentImage] = Field(
        ..., description="The content of the tool result"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                    "content": [{"type": "text", "text": "Current price: $150.25"}],
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                    "is_error": True,
                    "content": [{"type": "text", "text": "Error fetching stock price"}],
                },
            ]
        }
    }
