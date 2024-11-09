from typing import Literal

from pydantic import BaseModel, Field

from lion_anthropic.api_endpoints.models import CacheControl

from .types import ContentTypes


class ContentBase(BaseModel):
    """Base model for all content types."""

    type: ContentTypes
    cache_control: CacheControl | None = Field(
        None, description="Cache control settings for the content"
    )


class TextContent(ContentBase):
    """Model for text content."""

    type: Literal[ContentTypes.TEXT]
    text: str = Field(..., description="The text content")

    model_config = {
        "json_schema_extra": {
            "examples": [{"type": "text", "text": "Hello, how can I help you today?"}]
        }
    }
