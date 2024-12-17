from typing import Literal

from pydantic import BaseModel, Field

from lion_anthropic.api_endpoints.models import CacheControl


class RequestMetadata(BaseModel):
    """
    Metadata about the request. Used for tracking and abuse prevention.

    Examples:
        >>> metadata = RequestMetadata(user_id="a1b2c3d4")
    """

    user_id: str | None = Field(
        None,
        description="An external identifier for the user who is associated with the request. "
        "This should be a uuid, hash value, or other opaque identifier. "
        "Anthropic may use this id to help detect abuse. Do not include any identifying "
        "information such as name, email address, or phone number.",
    )

    model_config = {
        "json_schema_extra": {"examples": [{"user_id": "user_12345abcde"}]}
    }


class SystemPromptContent(BaseModel):
    """
    Model for structured system prompt content.

    Examples:
        >>> system = SystemPromptContent(
        ...     type="text",
        ...     text="You are a helpful assistant",
        ...     cache_control=CacheControl(type="ephemeral")
        ... )
    """

    type: Literal["text"] = Field(
        "text", description="Type of system prompt content"
    )
    text: str = Field(..., description="The system prompt text")
    cache_control: CacheControl | None = Field(
        None, description="Cache control settings for the system prompt"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant specialized in financial analysis.",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        }
    }
