from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..response_body import AnthropicMessageResponseBody
from .types import StopReason


class ErrorType(str, Enum):
    """Types of errors that can occur in streaming."""

    OVERLOADED_ERROR = "overloaded_error"


class StreamError(BaseModel):
    """
    Model for streaming error events.

    Examples:
        >>> error = StreamError(
        ...     type="error",
        ...     error={"type": "overloaded_error", "message": "Overloaded"}
        ... )
    """

    type: Literal["error"]
    error: dict[str, str] = Field(
        ..., description="Error details including type and message"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "Overloaded",
                    },
                }
            ]
        }
    }


class ContentBlockStop(BaseModel):
    """Model for content_block_stop events."""

    type: Literal["content_block_stop"]
    index: int = Field(
        ..., description="Index of the content block that is complete"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"type": "content_block_stop", "index": 0}]
        }
    }


class MessageStop(BaseModel):
    """Model for the message_stop event."""

    type: Literal["message_stop"]

    model_config = {
        "json_schema_extra": {"examples": [{"type": "message_stop"}]}
    }


class PingEvent(BaseModel):
    """Model for ping events in the stream."""

    type: Literal["ping"]

    model_config = {"json_schema_extra": {"examples": [{"type": "ping"}]}}


class MessageStartEvent(BaseModel):
    """
    Model for the message_start event.

    This event contains a Message object with empty content.
    """

    type: Literal["message_start"]
    message: dict = Field(
        ...,
        description="Message object with initial metadata. Will be validated as AnthropicMessageResponseBody.",
    )

    @model_validator(mode="after")
    def validate_message(self):
        """Validate that the message field can be converted to AnthropicMessageResponseBody."""
        if not isinstance(self.message, dict):
            raise ValueError("Message must be a dictionary")
        try:
            AnthropicMessageResponseBody.model_validate(self.message)
        except ValueError as e:
            raise ValueError(f"Invalid message format: {e}")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-3-5-sonnet-20241022",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 25, "output_tokens": 1},
                    },
                }
            ]
        }
    }
