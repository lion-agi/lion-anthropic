from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from lion_anthropic.api_endpoints.data_models import AnthropicEndpointResponseBody


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
                    "error": {"type": "overloaded_error", "message": "Overloaded"},
                }
            ]
        }
    }


class StopReason(str, Enum):
    """
    Enumeration of possible reasons why the model stopped generating.

    Attributes:
        END_TURN: the model reached a natural stopping point
        MAX_TOKENS: exceeded the requested max_tokens or the model's maximum
        STOP_SEQUENCE: one of the provided custom stop_sequences was generated
        TOOL_USE: the model invoked one or more tools
    """

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


class ContentBlockStop(BaseModel):
    """Model for content_block_stop events."""

    type: Literal["content_block_stop"]
    index: int = Field(..., description="Index of the content block that is complete")

    model_config = {
        "json_schema_extra": {"examples": [{"type": "content_block_stop", "index": 0}]}
    }


class MessageStop(BaseModel):
    """Model for the message_stop event."""

    type: Literal["message_stop"]

    model_config = {"json_schema_extra": {"examples": [{"type": "message_stop"}]}}


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
    message: AnthropicEndpointResponseBody

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
