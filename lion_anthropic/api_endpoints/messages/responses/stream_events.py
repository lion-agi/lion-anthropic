from pydantic import BaseModel

from .delta import ContentBlockDelta, ContentBlockStart
from .events import (ContentBlockStop, MessageStartEvent, MessageStop,
                     PingEvent, StopReason, StreamError)


class StreamEvent(BaseModel):
    """
    Union of all possible streaming event types.

    This model discriminates based on the 'type' field to determine which
    specific event model to use.
    """

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
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            ]
        }
    }

    @classmethod
    def model_validate(cls, obj):
        """Validate and convert raw data into appropriate event type."""
        if not isinstance(obj, dict):
            raise ValueError("Input must be a dictionary")

        event_type = obj.get("type")
        if not event_type:
            raise ValueError("Event must have a 'type' field")

        # Map event types to their models
        EVENT_MODELS = {
            "message_start": MessageStartEvent,
            "content_block_start": ContentBlockStart,
            "content_block_delta": ContentBlockDelta,
            "content_block_stop": ContentBlockStop,
            "message_stop": MessageStop,
            "ping": PingEvent,
            "error": StreamError,
            "stop_reason": StopReason,
        }

        model: type[BaseModel] = EVENT_MODELS.get(event_type)
        if not model:
            raise ValueError(f"Unknown event type: {event_type}")

        return model.model_validate(obj)
