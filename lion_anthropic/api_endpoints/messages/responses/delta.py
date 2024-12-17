from typing import Literal

from pydantic import BaseModel, Field

from .content import TextResponseContent, ToolUseResponseContent


class TextDelta(BaseModel):
    """Model for text deltas in content block updates."""

    type: Literal["text_delta"]
    text: str = Field(..., description="The text content being added")


class InputJsonDelta(BaseModel):
    """Model for input JSON deltas in content block updates."""

    type: Literal["input_json_delta"]
    partial_json: str = Field(
        ..., description="Partial JSON string for tool input"
    )


Delta = TextDelta | InputJsonDelta


class ContentBlockStart(BaseModel):
    """Model for content_block_start events."""

    type: Literal["content_block_start"]
    index: int = Field(
        ..., description="Index in the final Message content array"
    )
    content_block: TextResponseContent | ToolUseResponseContent = Field(
        ..., description="The initial content block with empty content"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            ]
        }
    }


class ContentBlockDelta(BaseModel):
    """
    Model for content_block_delta events.

    These events contain updates to specific content blocks identified by their index.
    """

    type: Literal["content_block_delta"]
    index: int = Field(
        ..., description="Index of the content block being updated"
    )
    delta: Delta = Field(..., description="The delta update")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"location": "San Fra',
                    },
                },
            ]
        }
    }
