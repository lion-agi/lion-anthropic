from typing import Literal

from pydantic import BaseModel, Field


class MessageResponseContent(BaseModel):
    """
    Base model for message response content blocks.
    Each content block must have a type field.
    """

    type: str = Field(..., description="The type of content block")


class TextResponseContent(MessageResponseContent):
    """
    Model for text content in responses.

    Examples:
        >>> text_content = TextResponseContent(
        ...     type="text",
        ...     text="Hi, I'm Claude."
        ... )
    """

    type: Literal["text"]
    text: str = Field(..., description="The text content")

    model_config = {
        "json_schema_extra": {
            "examples": [{"type": "text", "text": "Hi, I'm Claude."}]
        }
    }


class ToolUseResponseContent(MessageResponseContent):
    """
    Model for tool use content in responses.

    Examples:
        >>> tool_use = ToolUseResponseContent(
        ...     type="tool_use",
        ...     id="toolu_01D7FLrfh4GYq7yT1ULFeyMV",
        ...     name="get_stock_price",
        ...     input={"ticker": "AAPL"}
        ... )
    """

    type: Literal["tool_use"]
    id: str = Field(..., description="Unique identifier for this tool use")
    name: str = Field(..., description="Name of the tool being used")
    input: dict = Field(..., description="Input parameters for the tool")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "tool_use",
                    "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                    "name": "get_stock_price",
                    "input": {"ticker": "AAPL"},
                }
            ]
        }
    }


ResponseContent = TextResponseContent | ToolUseResponseContent
