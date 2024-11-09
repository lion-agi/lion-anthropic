from typing import Literal

from pydantic import Field

from .base_content import ContentBase
from .types import ContentTypes


class ToolUseContent(ContentBase):
    """Model for tool use content."""

    type: Literal[ContentTypes.TOOL_USE]
    id: str = Field(..., description="Unique identifier for the tool use")
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
