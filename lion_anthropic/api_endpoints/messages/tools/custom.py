from typing import Literal

from pydantic import Field

from .base import BaseTool
from .tool_input_schema import ToolInputSchema, ToolTypes


class CustomTool(BaseTool):
    """
    Model for custom user-defined tools.
    """

    type: Literal[ToolTypes.CUSTOM] = Field(
        ToolTypes.CUSTOM, description="Custom tool type identifier"
    )
    input_schema: ToolInputSchema = Field(
        ..., description="JSON schema for this tool's input"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "get_stock_price",
                    "description": "Get the current stock price for a given ticker symbol.",
                    "type": "custom",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol",
                            }
                        },
                        "required": ["ticker"],
                    },
                }
            ]
        }
    }
