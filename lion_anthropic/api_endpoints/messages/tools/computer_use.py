from typing import Literal

from pydantic import Field

from .base import BaseTool
from .tool_input_schema import ToolTypes


class ComputerUseTool(BaseTool):
    """
    Model for computer interaction tool.
    """

    type: Literal[ToolTypes.COMPUTER_USE_TOOL] = Field(
        ToolTypes.COMPUTER_USE_TOOL,
        description="Computer use tool type identifier",
    )
    name: Literal["computer"] = Field(
        "computer", description="Fixed name for computer use tool"
    )
    display_height_px: int = Field(
        ..., description="The height of the display in pixels"
    )
    display_width_px: int = Field(
        ..., description="The width of the display in pixels"
    )
    display_number: int | None = Field(
        None, description="The X11 display number (e.g. 0, 1) for the display"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "computer_20241022",
                    "name": "computer",
                    "display_height_px": 1080,
                    "display_width_px": 1920,
                    "display_number": 0,
                }
            ]
        }
    }
