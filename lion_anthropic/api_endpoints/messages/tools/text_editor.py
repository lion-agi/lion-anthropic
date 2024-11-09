from typing import Literal

from pydantic import Field

from .base import BaseTool
from .tool_input_schema import ToolTypes


class TextEditorTool(BaseTool):
    """
    Model for text editing tool.
    """

    type: Literal[ToolTypes.TEXT_EDITOR_TOOL] = Field(
        ToolTypes.TEXT_EDITOR_TOOL, description="Text editor tool type identifier"
    )
    name: Literal["str_replace_editor"] = Field(
        "str_replace_editor", description="Fixed name for text editor tool"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"type": "text_editor_20241022", "name": "str_replace_editor"}]
        }
    }
