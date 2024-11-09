from typing import Literal

from pydantic import Field

from .base import BaseTool
from .tool_input_schema import ToolTypes


class BashTool(BaseTool):
    """
    Model for bash command execution tool.
    """

    type: Literal[ToolTypes.BASH_TOOL] = Field(
        ToolTypes.BASH_TOOL, description="Bash tool type identifier"
    )
    name: Literal["bash"] = Field("bash", description="Fixed name for bash tool")

    model_config = {
        "json_schema_extra": {"examples": [{"type": "bash_20241022", "name": "bash"}]}
    }
