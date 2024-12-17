from pydantic import BaseModel, Field

from lion_anthropic.api_endpoints.models import CacheControl

from .tool_input_schema import ToolTypes


class BaseTool(BaseModel):
    """
    Base model for all tool types.
    """

    type: ToolTypes | None = Field(None, description="The type of the tool")
    description: str | None = Field(
        None,
        description="Description of what this tool does. Tool descriptions should be as detailed as "
        "possible. The more information that the model has about what the tool is and how "
        "to use it, the better it will perform.",
    )
    name: str = Field(
        ...,
        description="This is how the tool will be called by the model and in tool_use blocks",
    )
    cache_control: CacheControl | None = Field(
        None, description="Cache control settings for the tool"
    )
