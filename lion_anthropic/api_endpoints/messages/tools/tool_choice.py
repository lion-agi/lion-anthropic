from enum import Enum

from pydantic import BaseModel, Field, model_serializer, model_validator


class ToolChoiceType(str, Enum):
    """
    Enum for different types of tool choice options.
    """

    AUTO = "auto"
    ANY = "any"
    TOOL = "tool"


class ToolChoice(BaseModel):
    """
    Model for specifying how the model should use the provided tools.
    """

    tool_choice: ToolChoiceType = Field(
        ..., description="How the model should use the provided tools"
    )
    name: str | None = Field(
        None,
        description="The name of the tool to use. Only required when tool_choice is set to 'tool'",
    )
    disable_parallel_tool_use: bool = Field(
        False,
        description="If set to true, the model will output at most one tool use under auto mode, "
        "exactly one tool when under any/tool mode",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"tool_choice": "auto", "disable_parallel_tool_use": False},
                {
                    "tool_choice": "tool",
                    "name": "get_stock_price",
                    "disable_parallel_tool_use": True,
                },
                {"tool_choice": "any", "disable_parallel_tool_use": False},
            ]
        }
    }

    @model_validator(mode="before")
    def _validate_tool_choice(cls, data):
        """Validate tool choice configuration."""
        tool_choice = data.get("tool_choice")
        if not isinstance(tool_choice, ToolChoiceType):
            try:
                tool_choice = ToolChoiceType(tool_choice)
            except ValueError:
                raise ValueError(f"Invalid tool_choice value: {tool_choice}")

        if tool_choice == ToolChoiceType.TOOL and data.get("name") is None:
            raise ValueError("Tool name is required when tool_choice is set to 'tool'")
        return data

    @model_serializer
    def _serialize_tool_choice(self):
        """Serialize the tool choice configuration."""
        data = {
            "type": self.tool_choice.value,
            "disable_parallel_tool_use": self.disable_parallel_tool_use,
        }
        if self.tool_choice == ToolChoiceType.TOOL:
            data["name"] = self.name
        return data
