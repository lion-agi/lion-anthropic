from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ToolTypes(str, Enum):
    """
    Enum for different types of tools available in the API.
    """

    CUSTOM = "custom"
    COMPUTER_USE_TOOL = "computer_20241022"
    BASH_TOOL = "bash_20241022"
    TEXT_EDITOR_TOOL = "text_editor_20241022"


class ToolInputSchemaProperty(BaseModel):
    """
    Model for defining a property in a tool's input schema.
    """

    type: str = Field(
        ..., description="The type of the property (e.g., 'string', 'number', etc.)"
    )
    description: str | None = Field(None, description="Description of the property")
    enum: list[str] | None = Field(
        None, description="List of allowed values for the property"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"type": "string", "description": "The stock ticker symbol"},
                {
                    "type": "string",
                    "description": "The currency code",
                    "enum": ["USD", "EUR", "GBP"],
                },
            ]
        }
    }


class ToolInputSchema(BaseModel):
    """
    Model for defining the input schema for a tool.
    """

    type: Literal["object"] = Field(..., description="The type of the input schema")
    properties: dict[str, ToolInputSchemaProperty] = Field(
        ...,
        description="Dictionary of properties that define the structure of the input",
    )
    required: list[str] = Field(
        default_factory=list,
        description="List of property names that are required in the input",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol",
                        }
                    },
                    "required": ["ticker"],
                }
            ]
        }
    }
