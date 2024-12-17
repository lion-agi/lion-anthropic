from enum import Enum

from pydantic import BaseModel, Field


class CacheControlType(str, Enum):
    """Cache control type enumeration."""

    EPHEMERAL = "ephemeral"


class CacheControl(BaseModel):
    """
    Model for cache control settings.
    """

    type: CacheControlType = Field(..., description="Type of cache control")

    model_config = {"json_schema_extra": {"examples": [{"type": "ephemeral"}]}}
