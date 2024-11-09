from typing import Literal

from pydantic import BaseModel, Field

from .base_content import ContentBase
from .types import ContentTypes


class ImageSource(BaseModel):
    """Model for image source information."""

    type: Literal["base64"] = Field(..., description="Type of image source")
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = Field(
        ..., description="Media type of the image"
    )
    data: str = Field(..., description="Base64-encoded image data")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "base64_encoded_image_data...",
                }
            ]
        }
    }


class ImageContent(ContentBase):
    """Model for image content."""

    type: Literal[ContentTypes.IMAGE]
    source: ImageSource = Field(..., description="Image source information")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "base64_encoded_image_data...",
                    },
                }
            ]
        }
    }
