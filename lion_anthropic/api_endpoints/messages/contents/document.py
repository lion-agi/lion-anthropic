from typing import Literal

from pydantic import BaseModel, Field

from .base_content import ContentBase
from .types import ContentTypes


class DocumentSource(BaseModel):
    """Model for document source information."""

    type: Literal["base64"] = Field(..., description="Type of document source")
    media_type: Literal["application/pdf"] = Field(
        ..., description="Media type of the document"
    )
    data: str = Field(..., description="Base64-encoded document data")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "base64_encoded_pdf_data...",
                }
            ]
        }
    }


class DocumentContent(ContentBase):
    """Model for document content."""

    type: Literal[ContentTypes.DOCUMENT]
    source: DocumentSource = Field(
        ..., description="Document source information"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "base64_encoded_pdf_data...",
                    },
                }
            ]
        }
    }
