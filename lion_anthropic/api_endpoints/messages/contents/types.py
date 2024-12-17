from enum import Enum


class ContentTypes(str, Enum):
    """Content type enumeration for different types of message content."""

    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    DOCUMENT = "document"
