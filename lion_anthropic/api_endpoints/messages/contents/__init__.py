from .base_content import TextContent
from .document import DocumentContent
from .image import ImageContent
from .tool_result import ToolResultContent
from .tool_use import ToolUseContent

MessageContent = (
    TextContent | ImageContent | ToolUseContent | ToolResultContent | DocumentContent
)
