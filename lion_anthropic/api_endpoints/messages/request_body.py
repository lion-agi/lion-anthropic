from pydantic import Field, model_validator

from ..data_models import AnthropicEndpointRequestBody
from .message import Message
from .system import SystemPromptContent
from .tools import Tool
from .tools.tool_choice import ToolChoice


class AnthropicMessageRequestBody(AnthropicEndpointRequestBody):
    """
    Model for creating a message request to the Anthropic API.

    This model represents all possible parameters that can be sent in a message request.
    Messages API operates on alternating user and assistant conversational turns.

    The model will generate the next Message in the conversation based on the provided messages.
    Consecutive user or assistant turns in your request will be combined into a single turn.
    """

    model: str = Field(..., description="The model that will complete your prompt")

    messages: list[Message] = Field(
        ...,
        description="Input messages. Each message must have a role and content. You can specify "
        "a single user-role message, or include multiple user and assistant messages. "
        "If the final message uses the assistant role, the response content will continue "
        "immediately from the content in that message.",
    )

    max_tokens: int = Field(
        ...,
        description="The maximum number of tokens to generate before stopping. Note that our models "
        "may stop before reaching this maximum. This parameter only specifies the absolute "
        "maximum number of tokens to generate. Different models have different maximum "
        "values for this parameter.",
    )

    metadata: dict | None = Field(
        None, description="An object describing metadata about the request"
    )

    stop_sequences: list[str] | None = Field(
        None,
        description="Custom text sequences that will cause the model to stop generating. "
        "If the model encounters one of the custom sequences, the response stop_reason "
        "value will be 'stop_sequence' and the response stop_sequence value will contain "
        "the matched stop sequence.",
    )

    stream: bool | None = Field(
        None,
        description="Whether to incrementally stream the response using server-sent events",
    )

    system: str | SystemPromptContent | None = Field(
        None,
        description="System prompt. A way of providing context and instructions to Claude, "
        "such as specifying a particular goal or role",
    )

    temperature: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Amount of randomness injected into the response. Defaults to 1.0. "
        "Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / "
        "multiple choice, and closer to 1.0 for creative and generative tasks. "
        "Note that even with temperature of 0.0, the results will not be fully deterministic.",
    )

    top_k: int | None = Field(
        None,
        description="Only sample from the top K options for each subsequent token. "
        "Used to remove 'long tail' low probability responses. "
        "Recommended for advanced use cases only. "
        "You usually only need to use temperature.",
    )

    top_p: float | None = Field(
        None,
        description="Use nucleus sampling. In nucleus sampling, we compute the cumulative "
        "distribution over all the options for each subsequent token in decreasing "
        "probability order and cut it off once it reaches a particular probability "
        "specified by top_p. You should either alter temperature or top_p, but not both. "
        "Recommended for advanced use cases only. "
        "You usually only need to use temperature.",
    )

    tools: list[Tool] | None = Field(
        None,
        description="Definitions of tools that the model may use. If included, the model may "
        "return tool_use content blocks that represent the model's use of those tools.",
    )

    tool_choice: ToolChoice | None = Field(
        None,
        description="How the model should use the provided tools. The model can use a specific tool, "
        "any available tool, or decide by itself.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    # Simple text message
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hello, Claude"}],
                    "max_tokens": 1024,
                },
                {
                    # Complex message with system prompt and tools
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What's the current price of AAPL?",
                                }
                            ],
                        }
                    ],
                    "max_tokens": 1024,
                    "system": "You are a helpful financial assistant.",
                    "temperature": 0.1,
                    "tools": [
                        {
                            "name": "get_stock_price",
                            "description": "Get the current stock price for a given ticker symbol.",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "ticker": {
                                        "type": "string",
                                        "description": "The stock ticker symbol",
                                    }
                                },
                                "required": ["ticker"],
                            },
                        }
                    ],
                    "tool_choice": {"type": "auto", "disable_parallel_tool_use": False},
                },
                {
                    # Multimodal message with image
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What's in this image?"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": "base64_encoded_image_data...",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
            ]
        }
    }

    @model_validator(mode="after")
    def validate_temperature_and_top_p(self):
        """Validate that temperature and top_p are not both set."""
        if self.temperature is not None and self.top_p is not None:
            raise ValueError(
                "You should either alter temperature or top_p, but not both"
            )
        return self

    @model_validator(mode="after")
    def validate_tools_and_tool_choice(self):
        """Validate that tool_choice is only set when tools are provided."""
        if self.tool_choice is not None and not self.tools:
            raise ValueError("tool_choice can only be set when tools are provided")
        return self
