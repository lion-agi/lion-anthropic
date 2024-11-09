from pydantic import BaseModel, Field


class Usage(BaseModel):
    """
    Model for token usage information in the response.

    Anthropic's API bills and rate-limits by token counts. Token counts in usage will not
    match one-to-one with the exact visible content of an API request or response due to
    internal transformations and parsing.

    Examples:
        >>> usage = Usage(
        ...     input_tokens=50,
        ...     output_tokens=100,
        ...     cache_creation_input_tokens=None,
        ...     cache_read_input_tokens=None
        ... )
    """

    input_tokens: int = Field(
        ..., description="The number of input tokens which were used"
    )
    output_tokens: int = Field(
        ..., description="The number of output tokens which were used"
    )
    cache_creation_input_tokens: int | None = Field(
        None, description="The number of input tokens used to create the cache entry"
    )
    cache_read_input_tokens: int | None = Field(
        None, description="The number of input tokens read from the cache"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input_tokens": 50,
                    "output_tokens": 100,
                    "cache_creation_input_tokens": None,
                    "cache_read_input_tokens": None,
                }
            ]
        }
    }