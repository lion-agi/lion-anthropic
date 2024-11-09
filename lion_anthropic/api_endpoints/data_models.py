from pydantic import BaseModel, ConfigDict


class AnthropicEndpointRequestBody(BaseModel):
    model_config = ConfigDict(
        extra="forbid", use_enum_values=True, validate_assignment=True
    )


class AnthropicEndpointResponseBody(BaseModel):
    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)
