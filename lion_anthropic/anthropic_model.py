import traceback

from lion_service.rate_limiter import RateLimiter
from lion_service.service_util import invoke_retry
from lion_service.token_calculator import TokenCalculator
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_serializer,
    model_validator,
)

from .api_endpoints.api_request import AnthropicRequest
from .api_endpoints.match_response import match_response
from .api_endpoints.messages.request_body import AnthropicMessageRequestBody


class AnthropicModel(BaseModel):
    """
    Model class for Anthropic API interactions.
    Handles request management, rate limiting, and token calculation.
    """

    model: str = Field(
        description="ID of the model to use (e.g., claude-3-opus-20240229)"
    )
    request_model: AnthropicRequest = Field(description="Making requests")
    rate_limiter: RateLimiter = Field(description="Rate Limiter to track usage")
    text_token_calculator: SerializeAsAny[TokenCalculator] = Field(
        default=None, description="Token Calculator"
    )
    estimated_output_len: int = Field(
        default=0, description="Expected output len before making request"
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def parse_input(cls, data: dict):
        if not isinstance(data, dict):
            raise ValueError("Invalid init param")

        # parse request model
        request_model_params = {
            "endpoint": data.pop("endpoint", None),
            "method": data.pop("method", None),
        }

        data["request_model"] = AnthropicRequest(**request_model_params)

        # parse rate limiter
        rate_limiter_params = {}
        if limit_tokens := data.pop("limit_tokens", None):
            rate_limiter_params["limit_tokens"] = limit_tokens
        if limit_requests := data.pop("limit_requests", None):
            rate_limiter_params["limit_requests"] = limit_requests

        data["rate_limiter"] = RateLimiter(**rate_limiter_params)

        if not data.get("text_token_calculator"):
            try:
                from lion_service.token_calculator import TiktokenCalculator

                text_calc = TiktokenCalculator(encoding_name="claude")
                data["text_token_calculator"] = text_calc
            except:
                pass

        return data

    @field_serializer("request_model")
    def serialize_request_model(self, value: AnthropicRequest):
        return value.model_dump(exclude_unset=True)

    @invoke_retry(max_retries=3, base_delay=1, max_delay=60)
    async def invoke(
        self,
        request_body: AnthropicMessageRequestBody,
        estimated_output_len: int = 0,
        output_file: str | None = None,
        parse_response: bool = True,
    ):
        if request_model := getattr(request_body, "model"):
            if request_model != self.model:
                raise ValueError(
                    f"Request model does not match. Model is {self.model}, but request is made for {request_model}."
                )

        if not getattr(request_body, "messages", None):
            response_body, response_headers = await self.request_model.invoke(
                json_data=request_body,
                output_file=output_file,
                with_response_header=True,
                parse_response=False,
            )

            if parse_response:
                return match_response(self.request_model, response_body)
            return response_body

        # check remaining rate limit
        input_token_len = self.get_input_token_len(request_body)

        invoke_viability_result = self.verify_invoke_viability(
            input_tokens_len=input_token_len, estimated_output_len=estimated_output_len
        )
        if not invoke_viability_result:
            raise ValueError("Rate limit reached for requests")

        try:
            if getattr(request_body, "stream", None):
                return await self.stream(request_body)

            response_body, response_headers = await self.request_model.invoke(
                json_data=request_body,
                output_file=output_file,
                with_response_header=True,
                parse_response=False,
            )

            total_token_usage = response_body.get("usage", {}).get(
                "input_tokens", 0
            ) + response_body.get("usage", {}).get("output_tokens", 0)
            self.rate_limiter.update_rate_limit(
                response_headers.get("Date"), total_token_usage
            )

            if parse_response:
                return match_response(self.request_model, response_body)
            return response_body

        except Exception as e:
            return Exception(f"Error: {e}. {traceback.format_exc()}")

    async def stream(
        self,
        request_body: AnthropicMessageRequestBody,
        output_file: str | None = None,
        parse_response: bool = True,
    ):
        response_list = []
        async for chunk in self.request_model.stream(
            json_data=request_body, output_file=output_file, with_response_header=True
        ):
            response_list.append(chunk)

        response_headers = response_list.pop()
        total_token_usage = response_list[-1].get("usage", {}).get(
            "input_tokens", 0
        ) + response_list[-1].get("usage", {}).get("output_tokens", 0)
        self.rate_limiter.update_rate_limit(
            response_headers.get("Date"), total_token_usage
        )

        if parse_response:
            return match_response(self.request_model, response_list)
        return response_list

    def get_input_token_len(self, request_body: AnthropicMessageRequestBody):
        if request_model := getattr(request_body, "model"):
            if request_model != self.model:
                raise ValueError(
                    f"Request model does not match. Model is {self.model}, but request is made for {request_model}."
                )

        messages_text = request_body.model_dump_json(
            include={"messages": {"__all__": {"role", "content"}}},
            exclude_unset=True,
        )
        text_tokens = self.text_token_calculator.calculate(messages_text)
        return text_tokens

    def verify_invoke_viability(
        self, input_tokens_len: int = 0, estimated_output_len: int = 0
    ):
        self.rate_limiter.release_tokens()

        estimated_output_len = (
            estimated_output_len
            if estimated_output_len != 0
            else self.estimated_output_len
        )

        if self.rate_limiter.check_availability(input_tokens_len, estimated_output_len):
            return True
        else:
            return False
