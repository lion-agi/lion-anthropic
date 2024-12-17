from pathlib import Path

import aiohttp
import yaml
from dotenv import load_dotenv
from lion_service.rate_limiter import RateLimiter, RateLimitError
from lion_service.service_util import invoke_retry
from lion_service.token_calculator import TiktokenCalculator, TokenCalculator
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
from .api_endpoints.messages.message import Message
from .api_endpoints.messages.request_body import AnthropicMessageRequestBody

load_dotenv()
path = Path(__file__).parent

price_config_file_name = path / "anthropic_price_data.yaml"
max_output_token_file_name = path / "anthropic_max_output_token_data.yaml"


class AnthropicModel(BaseModel):
    """
    Model class for Anthropic API interactions.
    Handles request management, rate limiting, and token calculation.
    """

    model: str = Field(
        description="ID of the model to use (e.g., claude-3-opus-20240229)"
    )
    request_model: AnthropicRequest = Field(description="Making requests")
    rate_limiter: RateLimiter = Field(
        description="Rate Limiter to track usage"
    )
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
            "api_key": data.pop("api_key", None),
            "endpoint": data.pop("endpoint", None),
            "method": data.pop("method", None),
            "content_type": data.pop("content_type", None),
            "api_version": data.pop("api_version", None),
        }
        data["request_model"] = AnthropicRequest(**request_model_params)

        # parse rate limiter
        if "rate_limiter" not in data:
            rate_limiter_params = {}
            if limit_tokens := data.pop("limit_tokens", None):
                rate_limiter_params["limit_tokens"] = limit_tokens
            if limit_requests := data.pop("limit_requests", None):
                rate_limiter_params["limit_requests"] = limit_requests

            data["rate_limiter"] = RateLimiter(**rate_limiter_params)

        # parse token calculator
        try:
            # Anthropic uses cl100k_base encoding
            text_calc = TiktokenCalculator(encoding_name="cl100k_base")
            data["text_token_calculator"] = text_calc
        except Exception:
            pass

        return data

    @field_serializer("request_model")
    def serialize_request_model(self, value: AnthropicRequest):
        return value.model_dump(exclude_unset=True)

    async def stream(
        self,
        request_body: AnthropicMessageRequestBody,
        output_file=None,
        parse_response=True,
    ):
        """Handle streaming response from the model."""
        if not getattr(request_body, "stream", False):
            request_body.stream = True

        # Check remaining rate limit
        input_token_len = await self.get_input_token_len(request_body)
        invoke_viability_result = self.verify_invoke_viability(
            input_tokens_len=input_token_len,
            estimated_output_len=request_body.max_tokens,
        )
        if not invoke_viability_result:
            raise RateLimitError("Rate limit reached for requests")

        try:
            # Get response stream
            response_chunks = []
            response_headers = None

            try:
                async for chunk in self.request_model.stream(
                    json_data=request_body,
                    output_file=output_file,
                    with_response_header=True,
                    verbose=True,
                ):
                    if isinstance(chunk, dict):
                        if chunk.get("type") in [
                            "content_block_delta",
                            "message_delta",
                            "message_stop",
                        ]:
                            response_chunks.append(chunk)
                    elif isinstance(
                        chunk, (dict, aiohttp.multidict.CIMultiDictProxy)
                    ):
                        response_headers = chunk

                # Update rate limit if we have usage information
                if response_chunks and response_chunks[-1].get("usage"):
                    total_token_usage = (
                        response_chunks[-1]["usage"]["input_tokens"]
                        + response_chunks[-1]["usage"]["output_tokens"]
                    )
                    if response_headers:
                        self.rate_limiter.update_rate_limit(
                            response_headers.get("Date"), total_token_usage
                        )

                if parse_response:
                    return match_response(self.request_model, response_chunks)
                else:
                    return response_chunks

            except AttributeError:
                # If streaming fails, fall back to regular invoke
                response_body = await self.request_model.invoke(
                    json_data=request_body,
                    output_file=output_file,
                    parse_response=False,
                )
                if parse_response:
                    return match_response(self.request_model, response_body)
                return response_body

        except Exception as e:
            raise e

    @invoke_retry(max_retries=3, base_delay=1, max_delay=60)
    async def invoke(self, **kwargs):
        """
        Invoke the model with the given parameters.
        Handles both streaming and non-streaming responses.
        """
        # Extract request data
        if "request_body" in kwargs:
            request_data = kwargs["request_body"]
            if isinstance(request_data, dict):
                request_data["model"] = self.model
            else:
                request_data.model = self.model
        else:
            # Extract messages from kwargs
            messages = kwargs.pop("messages", None)
            if not messages:
                raise ValueError("At least one message is required")

            # Convert to list if single message
            if not isinstance(messages, list):
                messages = [messages]

            # Convert messages to Message objects if needed
            processed_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    processed_messages.append(Message(**msg))
                elif isinstance(msg, Message):
                    processed_messages.append(msg)
                else:
                    raise ValueError(f"Invalid message format: {msg}")

            # Verify we have at least one message
            if not processed_messages:
                raise ValueError("At least one message is required")

            # Get max_tokens, ensuring it's at least 1
            max_tokens = kwargs.pop("max_tokens", None)
            if max_tokens is None:
                max_tokens = self.estimated_output_len
            if not max_tokens or max_tokens < 1:
                max_tokens = 1

            # Build request data
            request_data = {
                "model": self.model,
                "messages": processed_messages,
                "max_tokens": max_tokens,
                "stream": kwargs.pop("stream", False),
                "temperature": kwargs.pop("temperature", None),
                "top_p": kwargs.pop("top_p", None),
                "top_k": kwargs.pop("top_k", None),
                "stop_sequences": kwargs.pop("stop_sequences", None),
                "system": kwargs.pop("system", None),
                "metadata": kwargs.pop("metadata", None),
                "tools": kwargs.pop("tools", None),
                "tool_choice": kwargs.pop("tool_choice", None),
            }

        # Create request body
        if isinstance(request_data, dict):
            request_body = AnthropicMessageRequestBody(**request_data)
        else:
            request_body = request_data

        # Check remaining rate limit
        input_token_len = await self.get_input_token_len(request_body)
        invoke_viability_result = self.verify_invoke_viability(
            input_tokens_len=input_token_len,
            estimated_output_len=request_body.max_tokens,
        )
        if not invoke_viability_result:
            raise RateLimitError("Rate limit reached for requests")

        try:
            # Handle streaming requests
            if getattr(request_body, "stream", False):
                return await self.stream(
                    request_body,
                    output_file=kwargs.get("output_file"),
                    parse_response=kwargs.get("parse_response", True),
                )

            # Handle non-streaming requests
            try:
                response_body, response_headers = (
                    await self.request_model.invoke(
                        json_data=request_body,
                        output_file=kwargs.get("output_file"),
                        with_response_header=True,
                        parse_response=False,
                    )
                )
            except AttributeError as e:
                # If headers access fails, try without headers
                response_body = await self.request_model.invoke(
                    json_data=request_body,
                    output_file=kwargs.get("output_file"),
                    with_response_header=False,
                    parse_response=False,
                )
                response_headers = None

            # Update rate limit if we have usage information
            if isinstance(response_body, dict) and response_body.get("usage"):
                total_token_usage = (
                    response_body["usage"]["input_tokens"]
                    + response_body["usage"]["output_tokens"]
                )
                self.rate_limiter.update_rate_limit(
                    response_headers.get("Date"), total_token_usage
                )

            if kwargs.get("parse_response", True):
                return match_response(self.request_model, response_body)
            return response_body

        except Exception as e:
            raise e

    async def get_input_token_len(
        self, request_body: AnthropicMessageRequestBody
    ):
        if request_model := getattr(request_body, "model"):
            if request_model != self.model:
                raise ValueError(
                    f"Request model does not match. Model is {self.model}, but request is made for {request_model}."
                )

        total_tokens = 0
        for message in request_body.messages:
            total_tokens += self.text_token_calculator.calculate(
                message.content
            )

        if request_body.system:
            total_tokens += self.text_token_calculator.calculate(
                request_body.system
            )

        return total_tokens

    def verify_invoke_viability(
        self, input_tokens_len: int = 0, estimated_output_len: int = 0
    ):
        self.rate_limiter.release_tokens()

        estimated_output_len = (
            estimated_output_len
            if estimated_output_len != 0
            else self.estimated_output_len
        )
        if estimated_output_len == 0:
            with open(max_output_token_file_name) as file:
                output_token_config = yaml.safe_load(file)
                estimated_output_len = output_token_config.get(self.model, 0)
                self.estimated_output_len = estimated_output_len

        if self.rate_limiter.check_availability(
            input_tokens_len, estimated_output_len
        ):
            return True
        else:
            return False

    def estimate_text_price(
        self,
        input_text: str,
        estimated_num_of_output_tokens: int = 0,
    ):
        if self.text_token_calculator is None:
            raise ValueError("Token calculator not available")

        num_of_input_tokens = self.text_token_calculator.calculate(input_text)

        with open(price_config_file_name) as file:
            price_config = yaml.safe_load(file)

        model_price_info_dict = price_config["model"][self.model]
        estimated_price = (
            model_price_info_dict["input_tokens"] * num_of_input_tokens
            + model_price_info_dict["output_tokens"]
            * estimated_num_of_output_tokens
        )

        return estimated_price
