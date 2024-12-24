# Copyright (c) 2023 - 2024, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import yaml
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, MessageStreamEvent
from dotenv import load_dotenv
from lion_service.rate_limiter import RateLimiter, RateLimitError
from lion_service.service_util import invoke_retry
from lion_service.token_calculator import TiktokenCalculator
from pydantic import BaseModel, ConfigDict, Field

from .model_version import ModelVersion

load_dotenv()
path = Path(__file__).parent


class AnthropicModelConfig(BaseModel):
    """Configuration for AnthropicModel."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = Field(description="ID of the model to use")
    client: Optional[AsyncAnthropic] = None
    rate_limiter: Optional[RateLimiter] = None
    text_token_calculator: Optional[TiktokenCalculator] = None
    estimated_output_len: int = Field(default=0)
    api_key: Optional[str] = None


class AnthropicModel:
    """Anthropic model configuration and execution handler using official SDK."""

    def __init__(self, **kwargs):
        # Initialize configuration
        config = AnthropicModelConfig(**kwargs)

        # Set up API client
        if config.api_key:
            self.client = AsyncAnthropic(api_key=config.api_key)
        elif os.getenv("ANTHROPIC_API_KEY"):
            self.client = AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError("API key must be provided or set in environment")

        # Set up rate limiter
        if config.rate_limiter:
            self.rate_limiter = config.rate_limiter
        else:
            params = {}
            if "limit_tokens" in kwargs:
                params["limit_tokens"] = kwargs["limit_tokens"]
            if "limit_requests" in kwargs:
                params["limit_requests"] = kwargs["limit_requests"]
            self.rate_limiter = RateLimiter(**params)

        # Set up token calculator
        try:
            self.text_token_calculator = TiktokenCalculator(
                encoding_name="cl100k_base"
            )
        except Exception:
            self.text_token_calculator = None

        # Set other attributes
        self.model = config.model
        self.estimated_output_len = config.estimated_output_len

        # Load configurations
        self.path = Path(__file__).parent
        self.price_config_file = self.path / "config/anthropic_price_data.yaml"
        self.max_output_token_file = (
            self.path / "config/anthropic_max_output_token_data.yaml"
        )

    @invoke_retry(max_retries=3, base_delay=1, max_delay=60)
    async def invoke(
        self,
        messages: list[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        stop_sequences: Optional[list[str]] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Execute model request with validation and rate limiting."""
        if not self.client:
            raise ValueError("API client not initialized")

        # Calculate input tokens
        input_tokens = await self._calculate_input_tokens(messages, system)

        # Set max tokens if not provided
        if max_tokens is None:
            max_tokens = self._get_max_output_tokens()

        # Validate request
        if not self._validate_token_lengths(input_tokens, max_tokens):
            raise ValueError("Request exceeds model context window")

        # Check rate limits
        if not self.rate_limiter.check_availability(input_tokens, max_tokens):
            raise RateLimitError(
                "Rate limit exceeded", input_tokens, max_tokens
            )

        try:
            # Format request parameters
            request_params = {
                "model": self.model,
                "messages": messages,
            }

            # Add optional parameters only if they are not None
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens
            if temperature is not None:
                request_params["temperature"] = temperature
            if top_p is not None:
                request_params["top_p"] = top_p
            if top_k is not None:
                request_params["top_k"] = top_k
            if metadata is not None:
                request_params["metadata"] = metadata
            if stop_sequences is not None:
                request_params["stop_sequences"] = stop_sequences
            if system is not None:
                request_params["system"] = system

            request_params.update(kwargs)

            if stream:
                request_params["stream"] = True
                return self._stream_response(**request_params)

            response = await self.client.messages.create(**request_params)

            # Update rate limiter if usage info available
            if response.usage:
                total_tokens = (
                    response.usage.input_tokens + response.usage.output_tokens
                )
                await self._update_rate_limiter(total_tokens)

            return response

        except Exception as e:
            # Re-raise rate limit errors
            if isinstance(e, RateLimitError):
                raise
            raise self._handle_error(e)

    async def _stream_response(
        self, **kwargs
    ) -> AsyncGenerator[MessageStreamEvent, None]:
        """Handle streaming responses."""
        if not self.client:
            raise ValueError("API client not initialized")

        try:
            stream = await self.client.messages.create(**kwargs)
            async for event in stream:
                # Update rate limiter for MessageDeltaEvent with usage info
                if hasattr(event, "usage") and event.usage:
                    total_tokens = 0
                    if hasattr(event.usage, "input_tokens"):
                        total_tokens += event.usage.input_tokens
                    if hasattr(event.usage, "output_tokens"):
                        total_tokens += event.usage.output_tokens
                    if total_tokens > 0:
                        await self._update_rate_limiter(total_tokens)
                yield event

        except Exception as e:
            raise self._handle_error(e)

    async def stream_text(
        self, messages: list[Dict[str, Any]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream just the text content from the response.

        Args:
            messages: List of messages
            **kwargs: Additional parameters for create

        Yields:
            Text chunks from the response
        """
        async for event in await self.invoke(messages, stream=True, **kwargs):
            if event.type == "content_block_delta" and hasattr(event, "delta"):
                if event.delta.type == "text_delta":
                    yield event.delta.text

    async def _calculate_input_tokens(
        self, messages: list[MessageParam], system: Optional[str] = None
    ) -> int:
        """Calculate total input tokens."""
        if not self.text_token_calculator:
            # If no calculator, use conservative estimate
            return sum(len(str(msg)) for msg in messages) // 3

        total_tokens = 0
        for msg in messages:
            if isinstance(msg["content"], str):
                total_tokens += self.text_token_calculator.calculate(
                    msg["content"]
                )
            else:
                for block in msg["content"]:
                    if block["type"] == "text":
                        total_tokens += self.text_token_calculator.calculate(
                            block["text"]
                        )

        if system:
            total_tokens += self.text_token_calculator.calculate(system)

        return total_tokens

    def _validate_token_lengths(
        self, input_tokens: int, max_tokens: int
    ) -> bool:
        """Validate token lengths against model limits."""
        config = ModelVersion.get_model_config(self.model)
        return input_tokens + max_tokens <= config["context_window"]

    def _get_max_output_tokens(self) -> int:
        """Get maximum output tokens for model."""
        try:
            with open(self.max_output_token_file) as f:
                config = yaml.safe_load(f)
                return config.get(self.model, 4096)  # Default to 4096
        except Exception:
            # Fall back to model config
            return ModelVersion.get_model_config(self.model)[
                "max_output_tokens"
            ]

    async def _update_rate_limiter(self, total_tokens: int) -> None:
        """Update rate limiter with token usage."""
        time_str = "Thu, 01 Jan 2024 00:00:00 GMT"  # Placeholder timestamp
        self.rate_limiter.update_rate_limit(time_str, total_tokens)

    def _handle_error(self, error: Exception) -> Exception:
        """Map errors to appropriate types."""
        error_mapping = {
            "AnthropicError": ValueError,
            "InvalidRequestError": ValueError,
            "AuthenticationError": ValueError,
            "PermissionDeniedError": ValueError,
            "InvalidAPIKeyError": ValueError,
            "RateLimitError": RuntimeError,
            "ServiceUnavailableError": RuntimeError,
        }

        error_type = error.__class__.__name__
        error_class = error_mapping.get(error_type, RuntimeError)

        return error_class(f"Anthropic API error: {str(error)}")

    def estimate_text_price(
        self,
        input_text: str,
        estimated_num_of_output_tokens: int = 0,
    ) -> float:
        """Estimate request cost based on current pricing."""
        if self.text_token_calculator is None:
            raise ValueError("Token calculator not available")

        num_of_input_tokens = self.text_token_calculator.calculate(input_text)

        try:
            with open(self.price_config_file) as f:
                price_config = yaml.safe_load(f)

            if (
                not price_config
                or not isinstance(price_config, dict)
                or "model" not in price_config
                or self.model not in price_config["model"]
            ):
                raise ValueError("Invalid pricing configuration")

            model_price_info = price_config["model"][self.model]

            # Calculate price (per 1M tokens)
            input_cost = (num_of_input_tokens / 1_000_000) * model_price_info[
                "input_tokens"
            ]
            output_cost = (
                estimated_num_of_output_tokens / 1_000_000
            ) * model_price_info["output_tokens"]

            return input_cost + output_cost

        except Exception as e:
            raise ValueError(f"Error loading pricing configuration: {str(e)}")

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return ModelVersion.get_model_config(self.model)
