import json
from enum import Enum
from typing import AsyncGenerator

import aiohttp
from pydantic import BaseModel, Field

from .data_models import AnthropicEndpointResponseBody
from .match_response import match_response
from .messages.request_body import AnthropicMessageRequestBody
from .messages.responses.delta import ContentBlockDelta
from .messages.responses.stream_events import StreamEvent


class AnthropicEndpoint(str, Enum):
    """Available Anthropic API endpoints."""

    MESSAGES = "messages"
    # Add other endpoints as needed


class AnthropicRequest(BaseModel):
    """
    Base class for making requests to the Anthropic API.
    """

    endpoint: AnthropicEndpoint = Field(..., description="API endpoint to call")
    method: str = Field(..., description="HTTP method")
    api_key: str = Field(..., description="Anthropic API key")
    version: str = Field("2023-06-01", description="Anthropic API version")
    beta: str | None = Field(None, description="Optional beta version(s)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "endpoint": "messages",
                    "method": "POST",
                    "api_key": "your-api-key",
                    "version": "2023-06-01",
                }
            ]
        }
    }

    @property
    def base_url(self) -> str:
        """Get the base URL for the Anthropic API."""
        return "https://api.anthropic.com/v1/"

    @property
    def headers(self) -> dict[str, str]:
        """Get the required headers for Anthropic API requests."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "content-type": "application/json",
        }
        if self.beta:
            headers["anthropic-beta"] = self.beta
        return headers

    async def invoke(
        self,
        json_data: AnthropicMessageRequestBody | None = None,
        output_file: str | None = None,
        with_response_header: bool = False,
        parse_response: bool = True,
    ) -> AnthropicEndpointResponseBody | tuple[AnthropicEndpointResponseBody, dict]:
        """
        Make a request to the Anthropic API.

        Similar pattern to Ollama implementation, handles both streaming and non-streaming
        responses based on the content type.
        """
        url = self.base_url + self.endpoint
        json_data = json_data.model_dump(exclude_none=True) if json_data else None

        async with aiohttp.ClientSession() as client:
            async with client.request(
                method=self.method, url=url, headers=self.headers, json=json_data
            ) as response:
                if response.status != 200:
                    try:
                        error_text = await response.json()
                    except:
                        error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"HTTP Error {response.status}. Response Body: {error_text}",
                        headers=response.headers,
                    )

                # Handle stream response (text/event-stream)
                if response.headers.get("Content-Type") == "text/event-stream":
                    response_body = []
                    file_handle = None
                    if output_file:
                        try:
                            file_handle = open(output_file, "w")
                        except Exception as e:
                            raise ValueError(
                                f"Failed to write to {output_file}. Error: {e}"
                            )

                    try:
                        buffer = ""
                        async for chunk in response.content:
                            chunk_str = chunk.decode("utf-8")
                            buffer += chunk_str

                            while "\n\n" in buffer:
                                message, buffer = buffer.split("\n\n", 1)
                                if message.startswith("data: "):
                                    data = message[6:]  # Remove "data: " prefix
                                    if data.strip() == "[DONE]":
                                        continue

                                    event_dict = json.loads(data)
                                    response_body.append(event_dict)

                                    if file_handle:
                                        file_handle.write(json.dumps(event_dict) + "\n")

                    finally:
                        if file_handle:
                            file_handle.close()

                    if parse_response:
                        response_body = match_response(self, response_body)

                    if with_response_header:
                        return response_body, response.headers
                    return response_body

                # Handle regular response
                else:
                    try:
                        response_body = await response.json()
                    except:
                        response_body = await response.text()

                    if output_file:
                        try:
                            with open(output_file, "w") as f:
                                json.dump(response_body, f)
                        except Exception as e:
                            raise ValueError(
                                f"Failed to write to {output_file}. Error: {e}"
                            )

                    if parse_response:
                        response_body = match_response(self, response_body)

                    if with_response_header:
                        return response_body, response.headers
                    return response_body

    async def stream(
        self,
        json_data: AnthropicMessageRequestBody,
        verbose: bool = True,
        output_file: str | None = None,
        with_response_header: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream responses from the Anthropic API.

        Args:
            json_data: The request body with stream=True
            verbose: Whether to print responses
            output_file: Optional file to save the stream
            with_response_header: Whether to yield response headers at the end

        Yields:
            StreamEvent objects of various types based on the SSE stream

        Raises:
            ValueError: If streaming is not enabled
            aiohttp.ClientResponseError: If the API request fails
        """
        if not getattr(json_data, "stream", None):
            raise ValueError(
                "Request body must have stream=True for streaming responses"
            )

        url = self.base_url + self.endpoint
        json_data = json_data.model_dump(exclude_none=True)

        async with aiohttp.ClientSession() as client:
            async with client.request(
                method=self.method, url=url, headers=self.headers, json=json_data
            ) as response:
                if response.status != 200:
                    try:
                        error_text = await response.json()
                    except:
                        error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"HTTP Error {response.status}. Response Body: {error_text}",
                        headers=response.headers,
                    )

                file_handle = None
                if output_file:
                    try:
                        file_handle = open(output_file, "w")
                    except Exception as e:
                        raise ValueError(f"Failed to open {output_file}. Error: {e}")

                try:
                    buffer = ""
                    async for chunk in response.content:
                        chunk_str = chunk.decode("utf-8")
                        buffer += chunk_str

                        while "\n\n" in buffer:
                            message, buffer = buffer.split("\n\n", 1)
                            if message.startswith("data: "):
                                data = message[6:]  # Remove "data: " prefix
                                if data.strip() == "[DONE]":
                                    continue

                                try:
                                    event_dict = json.loads(data)
                                    event = StreamEvent.model_validate(event_dict)

                                    if file_handle:
                                        file_handle.write(json.dumps(event_dict) + "\n")

                                    if verbose:
                                        if isinstance(event, ContentBlockDelta):
                                            if event.delta.type == "text_delta":
                                                print(
                                                    event.delta.text, end="", flush=True
                                                )

                                    yield event

                                except ValueError as e:
                                    print(f"Warning: Failed to parse event: {str(e)}")
                                    continue

                        if with_response_header:
                            yield response.headers

                finally:
                    if file_handle:
                        file_handle.close()
