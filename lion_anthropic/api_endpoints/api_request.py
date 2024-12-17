import json
from enum import Enum

import aiohttp
from pydantic import BaseModel, Field

from .match_response import match_response


class AnthropicEndpoint(str, Enum):
    """Available Anthropic API endpoints."""

    MESSAGES = "messages"


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
        json_data=None,
        output_file=None,
        with_response_header=False,
        parse_response=True,
    ):
        """Make a request to the Anthropic API."""
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
                    collected_events = []
                    buffer = ""
                    async for chunk in response.content:
                        chunk_str = chunk.decode("utf-8")
                        buffer += chunk_str

                        while "\n\n" in buffer:
                            message, buffer = buffer.split("\n\n", 1)
                            if message.startswith("data: "):
                                data = message[6:].strip()  # Remove "data: " prefix
                                if data == "[DONE]":
                                    continue

                                try:
                                    event_dict = json.loads(data)
                                    collected_events.append(event_dict)
                                except ValueError:
                                    continue

                    if parse_response:
                        result = match_response(self, collected_events)
                        if with_response_header:
                            return result, response.headers
                        return result

                    if with_response_header:
                        return collected_events, response.headers
                    return collected_events

                # Regular JSON response
                else:
                    response_text = await response.text()
                    try:
                        response_body = json.loads(response_text)
                    except ValueError:
                        response_body = response_text

                    if output_file:
                        try:
                            with open(output_file, "w") as f:
                                json.dump(response_body, f)
                        except Exception as e:
                            raise ValueError(
                                f"Failed to write to {output_file}. Error: {e}"
                            )

                    if parse_response:
                        result = match_response(self, response_body)
                        if with_response_header:
                            return result, response.headers
                        return result

                    if with_response_header:
                        return response_body, response.headers
                    return response_body
