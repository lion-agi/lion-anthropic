# Copyright (c) 2023 - 2024, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncGenerator
from enum import Enum
from os import getenv
from typing import Any

import aiohttp
from pydantic import BaseModel, Field, field_validator

from .match_response import match_response


class AnthropicEndpoint(str, Enum):
    """Available Anthropic API endpoints."""

    MESSAGES = "messages"


class AnthropicRequest(BaseModel):
    api_key: str = Field(
        description="API key for authentication", exclude=True
    )

    endpoint: str = Field(description="Endpoint for request")

    method: str = Field(description="HTTP method")

    content_type: str | None = Field(
        default=None, description="HTTP Content-Type"
    )

    api_version: str = Field(
        default="2023-06-01", description="Anthropic API version"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",  # Allow extra attributes for mocking
    }

    @field_validator("api_key")
    @classmethod
    def get_id(cls, value: str) -> str:
        try:
            if api_key := getenv(value):
                return api_key
            else:
                return value
        except Exception:
            return value

    def get_endpoint(self, path_param=None):
        if path_param is None:
            return self.endpoint
        else:
            return self.endpoint.format(**path_param.model_dump())

    @property
    def base_url(self):
        return "https://api.anthropic.com/v1/"

    def get_headers(self) -> dict[str, str]:
        """Get the required headers for Anthropic API requests."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",  # Always set for Anthropic API
            "accept": "application/json",  # Always accept JSON
        }
        if self.content_type and self.content_type != "application/json":
            headers["content-type"] = self.content_type
        return headers

    async def invoke(
        self,
        json_data=None,
        params=None,
        path_param=None,
        output_file: str = None,
        with_response_header: bool = False,
        parse_response: bool = True,
    ) -> Any:
        url = self.base_url + self.get_endpoint(path_param)
        headers = self.get_headers()
        json_data = (
            json_data.model_dump(exclude_unset=True) if json_data else None
        )
        params = params.model_dump(exclude_unset=True) if params else None

        async with aiohttp.ClientSession() as client:
            async with client.request(
                method=self.method,
                url=url,
                headers=headers,
                json=json_data,
                params=params,
            ) as response:
                if response.status != 200:
                    try:
                        error_text = await response.json()
                    except Exception:
                        error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"{error_text}",
                        headers=response.headers,
                    )

                # handle stream in messages
                if json_data and json_data.get("stream"):
                    response_body = []
                    file_handle = None
                    if output_file:
                        try:
                            file_handle = open(output_file, "w")
                        except Exception as e:
                            raise ValueError(
                                f"Invalid to output the response "
                                f"to {output_file}. Error:{e}"
                            )

                    try:
                        async for chunk in response.content:
                            chunk_str = chunk.decode("utf-8")
                            if chunk_str.startswith("data: "):
                                chunk_str = chunk_str[
                                    6:
                                ]  # Remove "data: " prefix
                            if chunk_str.strip() == "[DONE]":
                                continue
                            try:
                                chunk_data = json.loads(chunk_str)
                                if file_handle:
                                    file_handle.write(
                                        json.dumps(chunk_data) + "\n"
                                    )
                                response_body.append(chunk_data)
                            except json.JSONDecodeError:
                                continue
                    except Exception as e:
                        raise e
                    finally:
                        if file_handle:
                            file_handle.close()

                    if parse_response:
                        response_body = match_response(self, response_body)
                    if with_response_header:
                        return response_body, response.headers
                    else:
                        return response_body

                if output_file:
                    try:
                        with open(output_file, "wb") as f:
                            f.write(await response.read())
                    except Exception as e:
                        raise ValueError(
                            f"Invalid to output the response "
                            f"to {output_file}. Error:{e}"
                        )

                response_body = await response.json()

                if parse_response:
                    response_body = match_response(self, response_body)
                if with_response_header:
                    return response_body, response.headers
                else:
                    return response_body

    async def stream(
        self,
        json_data=None,
        verbose: bool = True,
        output_file: str = None,
        with_response_header: bool = False,
    ) -> AsyncGenerator[Any, None]:
        """Stream responses from the Anthropic API."""
        if not getattr(json_data, "stream", False):
            raise ValueError(
                "Request does not support stream. "
                "Only messages requests with stream=True are supported"
            )

        url = self.base_url + self.endpoint
        json_data = (
            json_data.model_dump(exclude_unset=True) if json_data else None
        )

        try:
            async with aiohttp.ClientSession() as client:
                async with client.request(
                    method=self.method,
                    url=url,
                    headers=self.get_headers(),
                    json=json_data,
                ) as response:
                    if response.status != 200:
                        try:
                            error_text = await response.json()
                        except Exception:
                            error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"{error_text}",
                            headers=response.headers,
                        )

                    file_handle = None
                    if output_file:
                        try:
                            file_handle = open(output_file, "w")
                        except Exception as e:
                            raise ValueError(
                                f"Invalid to output the response "
                                f"to {output_file}. Error:{e}"
                            )

                    try:
                        async for chunk in response.content:
                            chunk_str = chunk.decode("utf-8")
                            if chunk_str.startswith("data: "):
                                chunk_str = chunk_str[
                                    6:
                                ]  # Remove "data: " prefix
                            if chunk_str.strip() == "[DONE]":
                                continue
                            try:
                                chunk_data = json.loads(chunk_str)
                                if file_handle:
                                    file_handle.write(
                                        json.dumps(chunk_data) + "\n"
                                    )

                                # Print content when verbose is True
                                if verbose:
                                    if (
                                        chunk_data.get("type")
                                        == "content_block_delta"
                                    ):
                                        if text := chunk_data.get(
                                            "delta", {}
                                        ).get("text"):
                                            print(text, end="", flush=True)

                                yield chunk_data
                            except json.JSONDecodeError:
                                continue

                        if with_response_header:
                            yield dict(response.headers)
                    except Exception as e:
                        raise e
                    finally:
                        if file_handle:
                            file_handle.close()
        except Exception as e:
            raise e

    def __repr__(self):
        return (
            f"AnthropicRequest(endpoint={self.endpoint}, method={self.method}, "
            f"content_type={self.content_type}, api_version={self.api_version})"
        )
