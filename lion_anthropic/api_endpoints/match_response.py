# Copyright (c) 2023 - 2024, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

imported_models = {}


def match_response(request_model, response: dict | list):
    global imported_models

    endpoint = request_model.endpoint
    method = request_model.method

    # Messages endpoint
    if endpoint == "messages":
        if isinstance(response, dict):
            # Single message response
            if "AnthropicMessageResponseBody" not in imported_models:
                from .messages.response_body import \
                    AnthropicMessageResponseBody
                from .messages.responses.content import TextResponseContent
                from .messages.responses.usage import Usage

                imported_models["AnthropicMessageResponseBody"] = (
                    AnthropicMessageResponseBody
                )

            # Convert to OpenAI-like format for AssistantResponse
            text_content = ""
            if response.get("content") and len(response["content"]) > 0:
                text_content = " ".join(
                    block["text"]
                    for block in response["content"]
                    if block["type"] == "text"
                )

            return {
                "choices": [
                    {
                        "message": {
                            "content": text_content.strip(),
                            "role": "assistant",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": response.get("model"),
                "usage": response.get("usage")
                or {
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            }
        else:
            # Stream response list
            if "AnthropicMessageResponseBody" not in imported_models:
                from .messages.response_body import \
                    AnthropicMessageResponseBody
                from .messages.responses.content import TextResponseContent
                from .messages.responses.usage import Usage

                imported_models["AnthropicMessageResponseBody"] = (
                    AnthropicMessageResponseBody
                )

            events = []
            for item in response:
                if not isinstance(item, dict) or "type" not in item:
                    continue

                event_type = item.get("type")

                if event_type == "message_start":
                    events.append(
                        {
                            "type": "message_start",
                            "message": item.get("message", {}),
                        }
                    )

                elif event_type == "content_block_start":
                    if "content_block" in item:
                        events.append(
                            {
                                "type": "content_block_start",
                                "content": item["content_block"],
                            }
                        )

                elif event_type == "content_block_delta":
                    delta = item.get("delta", {})
                    if delta.get("type") == "text_delta" and "text" in delta:
                        events.append(
                            {
                                "type": "content_block_delta",
                                "delta": delta,
                            }
                        )

                elif event_type == "message_delta":
                    if "usage" in item:
                        events.append(
                            {
                                "type": "message_delta",
                                "usage": item["usage"],
                            }
                        )

                elif event_type == "message_stop":
                    events.append(
                        {
                            "type": "message_stop",
                            "usage": item.get("usage"),
                        }
                    )

            return events

    raise ValueError(
        "There is no standard response model for the provided request and response"
    )
