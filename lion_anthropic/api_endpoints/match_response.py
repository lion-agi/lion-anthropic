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
                from .messages.response_body import AnthropicMessageResponseBody
                from .messages.responses.content import TextResponseContent
                from .messages.responses.usage import Usage

                imported_models["AnthropicMessageResponseBody"] = (
                    AnthropicMessageResponseBody
                )

            # Convert to AnthropicMessageResponseBody
            from .messages.responses.content import TextResponseContent
            from .messages.responses.usage import Usage

            # Create response content
            content = []
            if response.get("content") and len(response["content"]) > 0:
                for block in response["content"]:
                    if block["type"] == "text":
                        content.append(
                            TextResponseContent(type="text", text=block["text"])
                        )

            # Create usage info
            usage_data = response.get("usage") or {
                "input_tokens": 0,
                "output_tokens": 0,
            }
            usage = Usage(**usage_data)

            # Create full response
            return imported_models["AnthropicMessageResponseBody"](
                id=response.get("id", ""),
                type="message",
                role="assistant",
                content=content,
                model=response.get("model"),
                stop_reason=response.get("stop_reason"),
                stop_sequence=response.get("stop_sequence"),
                usage=usage,
            )
        else:
            # Stream response list
            if "AnthropicMessageResponseBody" not in imported_models:
                from .messages.response_body import AnthropicMessageResponseBody
                from .messages.responses.content import TextResponseContent
                from .messages.responses.usage import Usage

                imported_models["AnthropicMessageResponseBody"] = (
                    AnthropicMessageResponseBody
                )

            from .messages.responses.stream_events import StreamEvent

            events = []
            for item in response:
                if not isinstance(item, dict) or "type" not in item:
                    continue
                try:
                    event = StreamEvent.model_validate(item)
                    events.append(event)
                except ValueError:
                    continue

            return events

    raise ValueError(
        "There is no standard response model for the provided request and response"
    )
