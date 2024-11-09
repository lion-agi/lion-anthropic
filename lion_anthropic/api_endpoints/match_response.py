imported_models = {}


def match_response(request_model, response: dict | list):
    global imported_models

    endpoint = request_model.endpoint

    if endpoint == "messages":

        if isinstance(response, list):
            raise ValueError(
                "Streaming responses are not supported in this version of the library"
            )

        if "AnthropicMessageResponseBody" not in imported_models:
            from .messages.response_body import AnthropicMessageResponseBody

            imported_models["AnthropicMessageResponseBody"] = (
                AnthropicMessageResponseBody
            )
        return imported_models["OllamaChatCompletionResponseBody"](**response)

    elif not response:
        return

    else:
        raise ValueError(
            "There is no standard response model for the provided request and response"
        )
