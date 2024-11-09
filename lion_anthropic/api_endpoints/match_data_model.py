def match_data_model(task_name):
    if task_name in ["chat", "messages"]:
        from .messages.request_body import AnthropicMessageRequestBody

        return {"json_data": AnthropicMessageRequestBody}

    else:
        raise ValueError(f"Invalid task: {task_name}. Not supported in the service.")
