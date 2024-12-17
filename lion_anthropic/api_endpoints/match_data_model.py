def match_data_model(task_name):
    if task_name in ["chat", "messages", "create_message"]:
        from .messages.request_body import AnthropicMessageRequestBody

        return {"request_body": AnthropicMessageRequestBody}

    else:
        raise ValueError(
            f"Invalid task: {task_name}. Not supported in the service."
        )
