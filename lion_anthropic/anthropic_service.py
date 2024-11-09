import inspect

from lion_service import Service, register_service

from .anthropic_model import AnthropicModel
from .api_endpoints.match_data_model import match_data_model


@register_service
class AnthropicService(Service):
    """Service class for Anthropic API interactions."""

    def __init__(self, name: str = None):
        self.name = name
        self.rate_limiters = {}  # model: RateLimiter

    def check_rate_limiter(
        self,
        anthropic_model: AnthropicModel,
        limit_requests: int = None,
        limit_tokens: int = None,
    ):
        model = anthropic_model.model

        if model not in self.rate_limiters:
            self.rate_limiters[model] = anthropic_model.rate_limiter
        else:
            anthropic_model.rate_limiter = self.rate_limiters[model]
            if limit_requests:
                anthropic_model.rate_limiter.limit_requests = limit_requests
            if limit_tokens:
                anthropic_model.rate_limiter.limit_tokens = limit_tokens

        return anthropic_model

    @staticmethod
    def match_data_model(task_name):
        return match_data_model(task_name)

    @classmethod
    def list_tasks(cls):
        methods = []
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name not in ["__init__", "check_rate_limiter", "match_data_model"]:
                methods.append(name)
        return methods

    # Create a message
    def create_message(
        self, model: str, limit_tokens: int = None, limit_requests: int = None
    ):
        """Create a message using Messages API."""
        model_obj = AnthropicModel(
            model=model,
            endpoint="messages",
            method="POST",
            limit_tokens=limit_tokens,
            limit_requests=limit_requests,
        )

        return self.check_rate_limiter(
            model_obj, limit_requests=limit_requests, limit_tokens=limit_tokens
        )

    # Count message tokens
    def count_message_tokens(
        self, model: str, limit_tokens: int = None, limit_requests: int = None
    ):
        """Count tokens for a message."""
        model_obj = AnthropicModel(
            model=model,
            endpoint="messages/count_tokens",
            method="POST",
            limit_tokens=limit_tokens,
            limit_requests=limit_requests,
        )

        return self.check_rate_limiter(
            model_obj, limit_requests=limit_requests, limit_tokens=limit_tokens
        )
