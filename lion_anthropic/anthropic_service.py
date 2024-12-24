# Copyright (c) 2023 - 2024, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Dict, List, Optional

from anthropic.types import MessageParam
from dotenv import load_dotenv
from lion_service import Service, register_service

from .anthropic_model import AnthropicModel
from .model_capability import FeatureSupport, ModelCapabilities
from .model_version import ModelVersion

load_dotenv()


@register_service
class AnthropicService(Service):
    """Anthropic service implementation using official SDK."""

    def __init__(
        self,
        api_key: str,
        api_version: str = "2024-01-01",
        name: str = None,
    ):
        """Initialize Anthropic service.

        Args:
            api_key: Anthropic API key
            api_version: API version to use
            name: Optional service name
        """
        super().__setattr__("_initialized", False)
        self.api_key = api_key
        self.api_version = api_version
        self.name = name
        self.rate_limiters = {}  # model: RateLimiter
        super().__setattr__("_initialized", True)

    def __setattr__(self, key, value):
        """Prevent modification of key attributes after initialization."""
        if getattr(self, "_initialized", False) and key in [
            "api_key",
            "api_version",
        ]:
            raise AttributeError(
                f"Cannot modify '{key}' after initialization. "
                f"Please create a new service instance instead."
            )
        super().__setattr__(key, value)

    def check_rate_limiter(
        self,
        anthropic_model: AnthropicModel,
        limit_requests: Optional[int] = None,
        limit_tokens: Optional[int] = None,
    ) -> AnthropicModel:
        """Map model versions and manage shared rate limiters.

        Args:
            anthropic_model: Model instance
            limit_requests: Optional request rate limit
            limit_tokens: Optional token rate limit

        Returns:
            Updated model instance with rate limiter assigned
        """
        # Map models to base versions for shared rate limiting
        shared_models = {
            "claude-3-5-sonnet-20241022": "claude-3-5-sonnet",
            "claude-3-5-haiku-20241022": "claude-3-5-haiku",
            "claude-3-opus-20240229": "claude-3-opus",
            "claude-3-sonnet-20240229": "claude-3-sonnet",
            "claude-3-haiku-20240307": "claude-3-haiku",
        }

        base_model = shared_models.get(
            anthropic_model.model, anthropic_model.model
        )

        # Create or reuse rate limiter
        if base_model not in self.rate_limiters:
            self.rate_limiters[base_model] = anthropic_model.rate_limiter
        else:
            anthropic_model.rate_limiter = self.rate_limiters[base_model]
            if limit_requests:
                anthropic_model.rate_limiter.limit_requests = limit_requests
            if limit_tokens:
                anthropic_model.rate_limiter.limit_tokens = limit_tokens

        return anthropic_model

    @staticmethod
    def match_data_model(task_name: str) -> Dict[str, Any]:
        """Match task to appropriate request and response models."""
        if task_name == "create_message":
            return {"messages": List[MessageParam]}
        raise ValueError(f"No data models found for task: {task_name}")

    @classmethod
    def list_tasks(cls) -> List[str]:
        """List available service tasks."""
        methods = []
        for name, member in inspect.getmembers(
            cls, predicate=inspect.isfunction
        ):
            if name not in [
                "__init__",
                "__setattr__",
                "check_rate_limiter",
                "match_data_model",
                "list_tasks",
            ]:
                methods.append(name)
        return methods

    def create_message(
        self,
        model: str,
        limit_tokens: Optional[int] = None,
        limit_requests: Optional[int] = None,
    ) -> AnthropicModel:
        """Create a message model instance.

        Args:
            model: Model identifier
            limit_tokens: Optional token rate limit
            limit_requests: Optional request rate limit

        Returns:
            Configured model instance
        """
        # Resolve model name and check compatibility
        model = ModelVersion.resolve_model_name(model)

        # Create model instance
        model_obj = AnthropicModel(
            model=model,
            api_key=self.api_key,
            limit_tokens=limit_tokens,
            limit_requests=limit_requests,
        )

        # Set up rate limiting
        return self.check_rate_limiter(
            model_obj, limit_requests=limit_requests, limit_tokens=limit_tokens
        )

    async def invoke_batch(
        self, model: str, messages_list: List[List[MessageParam]], **kwargs
    ):
        """Create and process a batch of messages.

        Args:
            model: Model identifier
            messages_list: List of message lists for batch processing
            **kwargs: Additional parameters for each request

        Returns:
            Batch processing response
        """
        # Resolve model name
        model = ModelVersion.resolve_model_name(model)

        # Check if model supports batches
        config = ModelVersion.get_model_config(model)
        if not config.get("supports_message_batches"):
            raise ValueError(f"Model {model} does not support message batches")

        # Create model instance
        model_obj = self.create_message(model)

        # Create batch requests
        requests = []
        for i, messages in enumerate(messages_list):
            requests.append(
                {
                    "custom_id": f"batch_{i}",
                    "params": {"messages": messages, **kwargs},
                }
            )

        # Process batch
        response = await model_obj.client.messages.batches.create(
            requests=requests
        )
        return response

    @property
    def allowed_roles(self) -> List[str]:
        """Get allowed message roles."""
        return ["user", "assistant"]

    @property
    def sequential_exchange(self) -> bool:
        """Check if service requires alternating messages."""
        return True  # Anthropic requires alternating user/assistant turns

    async def process_with_tools(
        self,
        model: str,
        messages: List[MessageParam],
        tools: List[Dict[str, Any]],
        disable_parallel: bool = False,
        **kwargs,
    ):
        """Process a request with tool support.

        Args:
            model: Model identifier
            messages: Conversation messages
            tools: List of tool definitions
            disable_parallel: Whether to disable parallel tool use
            **kwargs: Additional parameters
        """
        # Verify tool support
        if not ModelCapabilities.supports_feature(model, "tools"):
            raise ValueError(f"Model {model} does not support tools")

        model_obj = self.create_message(model)

        tool_choice = {"type": "function"}
        if disable_parallel and ModelCapabilities.supports_feature(
            model, "parallel_tools"
        ):
            tool_choice["disable_parallel_tool_use"] = True

        return await model_obj.invoke(
            messages=messages, tools=tools, tool_choice=tool_choice, **kwargs
        )

    async def process_with_computer(
        self, model: str, messages: List[MessageParam], **kwargs
    ):
        """Process a request with computer usage.

        Args:
            model: Model identifier
            messages: Conversation messages
            **kwargs: Additional parameters
        """
        if not ModelCapabilities.supports_feature(
            model, "computer", min_support=FeatureSupport.BETA
        ):
            raise ValueError(f"Model {model} does not support computer usage")

        model_obj = self.create_message(model)
        return await model_obj.invoke(
            messages=messages,
            extra_headers={"anthropic-beta": "computer"},
            **kwargs,
        )

    async def process_with_cache(
        self,
        model: str,
        messages: List[MessageParam],
        cache_control: Dict[str, str],
        **kwargs,
    ):
        """Process a request with caching enabled.

        Args:
            model: Model identifier
            messages: Conversation messages
            cache_control: Cache control settings
            **kwargs: Additional parameters
        """
        if not ModelCapabilities.supports_feature(model, "cache"):
            raise ValueError(f"Model {model} does not support caching")

        model_obj = self.create_message(model)
        return await model_obj.invoke(
            messages=messages, cache_control=cache_control, **kwargs
        )

    async def process_in_json_mode(
        self, model: str, messages: List[MessageParam], **kwargs
    ):
        """Process a request in JSON mode.

        Args:
            model: Model identifier
            messages: Conversation messages
            **kwargs: Additional parameters
        """
        if not ModelCapabilities.supports_feature(model, "json_mode"):
            raise ValueError(f"Model {model} does not support JSON mode")

        model_obj = self.create_message(model)
        return await model_obj.invoke(
            messages=messages,
            extra_headers={"anthropic-beta": "json-mode"},
            **kwargs,
        )

    async def get_embeddings(self, model: str, texts: List[str], **kwargs):
        """Get embeddings for texts.

        Args:
            model: Model identifier
            texts: Texts to get embeddings for
            **kwargs: Additional parameters
        """
        if not ModelCapabilities.supports_feature(model, "embeddings"):
            raise ValueError(f"Model {model} does not support embeddings")

        model_obj = self.create_message(model)
        return await model_obj.client.embeddings.create(
            model=model, texts=texts, **kwargs
        )

    async def process_with_files(
        self,
        model: str,
        messages: List[MessageParam],
        files: List[Dict[str, Any]],
        **kwargs,
    ):
        """Process a request with file uploads.

        Args:
            model: Model identifier
            messages: Conversation messages
            files: List of file definitions
            **kwargs: Additional parameters
        """
        if not ModelCapabilities.supports_feature(model, "file_upload"):
            raise ValueError(f"Model {model} does not support file uploads")

        model_obj = self.create_message(model)

        # Process files
        processed_files = []
        for file in files:
            # Add file processing logic here
            processed_files.append(file)

        return await model_obj.invoke(
            messages=messages, files=processed_files, **kwargs
        )

    def get_model_capabilities(self, model: str) -> Dict[str, Any]:
        """Get complete capability information for a model.

        Args:
            model: Model identifier

        Returns:
            Dictionary of model capabilities
        """
        model = ModelVersion.resolve_model_name(model)
        return {
            "features": ModelCapabilities.get_supported_features(model),
            "config": ModelVersion.get_model_config(model),
        }

    @staticmethod
    def list_models_with_feature(
        feature: str, min_support: Optional[str] = None
    ) -> List[str]:
        """List all models supporting a specific feature.

        Args:
            feature: Feature to check
            min_support: Minimum required support level

        Returns:
            List of model identifiers
        """
        return ModelCapabilities.list_models_supporting(
            feature, FeatureSupport(min_support) if min_support else None
        )
