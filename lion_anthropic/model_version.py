# Copyright (c) 2023 - 2024, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict


class ModelVersion:
    """Handles model versioning and aliases for Anthropic models."""

    # Model aliases mapping to their current versions
    MODEL_ALIASES = {
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        # Latest aliases
        "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
        "claude-3-opus-latest": "claude-3-opus-20240229",
        # AWS Bedrock
        "anthropic.claude-3-5-sonnet-20241022-v2:0": "claude-3-5-sonnet-20241022",
        "anthropic.claude-3-5-haiku-20241022-v1:0": "claude-3-5-haiku-20241022",
        "anthropic.claude-3-opus-20240229-v1:0": "claude-3-opus-20240229",
        "anthropic.claude-3-sonnet-20240229-v1:0": "claude-3-sonnet-20240229",
        "anthropic.claude-3-haiku-20240307-v1:0": "claude-3-haiku-20240307",
        # Google Vertex AI
        "claude-3-5-sonnet-v2@20241022": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku@20241022": "claude-3-5-haiku-20241022",
        "claude-3-opus@20240229": "claude-3-opus-20240229",
        "claude-3-sonnet@20240229": "claude-3-sonnet-20240229",
        "claude-3-haiku@20240307": "claude-3-haiku@20240307",
    }

    # Model capabilities and limits
    MODEL_CONFIG = {
        # Claude 3.5 Models
        "claude-3-5-sonnet-20241022": {
            "context_window": 200000,
            "max_output_tokens": 8192,
            "supports_vision": True,
            "supports_message_batches": True,
            "family": "claude-3.5",
            "generation": "3.5",
        },
        "claude-3-5-haiku-20241022": {
            "context_window": 200000,
            "max_output_tokens": 8192,
            "supports_vision": False,
            "supports_message_batches": True,
            "family": "claude-3.5",
            "generation": "3.5",
        },
        # Claude 3 Models
        "claude-3-opus-20240229": {
            "context_window": 200000,
            "max_output_tokens": 4096,
            "supports_vision": True,
            "supports_message_batches": True,
            "family": "claude-3",
            "generation": "3",
        },
        "claude-3-sonnet-20240229": {
            "context_window": 200000,
            "max_output_tokens": 4096,
            "supports_vision": True,
            "supports_message_batches": False,
            "family": "claude-3",
            "generation": "3",
        },
        "claude-3-haiku-20240307": {
            "context_window": 200000,
            "max_output_tokens": 4096,
            "supports_vision": True,
            "supports_message_batches": True,
            "family": "claude-3",
            "generation": "3",
        },
        # Legacy Models
        "claude-2.1": {
            "context_window": 200000,
            "max_output_tokens": 4096,
            "supports_vision": False,
            "supports_message_batches": False,
            "family": "claude-2",
            "generation": "2",
        },
        "claude-2.0": {
            "context_window": 100000,
            "max_output_tokens": 4096,
            "supports_vision": False,
            "supports_message_batches": False,
            "family": "claude-2",
            "generation": "2",
        },
        "claude-instant-1.2": {
            "context_window": 100000,
            "max_output_tokens": 4096,
            "supports_vision": False,
            "supports_message_batches": False,
            "family": "claude-instant",
            "generation": "1",
        },
    }

    @classmethod
    def resolve_model_name(cls, model: str) -> str:
        """Resolve model name considering aliases."""
        if model in cls.MODEL_CONFIG:
            return model

        canonical_name = cls.MODEL_ALIASES.get(model)
        if canonical_name:
            return canonical_name

        raise ValueError(f"Unknown model name or alias: {model}")

    @classmethod
    def get_model_config(cls, model: str) -> Dict:
        """Get model configuration including capabilities."""
        canonical_name = cls.resolve_model_name(model)
        config = cls.MODEL_CONFIG.get(canonical_name)
        if not config:
            raise ValueError(
                f"No configuration found for model: {canonical_name}"
            )
        return config
