# Copyright (c) 2023 - 2024, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Dict, Optional


class FeatureSupport(str, Enum):
    """Support level for model features."""

    FULL = "full"  # Feature fully supported
    BETA = "beta"  # Feature in beta
    PARTIAL = "partial"  # Feature partially supported
    NOT_SUPPORTED = "none"  # Feature not supported


class ModelCapabilities:
    """Model capabilities and features."""

    CAPABILITIES = {
        # Claude 3.5 Models
        "claude-3-5-sonnet-20241022": {
            "context_window": 200000,
            "max_output_tokens": 8192,
            "features": {
                "vision": FeatureSupport.FULL,
                "message_batches": FeatureSupport.FULL,
                "tools": FeatureSupport.FULL,
                "computer": FeatureSupport.BETA,
                "cache": FeatureSupport.FULL,
                "embeddings": FeatureSupport.FULL,
                "file_upload": FeatureSupport.FULL,
                "parallel_tools": FeatureSupport.FULL,
                "json_mode": FeatureSupport.FULL,
            },
        },
        "claude-3-5-haiku-20241022": {
            "context_window": 200000,
            "max_output_tokens": 8192,
            "features": {
                "vision": FeatureSupport.NOT_SUPPORTED,
                "message_batches": FeatureSupport.FULL,
                "tools": FeatureSupport.FULL,
                "computer": FeatureSupport.BETA,
                "cache": FeatureSupport.FULL,
                "embeddings": FeatureSupport.FULL,
                "file_upload": FeatureSupport.FULL,
                "parallel_tools": FeatureSupport.FULL,
                "json_mode": FeatureSupport.FULL,
            },
        },
        # More models...
    }

    @classmethod
    def supports_feature(
        cls,
        model: str,
        feature: str,
        min_support: FeatureSupport = FeatureSupport.FULL,
    ) -> bool:
        """Check if model supports a specific feature.

        Args:
            model: Model identifier
            feature: Feature to check
            min_support: Minimum required support level

        Returns:
            Whether feature is supported at required level
        """
        if model not in cls.CAPABILITIES:
            return False

        feature_support = cls.CAPABILITIES[model]["features"].get(
            feature, FeatureSupport.NOT_SUPPORTED
        )

        # Convert to enum if string passed
        if isinstance(min_support, str):
            min_support = FeatureSupport(min_support)

        # Order of support levels
        support_levels = {
            FeatureSupport.NOT_SUPPORTED: 0,
            FeatureSupport.PARTIAL: 1,
            FeatureSupport.BETA: 2,
            FeatureSupport.FULL: 3,
        }

        return support_levels[feature_support] >= support_levels[min_support]

    @classmethod
    def get_supported_features(cls, model: str) -> Dict[str, FeatureSupport]:
        """Get all supported features and their levels for a model."""
        if model not in cls.CAPABILITIES:
            return {}
        return cls.CAPABILITIES[model]["features"]

    @classmethod
    def list_models_supporting(
        cls, feature: str, min_support: Optional[FeatureSupport] = None
    ) -> list[str]:
        """List all models that support a specific feature."""
        models = []
        for model in cls.CAPABILITIES:
            if cls.supports_feature(
                model, feature, min_support or FeatureSupport.PARTIAL
            ):
                models.append(model)
        return models
