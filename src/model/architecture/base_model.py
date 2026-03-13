"""Base neural network model class."""

from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import nn

from src.config.train_enum import ModelType, OutputType
from src.config.train_config import TrainConfig
from src.graph.spec.base_spec import BaseSpec
from src.config.batch_schema import x_dict_type, edge_index_dict_type


class BaseModel(ABC, nn.Module):
    """
    Abstract neural network model.

    RESPONSIBILITY:
    - Define encoder, graph convolution layers, decoder
    - Implement forward pass: x_dict, edge_index_dict → predictions
    - Return predictions dict keyed by output type

    DOES NOT:
    - Training loops (LitModel's job)
    - Loss computation (LitModel's job)
    - Data loading (Dataset's job)
    - Logging (LitModel's job)

    INPUT REQUIREMENTS:
    - Spec: input channels per node, edge types, output types
    - TrainConfig: hidden channels, num layers, activation

    OUTPUT:
    - Forward pass returns Dict[OutputType, torch.Tensor]
    """

    _registry: dict[ModelType, Self] = {}

    def __init__(self):
        """Initialize model."""
        super().__init__()

    @classmethod
    def register(cls, model_type: ModelType):
        """
        Decorator to register a model implementation.
        """

        def decorator(model_cls: Self) -> Self:
            cls._registry[model_type] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create_model(
        cls,
        train_config: TrainConfig,
        spec: BaseSpec,
    ) -> Self:
        """
        Factory method to create a model by type.

        INPUT:
        - train_config: Training config specifying model_type
        - spec: Graph specification

        OUTPUT:
        - Instantiated model

        RAISES:
        - ValueError: If model_type not registered
        """
        model_type = train_config.model_type

        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls._registry.keys())}"
            )

        model_cls: Self = cls._registry[model_type]
        return model_cls.build_from(train_config, spec)

    @classmethod
    @abstractmethod
    def build_from(
        cls,
        train_config: TrainConfig,
        spec: BaseSpec,
    ) -> Self:
        """
        Factory constructor. REQUIRED - each child implements this.

        This is the ONLY way to instantiate via factory pattern.
        Child classes must override and implement their initialization logic.

        INPUT:
        - train_config: Training configuration
        - spec: Graph specification
        - coefficients: Reflection coefficients (optional, not used in all models)

        OUTPUT:
        - Initialized model ready for forward pass
        """
        pass

    @abstractmethod
    def forward(
        self,
        x_dict: x_dict_type,
        edge_index_dict: edge_index_dict_type,
    ) -> dict[OutputType, torch.Tensor]:
        """
        Forward pass through the model.

        INPUT:
        - x_dict: Node features per type
            Keys: node type names (e.g., "base", "joint", "foot")
            Values: torch.Tensor [num_nodes_of_type, input_channels]

        - edge_index_dict: Edge connectivity per edge type
            Keys: (source_type, edge_type, target_type) tuples
            Values: torch.Tensor [2, num_edges] - source and target indices

        OUTPUT:
        - Dict mapping output types to predictions
            Keys: OutputType enum values (CONTACT, GRF, COM)
            Values: torch.Tensor [batch_size, output_channels]
        """
        pass
