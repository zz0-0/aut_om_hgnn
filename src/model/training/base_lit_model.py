"""Base PyTorch Lightning model wrapper."""

from abc import ABC, abstractmethod
from typing import Self

import torch
import lightning.pytorch as pl

from src.config.train_enum import ModelType
from src.config.train_config import TrainConfig
from src.config.batch_schema import HeteroDataBatch
from src.model.architecture.base_model import BaseModel
from src.graph.spec.base_spec import BaseSpec


class BaseLitModel(ABC, pl.LightningModule):
    """
    Abstract PyTorch Lightning training wrapper.

    RESPONSIBILITY:
    - Implement training loop (training_step, validation_step, test_step)
    - Compute loss for each output type
    - Compute metrics
    - Log to TensorBoard/W&B or other loggers
    - Configure optimizer

    DOES NOT:
    - Define architecture (Model's job)
    - Load data (Dataset's job)
    - Orchestrate (train.py's job)

    INPUT REQUIREMENTS:
    - Model: the neural network architecture
    - Spec: loss functions, output types, metrics

    INTERACTIONS:
    - PyTorch Lightning Trainer calls:
        - training_step(batch) → loss
        - validation_step(batch) → loss
        - test_step(batch) → loss
        - configure_optimizers() → optimizer
    """

    _registry: dict[ModelType, Self] = {}

    def __init__(self):
        """Initialize Lightning module."""
        super().__init__()

    @classmethod
    def register(cls, model_type: ModelType):
        """
        Decorator to register a LitModel implementation.
        """

        def decorator(lit_cls: Self) -> Self:
            cls._registry[model_type] = lit_cls
            return lit_cls

        return decorator

    @classmethod
    def create_lit_model(
        cls, model: BaseModel, spec: BaseSpec, train_config: TrainConfig
    ) -> Self:
        """
        Factory method to create a Lightning wrapper by model type.

        INPUT:
        - model: Neural network model
        - spec: Graph specification
        - train_config: Training configuration

        OUTPUT:
        - Instantiated Lightning module ready for training

        RAISES:
        - ValueError: If model's type not registered
        """

        model_type = train_config.model_type

        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown LitModel type: {model_type}. "
                f"Available: {list(cls._registry.keys())}"
            )

        lit_cls: Self = cls._registry[model_type]
        return lit_cls.build_from(model, spec, train_config)

    @classmethod
    @abstractmethod
    def build_from(
        cls, model: BaseModel, spec: BaseSpec, train_config: TrainConfig
    ) -> Self:
        """
        Factory constructor. REQUIRED - each child implements this.

        This is the ONLY way to instantiate via factory pattern.
        Child classes must override and implement their initialization logic.

        INPUT:
        - model: Neural network model
        - spec: Graph specification
        - train_config: Training configuration
        OUTPUT:
        - Initialized Lightning module ready for training
        """
        pass

    @abstractmethod
    def training_step(self, batch: HeteroDataBatch, batch_idx: int) -> torch.Tensor:
        """
        Training step called by Trainer for each batch.

        INPUT:
        - batch: HeteroData batch containing:
            - x_dict: {node_type: features}
            - edge_index_dict: {edge_type: connectivity}
            - y_*: labels (y_contact_states, y_contact_forces, y_com)
        - batch_idx: batch index in epoch

        OUTPUT:
        - Loss scalar (torch.Tensor)

        RESPONSIBILITY:
        - Forward pass through model
        - Compute loss for each output type
        - Sum losses
        - Log metrics
        - Return total loss
        """
        pass

    @abstractmethod
    def validation_step(self, batch: HeteroDataBatch, batch_idx: int) -> torch.Tensor:
        """
        Validation step called by Trainer after each epoch.

        Same as training_step but:
        - No gradient computation
        - Logs to "val_*" metrics
        - Often uses no-grad context

        INPUT/OUTPUT: Same as training_step
        """
        pass

    @abstractmethod
    def test_step(self, batch: HeteroDataBatch, batch_idx: int) -> torch.Tensor:
        """
        Test step called by Trainer for final evaluation.

        Same as validation_step but logs to "test_*" metrics.

        INPUT/OUTPUT: Same as training_step
        """
        pass

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer.

        OUTPUT:
        - torch.optim.Optimizer instance
        """
        pass
