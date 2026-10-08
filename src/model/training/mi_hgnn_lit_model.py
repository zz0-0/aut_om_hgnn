"""MI-HGNN PyTorch Lightning training wrapper."""

from src.config.train_enum import ModelType
from src.model.training.base_lit_model import BaseLitModel


class MI_HGNN_LitModel(BaseLitModel):
    """PyTorch Lightning wrapper for MI-HGNN (shared multi-task trainer)."""


BaseLitModel.register(ModelType.MI_HGNN)(MI_HGNN_LitModel)  # type: ignore[arg-type]
