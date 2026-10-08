"""MS-HGNN PyTorch Lightning training wrapper."""

from src.config.train_enum import ModelType
from src.model.training.base_lit_model import BaseLitModel


class MS_HGNN_LitModel(BaseLitModel):
    """PyTorch Lightning wrapper for MS-HGNN (shared multi-task trainer)."""


BaseLitModel.register(ModelType.MS_HGNN)(MS_HGNN_LitModel)  # type: ignore[arg-type]
