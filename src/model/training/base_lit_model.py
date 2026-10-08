"""Base PyTorch Lightning model wrapper with shared multi-task training logic."""

from typing import Self

import torch
import lightning.pytorch as pl
from torchmetrics import Metric

from src.config.train_enum import ModelType, OutputType, Stage
from src.config.train_config import TrainConfig
from src.config.batch_schema import HeteroDataBatch
from src.model.architecture.base_model import BaseModel
from src.graph.spec.base_spec import BaseSpec


class BaseLitModel(pl.LightningModule):
    """
    Shared PyTorch Lightning wrapper for all model architectures.

    RESPONSIBILITY:
    - Implement training/validation/test loops for ONE OR MORE output types
    - Compute losses and update metrics
    - Log losses per step and metrics per epoch
    - Configure optimizer
    """

    _registry: dict[ModelType, Self] = {}

    def __init__(
        self,
        model: BaseModel,
        spec: BaseSpec,
        train_config: TrainConfig,
    ):
        super().__init__()
        self.model = model
        self.spec = spec
        self.train_config = train_config

        if train_config.output_types:
            self.output_types: list[OutputType] = list(train_config.output_types)
        else:
            self.output_types = [train_config.output_type]
        self.output_type = self.output_types[0]
        self.optimizer = train_config.optimizer
        self.loss_weights = train_config.loss_weights or {}

        self.metrics: dict[OutputType, dict[Stage, dict[str, Metric]]] = {}
        for output_type in self.output_types:
            self.metrics[output_type] = self.spec.metric_functions(
                output_type,
                robot_mass=self.train_config.robot_mass,
                foot_contact_area=self.train_config.foot_contact_area,
            )

    @classmethod
    def register(cls, model_type: ModelType):
        """Decorator to register a LitModel implementation."""

        def decorator(lit_cls: Self) -> Self:
            cls._registry[model_type] = lit_cls
            return lit_cls

        return decorator

    @classmethod
    def create_lit_model(
        cls, model: BaseModel, spec: BaseSpec, train_config: TrainConfig
    ) -> Self:
        """Factory method to create a Lightning wrapper by model type."""
        model_type = train_config.model_type

        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown LitModel type: {model_type}. "
                f"Available: {list(cls._registry.keys())}"
            )

        lit_cls: Self = cls._registry[model_type]
        return lit_cls.build_from(model, spec, train_config)

    @classmethod
    def build_from(
        cls, model: BaseModel, spec: BaseSpec, train_config: TrainConfig
    ) -> Self:
        return cls(model, spec, train_config)

    @staticmethod
    def reshape_for_output(
        output_type: OutputType, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_type in (
            OutputType.CONTACT,
            OutputType.JOINT_ACCELERATION,
            OutputType.JOINT_FRICTION,
        ):
            return pred.reshape(-1), target.reshape(-1)
        if output_type == OutputType.GROUND_REACTION_FORCE:
            return pred.reshape(-1, 3), target.reshape(-1, 3)
        return pred, target.squeeze(1)

    def _forward(self, batch: HeteroDataBatch) -> dict[OutputType, torch.Tensor]:
        return self.model.forward(batch.x_dict, batch.edge_index_dict)

    def _compute_loss_and_metrics(
        self, batch: HeteroDataBatch, stage: Stage
    ) -> torch.Tensor:
        predictions = self._forward(batch)
        batch_size = int(getattr(batch, "num_graphs", 1))
        total_loss = torch.tensor(0.0, device=self.device)
        stage_str = stage.value

        for output_type in self.output_types:
            target_key = f"y_{output_type.value.lower()}"
            if not hasattr(batch, target_key):
                continue
            pred = predictions[output_type]
            target = batch[target_key]
            pred, target = self.reshape_for_output(output_type, pred, target)

            loss_fn = self.spec.loss_function(output_type)
            loss = loss_fn(pred, target)
            if not torch.isfinite(pred).all():
                raise RuntimeError(
                    f"Non-finite predictions detected at stage={stage_str} for {output_type.value}."
                )
            if not torch.isfinite(target).all():
                raise RuntimeError(
                    f"Non-finite targets detected at stage={stage_str} for {output_type.value}."
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at stage={stage_str} for {output_type.value}."
                )
            loss_weight = float(self.loss_weights.get(output_type.value, 1.0))
            total_loss = total_loss + loss_weight * loss
            self.log(
                f"{stage_str}_loss_{output_type.value.lower()}",
                loss,
                batch_size=batch_size,
            )

            for name, metric in self.metrics[output_type][stage].items():
                metric = metric.to(self.device)
                self.metrics[output_type][stage][name] = metric
                try:
                    metric.update(pred.detach().float(), target.detach().float())
                except (RuntimeError, ValueError):
                    continue

        self.log(
            f"{stage_str}_total_loss",
            total_loss,
            prog_bar=True,
            batch_size=batch_size,
        )
        return total_loss

    def _log_epoch_metrics(self, stage: Stage) -> None:
        stage_str = stage.value
        for output_type in self.output_types:
            for name, metric in self.metrics[output_type][stage].items():
                try:
                    value = torch.as_tensor(metric.compute(), dtype=torch.float32)
                    self.log(
                        f"{stage_str}_{output_type.value.lower()}_{name}",
                        value,
                    )
                except (RuntimeError, ValueError):
                    continue
                finally:
                    metric.reset()

    def training_step(self, batch: HeteroDataBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_and_metrics(batch, Stage.TRAIN)

    def validation_step(self, batch: HeteroDataBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_and_metrics(batch, Stage.VAL)

    def test_step(self, batch: HeteroDataBatch, batch_idx: int) -> torch.Tensor:
        return self._compute_loss_and_metrics(batch, Stage.TEST)

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.VAL)

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.TEST)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_kwargs = {"lr": self.train_config.learning_rate}
        return self.optimizer(self.parameters(), **optimizer_kwargs)  # type: ignore[call-arg]
