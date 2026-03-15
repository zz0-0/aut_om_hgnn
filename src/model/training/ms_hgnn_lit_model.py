"""MS-HGNN PyTorch Lightning training wrapper."""

import torch
from torchmetrics import Metric

from src.config.train_enum import ModelType, OutputType, Stage
from src.config.train_config import TrainConfig
from src.config.batch_schema import HeteroDataBatch
from src.model.architecture.base_model import BaseModel
from src.model.training.base_lit_model import BaseLitModel
from src.graph.spec.base_spec import BaseSpec


class MS_HGNN_LitModel(BaseLitModel):
    """
    PyTorch Lightning wrapper for MS-HGNN model.

    Handles training loop, loss computation, metrics, and logging.
    """

    def __init__(
        self,
        model: BaseModel,
        spec: BaseSpec,
        train_config: TrainConfig,
    ):
        """
        Initialize MS-HGNN Lightning module.

        INPUT:
        - model: MS-HGNN neural network
        - spec: Graph specification with loss functions and metrics
        - train_config: Training configuration
        """
        super().__init__()

        self.model = model
        self.spec = spec
        self.train_config = train_config

        # Get output type and its loss functions and metrics from spec
        self.output_type = self.train_config.output_type
        self.optimizer = self.train_config.optimizer

        # Create metrics for each output type and stage
        self.metrics: dict[OutputType, dict[Stage, dict[str, Metric]]] = {}
        self.metrics[self.output_type] = self.spec.metric_functions(
            self.output_type,
            robot_mass=self.train_config.robot_mass,
            foot_contact_area=self.train_config.foot_contact_area,
        )

    @classmethod
    def build_from(cls, model: BaseModel, spec: BaseSpec, train_config: TrainConfig):
        """
        Factory constructor for MS-HGNN LitModel.

        INPUT:
        - model: MS-HGNN neural network
        - spec: Graph specification
        - train_config: Training configuration

        OUTPUT:
        - Initialized MS_HGNN_LitModel
        """
        return cls(model, spec, train_config)

    def _forward(self, batch: HeteroDataBatch) -> dict[OutputType, torch.Tensor]:
        """
        Forward pass through the model.

        INPUT:
        - batch: A collated HeteroData Batch with x_dict and edge_index_dict

        OUTPUT:
        - Model predictions dict
        """
        # Extract x_dict and edge_index_dict from collated HeteroData batch
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        return self.model.forward(x_dict, edge_index_dict)

    def _target(self, batch: HeteroDataBatch) -> torch.Tensor:
        """
        Extract target labels from batch.

        INPUT:
        - batch: HeteroData batch with labels

        OUTPUT:
        - Target tensor for the output type being predicted
        """
        target_key = f"y_{self.output_type.value.lower()}"
        return batch[target_key]

    def _update_metrics(
        self, stage: Stage, pred: torch.Tensor, target: torch.Tensor
    ) -> None:
        if self.output_type not in self.metrics:
            return

        metrics = self.metrics[self.output_type][stage]
        for metric_name, metric in metrics.items():
            metric = metric.to(pred.device)
            metrics[metric_name] = metric
            try:
                metric.update(pred, target)  # type: ignore[arg-type]
            except (RuntimeError, ValueError):
                # Do not let a single metric abort the whole training step.
                continue

    def _log_epoch_metrics(self, stage: Stage) -> None:
        if self.output_type not in self.metrics:
            return

        metrics = self.metrics[self.output_type][stage]
        for metric_name, metric in metrics.items():
            try:
                metric_value = metric.compute()
            except (RuntimeError, ValueError):
                # Some metrics (e.g., R2) require sufficient valid samples.
                # Skip logging for this epoch if not enough data was accumulated.
                metric.reset()
                continue

            self.log(
                f"{stage.value}_{self.output_type.value.lower()}_{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
            )
            metric.reset()

    def _compute_loss_and_metrics(
        self, batch: HeteroDataBatch, stage: Stage
    ) -> torch.Tensor:
        """
        Compute loss and log metrics for a batch.

        INPUT:
        - batch: HeteroData batch with labels
        - predictions: Model predictions dict
        - stage: Stage enum (TRAIN, VAL, TEST)

        OUTPUT:
        - Total loss scalar
        """
        total_loss = torch.tensor(0.0, device=self.device)

        output_type = self.output_type

        # Get predictions and targets
        pred = self._forward(batch)[output_type]
        target = self._target(batch)

        if output_type == OutputType.CONTACT:
            pred = pred.reshape(-1)
            target = target.reshape(-1)
        elif output_type == OutputType.GROUND_REACTION_FORCE:
            pred = pred.reshape(-1, 3)
            target = target.reshape(-1, 3)
        else:
            target = target.squeeze(1)

        batch_size = int(getattr(batch, "num_graphs", target.shape[0]))

        # Compute loss
        loss_fn = self.spec.loss_function(output_type)
        loss = loss_fn(pred, target)
        if not torch.isfinite(pred).all():
            raise RuntimeError(
                f"Non-finite predictions detected at stage={stage.value} for {output_type.value}."
            )
        if not torch.isfinite(target).all():
            raise RuntimeError(
                f"Non-finite targets detected at stage={stage.value} for {output_type.value}."
            )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected at stage={stage.value} for {output_type.value}."
            )
        total_loss += loss

        # Log loss
        self.log(
            f"{stage.value}_loss_{output_type.value.lower()}",
            loss,
            prog_bar=True,
            batch_size=batch_size,
        )

        self._update_metrics(stage, pred.detach().float(), target.detach().float())

        # Log total loss
        self.log(
            f"{stage.value}_total_loss",
            total_loss,
            prog_bar=True,
            batch_size=batch_size,
        )

        return total_loss

    def training_step(self, batch: HeteroDataBatch, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        INPUT:
        - batch: HeteroData batch
        - batch_idx: Batch index

        OUTPUT:
        - Loss scalar
        """

        return self._compute_loss_and_metrics(batch, Stage.TRAIN)

    def validation_step(self, batch: HeteroDataBatch, batch_idx: int) -> torch.Tensor:
        """
        Validation step.

        INPUT:
        - batch: HeteroData batch
        - batch_idx: Batch index

        OUTPUT:
        - Loss scalar
        """

        return self._compute_loss_and_metrics(batch, Stage.VAL)

    def test_step(self, batch: HeteroDataBatch, batch_idx: int) -> torch.Tensor:
        """
        Test step.

        INPUT:
        - batch: HeteroData batch
        - batch_idx: Batch index

        OUTPUT:
        - Loss scalar
        """
        return self._compute_loss_and_metrics(batch, Stage.TEST)

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.VAL)

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(Stage.TEST)

    def configure_optimizers(self):
        """
        Configure optimizer.

        OUTPUT:
        - Optimizer instance
        """
        optimizer_kwargs = {"lr": self.train_config.learning_rate}
        return self.optimizer(self.parameters(), **optimizer_kwargs)  # type: ignore[call-arg]


BaseLitModel.register(ModelType.MS_HGNN)(MS_HGNN_LitModel)  # type: ignore[arg-type]
