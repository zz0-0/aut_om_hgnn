from abc import ABC, abstractmethod
from typing import Optional, Self

from torch import nn
import torch
from torchmetrics import Metric
import torchmetrics

from src.config.train_enum import OutputType, SpecType, Stage, SymmetryType

type node_edge_relations_type = dict[str, list[tuple[str, str]]]
type symmetry_edge_dict_type = dict[str, dict[str, list[int]]]
type symmetry_permutation_dict_type = dict[str, dict[str, dict[str, list[int]]]]


class AxisMetric(Metric):
    """
    Wrapper metric that computes a metric on a specific axis of the output.

    For GRF (3D output), this allows computing metrics separately for x, y, z axes.
    """

    def __init__(self, base_metric: Metric, axis: int):
        """
        INPUT:
            - base_metric: The underlying torchmetrics metric to compute
            - axis: The axis index to slice (0=x, 1=y, 2=z)
        """
        super().__init__()
        self.base_metric = base_metric
        self.axis = axis

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric with sliced predictions and targets."""
        self.base_metric.update(  # type: ignore
            preds[..., self.axis].contiguous(),
            target[..., self.axis].contiguous(),
        )

    def compute(self) -> torch.Tensor:
        """Compute the metric value."""
        return self.base_metric.compute()  # type: ignore

    def reset(self) -> None:
        """Reset metric state."""
        self.base_metric.reset()


class PerFootMetric(Metric):
    """
    Wrapper metric that computes a metric for a single foot index.

    Supports flattened contact tensors (1D) and flattened GRF tensors (2D with last
    dim = output channels, e.g., 3 for x/y/z).
    """

    def __init__(self, base_metric: Metric, foot_index: int, num_feet: int):
        super().__init__()
        if num_feet < 1:
            raise ValueError(f"num_feet must be >= 1, got {num_feet}")
        if foot_index < 0 or foot_index >= num_feet:
            raise ValueError(
                f"foot_index must be in [0, {num_feet - 1}], got {foot_index}"
            )
        self.base_metric = base_metric
        self.foot_index = foot_index
        self.num_feet = num_feet

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric with the selected foot slice."""
        if preds.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch for PerFootMetric: "
                f"{tuple(preds.shape)} != {tuple(target.shape)}"
            )

        if preds.ndim == 1:
            # Contact path after flatten: [batch_size * num_feet]
            if preds.numel() % self.num_feet != 0:
                raise ValueError(
                    f"Cannot reshape 1D tensor of length {preds.numel()} into "
                    f"num_feet={self.num_feet}"
                )
            preds_per_foot = preds.reshape(-1, self.num_feet)
            target_per_foot = target.reshape(-1, self.num_feet)
            self.base_metric.update(  # type: ignore
                preds_per_foot[:, self.foot_index].contiguous(),
                target_per_foot[:, self.foot_index].contiguous(),
            )
            return

        if preds.ndim == 2:
            # GRF path after reshape: [batch_size * num_feet, channels]
            if preds.shape[0] % self.num_feet != 0:
                raise ValueError(
                    f"Cannot reshape first dim {preds.shape[0]} into "
                    f"num_feet={self.num_feet}"
                )
            channels = preds.shape[1]
            preds_per_foot = preds.reshape(-1, self.num_feet, channels)
            target_per_foot = target.reshape(-1, self.num_feet, channels)
            self.base_metric.update(  # type: ignore
                preds_per_foot[:, self.foot_index, :].contiguous(),
                target_per_foot[:, self.foot_index, :].contiguous(),
            )
            return

        raise ValueError(
            f"Unsupported tensor rank for PerFootMetric: preds.ndim={preds.ndim}"
        )

    def compute(self) -> torch.Tensor:
        return self.base_metric.compute()  # type: ignore

    def reset(self) -> None:
        self.base_metric.reset()


class ComponentMetric(Metric):
    """
    Wrapper metric that computes a metric on a component (linear or angular) of the output.

    For COM (6D output), this allows computing metrics separately for:
    - Linear component: indices 0:3 (position/linear velocity)
    - Angular component: indices 3:6 (angular velocity/rotation)
    """

    def __init__(self, base_metric: Metric, component: str):
        """
        INPUT:
            - base_metric: The underlying torchmetrics metric to compute
            - component: Either "linear" (indices 0:3) or "angular" (indices 3:6)
        """
        super().__init__()
        self.base_metric = base_metric
        if component == "linear":
            self.slice_start, self.slice_end = 0, 3
        elif component == "angular":
            self.slice_start, self.slice_end = 3, 6
        else:
            raise ValueError(
                f"Unknown component: {component}. Use 'linear' or 'angular'."
            )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric with sliced predictions and targets."""
        self.base_metric.update(  # type: ignore
            preds[..., self.slice_start : self.slice_end].contiguous(),
            target[..., self.slice_start : self.slice_end].contiguous(),
        )

    def compute(self) -> torch.Tensor:
        """Compute the metric value."""
        return self.base_metric.compute()  # type: ignore

    def reset(self) -> None:
        """Reset metric state."""
        self.base_metric.reset()


class WeightNormalizedMetric(Metric):
    """
    Wrapper metric that normalizes the computed metric by robot mass.

    Useful for GRF metrics to get mass-normalized values (e.g., acceleration instead of force).
    """

    def __init__(self, base_metric: Metric, robot_mass: Optional[float] = None):
        """
        INPUT:
            - base_metric: The underlying torchmetrics metric to compute
            - robot_mass: Robot mass in kg for normalization. If None, normalization is skipped.
        """
        super().__init__()
        self.base_metric = base_metric
        self.robot_mass = robot_mass

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric with predictions and targets."""
        self.base_metric.update(preds, target)  # type: ignore

    def compute(self) -> torch.Tensor:
        """Compute the metric value and normalize by mass if provided."""
        value = self.base_metric.compute()  # type: ignore
        if self.robot_mass is not None and self.robot_mass > 0:
            # Normalize by mass (for force → acceleration)
            # Also normalize by gravity to get in terms of "body weights"
            g = 9.81
            value = value / (self.robot_mass * g)  # type: ignore
        return value  # type: ignore

    def reset(self) -> None:
        """Reset metric state."""
        self.base_metric.reset()


class AreaNormalizedMetric(Metric):
    """
    Wrapper metric that normalizes the computed metric by one-foot contact area.

    Useful for GRF metrics to get area-normalized values (force -> pressure).
    """

    def __init__(self, base_metric: Metric, foot_contact_area: Optional[float] = None):
        """
        INPUT:
            - base_metric: The underlying torchmetrics metric to compute
            - foot_contact_area: One-foot contact area in m^2. If None, normalization is skipped.
        """
        super().__init__()
        self.base_metric = base_metric
        self.foot_contact_area = foot_contact_area

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric with predictions and targets."""
        self.base_metric.update(preds, target)  # type: ignore

    def compute(self) -> torch.Tensor:
        """Compute the metric value and normalize by area if provided."""
        value = self.base_metric.compute()  # type: ignore
        if self.foot_contact_area is not None and self.foot_contact_area > 0:
            value = value / self.foot_contact_area  # type: ignore
        return value  # type: ignore

    def reset(self) -> None:
        """Reset metric state."""
        self.base_metric.reset()


class WeightAndAreaNormalizedMetric(Metric):
    """
    Wrapper metric that normalizes the computed metric by robot weight and contact area.

    Useful for GRF metrics to get combined mass-and-area normalized values.
    """

    def __init__(
        self,
        base_metric: Metric,
        robot_mass: Optional[float] = None,
        foot_contact_area: Optional[float] = None,
    ):
        """
        INPUT:
            - base_metric: The underlying torchmetrics metric to compute
            - robot_mass: Robot mass in kg
            - foot_contact_area: One-foot contact area in m^2
        """
        super().__init__()
        self.base_metric = base_metric
        self.robot_mass = robot_mass
        self.foot_contact_area = foot_contact_area

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric with predictions and targets."""
        self.base_metric.update(preds, target)  # type: ignore

    def compute(self) -> torch.Tensor:
        """Compute the metric value and normalize by mass*gravity*area if provided."""
        value = self.base_metric.compute()  # type: ignore
        if (
            self.robot_mass is not None
            and self.robot_mass > 0
            and self.foot_contact_area is not None
            and self.foot_contact_area > 0
        ):
            g = 9.81
            value = value / (self.robot_mass * g * self.foot_contact_area)  # type: ignore
        return value  # type: ignore

    def reset(self) -> None:
        """Reset metric state."""
        self.base_metric.reset()


class BaseSpec(ABC):
    """
    Abstract specification that defines the graph schema.

    Child classes should implement specific morphologies (BHMG, QHMG, etc.)

    RESPONSIBILITY:
    - Define what node types exist and their input dimensions
    - Define what edge relations exist between nodes
    - Define what output types are supported for prediction
    - Provide loss functions and metrics for outputs

    DOES NOT:
    - Parse robot files
    - Load data
    - Extract features
    - Build neural networks
    """

    _registry: dict[SpecType, Self] = {}

    def __init__(self):
        """Initialize specification."""
        super().__init__()

    @classmethod
    def register(cls, spec_name: SpecType):
        """
        Decorator to register a spec implementation.
        """

        def decorator(spec_cls: Self) -> Self:
            cls._registry[spec_name] = spec_cls
            return spec_cls

        return decorator

    @classmethod
    def create_spec(cls, spec_name: SpecType, symmetry_type: SymmetryType) -> Self:
        """
        Factory method to create a spec by name.

        INPUT:
        - spec_name: Name of spec to instantiate (e.g., SpecType.BHMG, SpecType.QHMG)

        OUTPUT:
        - Instantiated spec object

        RAISES:
        - ValueError: If spec_name not registered
        """
        if spec_name not in cls._registry:
            raise ValueError(
                f"Unknown spec type: {spec_name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        spec_cls: Self = cls._registry[spec_name]
        return spec_cls.build_from(symmetry_type)

    @classmethod
    @abstractmethod
    def build_from(cls, symmetry_type: SymmetryType) -> Self:
        """
        Factory constructor. REQUIRED - each child implements this.

        This is the ONLY way to instantiate via factory pattern.
        Child classes must override this method.

        INPUT:
        - None (or spec-specific parameters if needed)

        OUTPUT:
        - Initialized spec object ready to use for parsing, feature extraction, and model building
        """
        pass

    @abstractmethod
    def node_types_with_history(self, history_length: int) -> dict[str, int]:
        """
        Define node types and input feature dimensions with temporal stacking.

        INPUT:
        - history_length: Number of timesteps stacked into each sample

        OUTPUT:
        - Dict mapping node type to history-aware input feature dimension
        """
        pass

    @abstractmethod
    def node_edge_relations(self) -> node_edge_relations_type:
        """
        Define edges between node types.

        OUTPUT:
        - Dict mapping:
            - key: edge type name (e.g., "connect", "adjacent")
            - value: list of (source_node_type, target_node_type) tuples
        """
        pass

    @abstractmethod
    def node_edge_symmetry_relations(self) -> node_edge_relations_type:
        """
        Define symmetry edges between node types.

        Symmetry edges connect nodes that are symmetric (e.g., left and right limbs).
        These can be used to enforce weight sharing or regularization in the model.

        OUTPUT:
        - Dict mapping:
            - key: edge type name (e.g., "symmetry")
            - value: list of (source_node_type, target_node_type) tuples representing symmetric pairs
        """
        pass

    @abstractmethod
    def output_node_type(self, output_type: OutputType) -> str:
        """
        Return which node type produces predictions for this output.

        OUTPUT:
        - Node type name: "foot", "base", etc.
        """
        pass

    @abstractmethod
    def symmetry_edge_mapping(self) -> symmetry_edge_dict_type:
        """
        Define which symmetry edge types correspond to each symmetry type.

        This is used to determine which edges in the graph represent the symmetries
        for a given Spec. For example, a C2 symmetry might have one edge type "gt"
        representing left-right symmetry, while a K4 symmetry might have multiple edge
        types representing different symmetry relations.

        OUTPUT:
        - Dict mapping:
            - key: SymmetryType (e.g., SymmetryType.C2, SymmetryType.K4)
            - value: dict mapping:
                - key: edge type name (e.g., "gt", "gs")
                - value: list of axis indices to flip for that symmetry edge type (e.g., [1] for y-axis flip)
        """
        pass

    @abstractmethod
    def symmetry_permutation_mapping(self) -> symmetry_permutation_dict_type:
        """
        Define row-group permutation mappings for each symmetry operator.

        OUTPUT:
        - Dict mapping:
            - key: SymmetryType value (e.g., "C2", "K4", "S4")
            - value: dict mapping node type to operator map:
                - key: node type name (e.g., "joint", "foot")
                - value: dict mapping symmetry edge type to group permutation:
                    - key: operator name (e.g., "gt", "gs", "gr")
                    - value: list of target group indices (e.g., [1, 0, 3, 2])

        Notes:
        - Group permutation is applied on contiguous row blocks per node type.
        - If mapping is missing or incompatible with tensor row count, identity is used.
        """
        pass

    def loss_function(self, output_type: OutputType) -> nn.Module:
        """
        Return the loss function for this output type.

        OUTPUT:
        - Loss module (BCEWithLogitsLoss for binary, MSELoss for regression)
        """
        if output_type == OutputType.CONTACT:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.MSELoss()

    def output_channels(self, output_type: OutputType) -> int:
        """
        Return output prediction dimension for this type.

        OUTPUT:
        - Number of output features (1, 3, 6, etc.)
        """
        if output_type == OutputType.CONTACT:
            return 1
        elif output_type == OutputType.GROUND_REACTION_FORCE:
            return 3
        elif output_type == OutputType.CENTER_OF_MASS:
            return 6
        else:
            raise ValueError(f"Unknown output type: {output_type}")

    def _infer_foot_node_count(self) -> int:
        """Infer number of foot nodes from symmetry permutation definitions."""
        permutation_mapping = self.symmetry_permutation_mapping()
        for per_symmetry in permutation_mapping.values():
            foot_mapping = per_symmetry.get("foot")
            if foot_mapping is None:
                continue
            for permutation in foot_mapping.values():
                if len(permutation) > 0:
                    return len(permutation)
        raise ValueError(
            "Could not infer foot node count from symmetry_permutation_mapping()."
        )

    def metric_functions(
        self,
        output_type: OutputType,
        robot_mass: Optional[float] = None,
        foot_contact_area: Optional[float] = None,
    ) -> dict[Stage, dict[str, Metric]]:
        """
        Return appropriate metric functions for this output type.

        INPUT:
            - output_type: Type of output to create metrics for
            - robot_mass: Robot mass in kg for weight-normalized GRF metrics
            - foot_contact_area: One-foot contact area in m^2 for area-normalized GRF metrics

        OUTPUT:
            - Dict mapping:
                - key: stage name ("train", "val", "test")
                - value: Dict of metric name to Metric instance
        """

        if output_type == OutputType.CONTACT:
            base_metrics: dict[str, Metric] = {
                "accuracy": torchmetrics.Accuracy(task="binary"),
                "precision": torchmetrics.Precision(task="binary"),
                "recall": torchmetrics.Recall(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
                "specificity": torchmetrics.Specificity(task="binary"),
                "mcc": torchmetrics.MatthewsCorrCoef(task="binary"),
            }

            num_feet = self._infer_foot_node_count()
            for foot_idx in range(num_feet):
                base_metrics[f"accuracy_foot_{foot_idx}"] = PerFootMetric(
                    torchmetrics.Accuracy(task="binary"), foot_idx, num_feet
                )
                base_metrics[f"precision_foot_{foot_idx}"] = PerFootMetric(
                    torchmetrics.Precision(task="binary"), foot_idx, num_feet
                )
                base_metrics[f"recall_foot_{foot_idx}"] = PerFootMetric(
                    torchmetrics.Recall(task="binary"), foot_idx, num_feet
                )
                base_metrics[f"f1_foot_{foot_idx}"] = PerFootMetric(
                    torchmetrics.F1Score(task="binary"), foot_idx, num_feet
                )
                base_metrics[f"specificity_foot_{foot_idx}"] = PerFootMetric(
                    torchmetrics.Specificity(task="binary"), foot_idx, num_feet
                )
                base_metrics[f"mcc_foot_{foot_idx}"] = PerFootMetric(
                    torchmetrics.MatthewsCorrCoef(task="binary"), foot_idx, num_feet
                )

            return {stage: base_metrics.copy() for stage in Stage}
        elif output_type == OutputType.GROUND_REACTION_FORCE:
            # Create both aggregate metrics and per-axis metrics
            # Also create weight-normalized versions if robot_mass is provided
            axis_names = ["x", "y", "z"]

            base_metrics = {}

            # Overall metrics (computed on all 3 axes together)
            base_metrics["mae"] = torchmetrics.MeanAbsoluteError()
            base_metrics["mse"] = torchmetrics.MeanSquaredError()
            base_metrics["rmse"] = torchmetrics.MeanSquaredError(squared=False)
            base_metrics["r2_score"] = torchmetrics.R2Score()

            # Per-axis metrics
            for i, axis in enumerate(axis_names):
                base_metrics[f"mae_{axis}"] = AxisMetric(
                    torchmetrics.MeanAbsoluteError(), axis=i
                )
                base_metrics[f"mse_{axis}"] = AxisMetric(
                    torchmetrics.MeanSquaredError(), axis=i
                )
                base_metrics[f"rmse_{axis}"] = AxisMetric(
                    torchmetrics.MeanSquaredError(squared=False), axis=i
                )
                base_metrics[f"r2_score_{axis}"] = AxisMetric(
                    torchmetrics.R2Score(), axis=i
                )

            # Per-foot metrics
            num_feet = self._infer_foot_node_count()
            for foot_idx in range(num_feet):
                base_metrics[f"mae_foot_{foot_idx}"] = PerFootMetric(
                    torchmetrics.MeanAbsoluteError(), foot_idx, num_feet
                )
                base_metrics[f"mse_foot_{foot_idx}"] = PerFootMetric(
                    torchmetrics.MeanSquaredError(), foot_idx, num_feet
                )
                base_metrics[f"rmse_foot_{foot_idx}"] = PerFootMetric(
                    torchmetrics.MeanSquaredError(squared=False), foot_idx, num_feet
                )
                base_metrics[f"r2_score_foot_{foot_idx}"] = PerFootMetric(
                    torchmetrics.R2Score(), foot_idx, num_feet
                )
                for i, axis in enumerate(axis_names):
                    base_metrics[f"mae_{axis}_foot_{foot_idx}"] = PerFootMetric(
                        AxisMetric(torchmetrics.MeanAbsoluteError(), axis=i),
                        foot_idx,
                        num_feet,
                    )
                    base_metrics[f"mse_{axis}_foot_{foot_idx}"] = PerFootMetric(
                        AxisMetric(torchmetrics.MeanSquaredError(), axis=i),
                        foot_idx,
                        num_feet,
                    )
                    base_metrics[f"rmse_{axis}_foot_{foot_idx}"] = PerFootMetric(
                        AxisMetric(
                            torchmetrics.MeanSquaredError(squared=False), axis=i
                        ),
                        foot_idx,
                        num_feet,
                    )
                    base_metrics[f"r2_score_{axis}_foot_{foot_idx}"] = PerFootMetric(
                        AxisMetric(torchmetrics.R2Score(), axis=i),
                        foot_idx,
                        num_feet,
                    )

            # Weight-normalized metrics if robot_mass provided
            if robot_mass is not None and robot_mass > 0:
                base_metrics["mae_weight_norm"] = WeightNormalizedMetric(
                    torchmetrics.MeanAbsoluteError(), robot_mass
                )
                base_metrics["rmse_weight_norm"] = WeightNormalizedMetric(
                    torchmetrics.MeanSquaredError(squared=False), robot_mass
                )

                for i, axis in enumerate(axis_names):
                    base_metrics[f"mae_{axis}_weight_norm"] = WeightNormalizedMetric(
                        AxisMetric(torchmetrics.MeanAbsoluteError(), axis=i), robot_mass
                    )
                    base_metrics[f"rmse_{axis}_weight_norm"] = WeightNormalizedMetric(
                        AxisMetric(
                            torchmetrics.MeanSquaredError(squared=False), axis=i
                        ),
                        robot_mass,
                    )

            # Area-normalized metrics if foot_contact_area provided
            if foot_contact_area is not None and foot_contact_area > 0:
                base_metrics["mae_area_norm"] = AreaNormalizedMetric(
                    torchmetrics.MeanAbsoluteError(), foot_contact_area
                )
                base_metrics["rmse_area_norm"] = AreaNormalizedMetric(
                    torchmetrics.MeanSquaredError(squared=False), foot_contact_area
                )

                for i, axis in enumerate(axis_names):
                    base_metrics[f"mae_{axis}_area_norm"] = AreaNormalizedMetric(
                        AxisMetric(torchmetrics.MeanAbsoluteError(), axis=i),
                        foot_contact_area,
                    )
                    base_metrics[f"rmse_{axis}_area_norm"] = AreaNormalizedMetric(
                        AxisMetric(
                            torchmetrics.MeanSquaredError(squared=False), axis=i
                        ),
                        foot_contact_area,
                    )

            # Mass-and-area-normalized metrics if both provided
            if (
                robot_mass is not None
                and robot_mass > 0
                and foot_contact_area is not None
                and foot_contact_area > 0
            ):
                base_metrics["mae_weight_and_area_norm"] = (
                    WeightAndAreaNormalizedMetric(
                        torchmetrics.MeanAbsoluteError(), robot_mass, foot_contact_area
                    )
                )
                base_metrics["rmse_weight_and_area_norm"] = (
                    WeightAndAreaNormalizedMetric(
                        torchmetrics.MeanSquaredError(squared=False),
                        robot_mass,
                        foot_contact_area,
                    )
                )

                for i, axis in enumerate(axis_names):
                    base_metrics[f"mae_{axis}_weight_and_area_norm"] = (
                        WeightAndAreaNormalizedMetric(
                            AxisMetric(torchmetrics.MeanAbsoluteError(), axis=i),
                            robot_mass,
                            foot_contact_area,
                        )
                    )
                    base_metrics[f"rmse_{axis}_weight_and_area_norm"] = (
                        WeightAndAreaNormalizedMetric(
                            AxisMetric(
                                torchmetrics.MeanSquaredError(squared=False), axis=i
                            ),
                            robot_mass,
                            foot_contact_area,
                        )
                    )

            return {stage: base_metrics.copy() for stage in Stage}

        elif output_type == OutputType.CENTER_OF_MASS:
            # Create both aggregate metrics and per-component (linear/angular) metrics
            base_metrics = {}

            # Overall metrics (computed on all 6 dimensions together)
            base_metrics["mae"] = torchmetrics.MeanAbsoluteError()
            base_metrics["mse"] = torchmetrics.MeanSquaredError()
            base_metrics["rmse"] = torchmetrics.MeanSquaredError(squared=False)
            base_metrics["r2_score"] = torchmetrics.R2Score()
            base_metrics["cos_sim"] = torchmetrics.CosineSimilarity(reduction="mean")

            # Per-component metrics
            for component in ["linear", "angular"]:
                base_metrics[f"mae_{component}"] = ComponentMetric(
                    torchmetrics.MeanAbsoluteError(), component
                )
                base_metrics[f"mse_{component}"] = ComponentMetric(
                    torchmetrics.MeanSquaredError(), component
                )
                base_metrics[f"rmse_{component}"] = ComponentMetric(
                    torchmetrics.MeanSquaredError(squared=False), component
                )
                base_metrics[f"r2_score_{component}"] = ComponentMetric(
                    torchmetrics.R2Score(), component
                )
                base_metrics[f"cos_sim_{component}"] = ComponentMetric(
                    torchmetrics.CosineSimilarity(reduction="mean"), component
                )

            return {stage: base_metrics.copy() for stage in Stage}

        else:
            raise ValueError(f"Unknown output type: {output_type}")
