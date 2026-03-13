"""Training configuration holder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self

from torch import nn, optim
import yaml

from src.config.train_enum import (
    RobotType,
    SpecType,
    ModelType,
    OutputType,
    SymmetryType,
)


@dataclass
class TrainConfig:
    """
    Training configuration.

    Holds all parameters for a training run, loaded from YAML file.
    This is the central config object passed through the pipeline.
    """

    # ===== Model & Architecture =====
    model_type: ModelType
    spec_type: SpecType
    robot_type: RobotType
    output_type: OutputType  # Output type to train/predict
    symmetry_type: (
        SymmetryType  # Symmetry type for data augmentation (e.g., C2, S4, K4)
    )

    # ===== File Paths =====
    parser_path: Path  # USD file location
    dataset_path: Path  # Numpy memmap file location

    # ===== Data Pipeline =====
    val_split_ratio: float  # Train/val split (e.g., 0.2 = 80/20)
    batch_size: int
    num_workers: int  # DataLoader workers
    history_length: int  # Temporal window size (1 = no history stacking)

    # ===== Model Architecture =====
    hidden_channels: int  # Hidden layer dimension
    num_layers: int  # Number of graph convolution layers
    activation: nn.Module  # Activation function (ReLU, etc.)
    optimizer: type[optim.Optimizer]  # Optimizer class

    # ===== Training =====
    max_epochs: int
    learning_rate: float

    # ===== Hardware =====
    accelerator: str  # "cpu" or "gpu"
    precision: str  # "32-true", "16-mixed", "bf16-mixed"

    # ===== Optional =====
    resume_from_checkpoint: Optional[str] = None
    robot_mass: Optional[float] = None
    foot_contact_area: Optional[float] = None
    gradient_clip_val: Optional[float] = 1.0
    gradient_clip_algorithm: str = "norm"

    @classmethod
    def build_from(cls, config_path: str) -> Self:
        """
        Load configuration from YAML file.

        INPUT:
        - config_path: Path to YAML configuration file

        OUTPUT:
        - TrainConfig: Fully initialized configuration object

        RAISES:
        - FileNotFoundError: If config file doesn't exist
        - KeyError: If required fields are missing
        - ValueError: If enum values are invalid
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        output_type = OutputType[config_dict["output_type"]]

        activation_str = config_dict.get("activation")
        if activation_str == "ReLU":
            activation = nn.ReLU()
        elif activation_str == "LeakyReLU":
            activation = nn.LeakyReLU()
        elif activation_str == "Tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation_str}")

        optimizer_str = config_dict.get("optimizer")
        if optimizer_str == "Adam":
            optimizer = optim.Adam
        elif optimizer_str == "SGD":
            optimizer = optim.SGD
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_str}")

        symmetry_type = SymmetryType[config_dict["symmetry_type"]]

        return cls(
            model_type=ModelType[config_dict["model_type"]],
            spec_type=SpecType[config_dict["spec_type"]],
            robot_type=RobotType[config_dict["robot_type"]],
            output_type=output_type,
            symmetry_type=symmetry_type,
            parser_path=Path(config_dict["parser_path"]),
            dataset_path=Path(config_dict["dataset_path"]),
            val_split_ratio=float(config_dict["val_split_ratio"]),
            batch_size=int(config_dict["batch_size"]),
            num_workers=int(config_dict.get("num_workers", 0)),
            history_length=int(config_dict.get("history_length", 1)),
            hidden_channels=int(config_dict["hidden_channels"]),
            num_layers=int(config_dict["num_layers"]),
            activation=activation,
            optimizer=optimizer,
            max_epochs=int(config_dict["max_epochs"]),
            learning_rate=float(config_dict["learning_rate"]),
            gradient_clip_val=(
                float(config_dict["gradient_clip_val"])
                if "gradient_clip_val" in config_dict
                and config_dict["gradient_clip_val"] is not None
                else (None if "gradient_clip_val" in config_dict else 1.0)
            ),
            gradient_clip_algorithm=str(
                config_dict.get("gradient_clip_algorithm", "norm")
            ),
            accelerator=config_dict.get("accelerator", "cpu"),
            precision=config_dict.get("precision", "32-true"),
            resume_from_checkpoint=config_dict.get("resume_from_checkpoint", None),
            robot_mass=config_dict.get("robot_mass", None),
            foot_contact_area=config_dict.get("foot_contact_area", None),
        )
