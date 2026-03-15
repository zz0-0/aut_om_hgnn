"""
OM-HGNN Training Orchestrator

This script coordinates the entire training pipeline:
YAML Config → Spec → Parser → Morphology → Dataset → Model → LitModel → Trainer

Each step is clearly separated to match the architecture layers.
"""

import argparse
import logging
from pathlib import Path
from typing import Literal, cast

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
import torch
from torch_geometric.data import Batch  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore

# ============================================================================
# IMPORTS FROM MODULES (in order of usage)
# ============================================================================

# Layer 1: Data Definition
from src.config.train_enum import ModelType
from src.graph.symmetry.base_symmetry import BaseSymmetry
from src.config.train_config import TrainConfig
from src.graph.spec.base_spec import BaseSpec

# Layer 2: Robot Description
from src.graph.parser.base_parser import BaseParser
from src.graph.morphology.base_morphology import RobotMorphology

# Layer 3: Data Pipeline
from src.data.dataset.base_dataset import BaseDataset

# Layer 4: Model
from src.model.architecture.base_model import BaseModel

# Layer 5: Training
from src.model.training.base_lit_model import BaseLitModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    """
    Orchestrate the full training pipeline.

    Flow:
    YAML → TrainConfig → Spec → Parser → Morphology → Dataset → Model → LitModel → Trainer
    """

    # =================================================================
    # STEP 0: Parse command-line arguments
    # =================================================================
    # User provides: config path, optional W&B settings, etc.
    args = parse_args()
    logger = setup_logging()

    torch.set_float32_matmul_precision("high")

    # Set global RNG seeds for reproducible multi-seed experiments.
    seed_everything(args.seed, workers=True)

    logger.info("=" * 50)
    logger.info("OM-HGNN TRAINING PIPELINE")
    logger.info("=" * 50)

    # =================================================================
    # STEP 1: Load Configuration (Layer 1)
    # =================================================================
    # Input: YAML file path
    # Output: TrainConfig with all parameters
    #
    # The config declares:
    # - model_type (MI_HGNN, MS_HGNN)
    # - spec_type (BHMG, QHMG)
    # - robot_type (G1_29DOF, GO2)
    # - parser_path, dataset_path
    # - hyperparams (learning_rate, batch_size, etc.)
    logger.info("Step 1: Loading configuration...")
    config_path = Path(args.config_path)
    train_config = TrainConfig.build_from(str(config_path))
    logger.info(f"✓ Config loaded: {train_config.spec_type} {train_config.robot_type}")

    # =================================================================
    # STEP 2: Create Specification (Layer 1)
    # =================================================================
    # Input: spec_type from config (e.g., "BHMG", "QHMG")
    # Output: Spec object that defines:
    # - Node types and input channels (BASE: 6, JOINT: 10, FOOT: 6, etc.)
    # - Edge relations (BASE ↔ JOINT ↔ FOOT, etc.)
    # - Output types to predict (CONTACT, GRF, COM)
    # - Loss functions and metrics for each output
    #
    # REVERSE CONSTRAINT:
    # Spec is NOT just a passive schema. It's a BIDIRECTIONAL CONTRACT:
    # - Reflects what Model REQUIRES as inputs (x_dict shape, edge types)
    # - Declares what Dataset MUST PRODUCE
    # - All upstream components reverse-depend on Spec's decisions
    # See ARCHITECTURE_CONSTRAINTS.md for detailed dependency analysis.
    logger.info("Step 2: Creating specification...")
    spec = BaseSpec.create_spec(train_config.spec_type, train_config.symmetry_type)
    logger.info(f"✓ {train_config.spec_type} specification created")
    logger.info(
        f"  - Node types: {list(spec.node_types_with_history(train_config.history_length).keys())}"
    )

    # =================================================================
    # STEP 3: Parse Robot Morphology (Layer 2)
    # =================================================================
    # Input:
    # - parser_path (USD file) from config
    # - Spec (tells parser what to validate)
    #
    # Output: Morphology object containing:
    # - Edge indices (which joints connect to which)
    # - Edge attributes (connection weights)
    # - Node metadata
    #
    # This is the CONCRETE structure of THIS specific robot instance
    #
    # REVERSE CONSTRAINT:
    # Parser is constrained by BOTH:
    # 1. Spec: "I need these node types (BASE, JOINT, FOOT) and these edges"
    # 2. Model (indirectly): "My layers expect edges in this exact structure"
    # Parser must extract morphology that matches Spec's requirements.
    # If Parser fails, it means USD file doesn't match model requirements.
    logger.info("Step 3: Parsing robot morphology...")
    parser_path = Path(train_config.parser_path)
    parser = BaseParser.create_parser(
        robot_type=train_config.robot_type,
        model_type=train_config.model_type,
        spec=spec,
        parser_path=parser_path,
    )
    morphology: RobotMorphology = parser.parse()
    # coefficients = parser.get_reflection_coefficients()
    logger.info(f"✓ Robot parsed from {parser_path.name}")
    # logger.info(f"  - Nodes: {morphology.node_count}")
    # logger.info(f"  - Edges: {morphology.edge_count}")

    # =================================================================
    # STEP 4: Create Dataset (Layer 3)
    # =================================================================
    # Input:
    # - dataset_path (Numpy memmap file) from config
    # - Morphology (edge structure from parser)
    # - Spec (tells dataset what node types to expect)
    #
    # Output: PyTorch Dataset yielding HeteroData samples
    # Each sample contains:
    # - x_dict: {node_type: feature_tensor} (e.g., BASE, JOINT, FOOT, etc.)
    # - edge_index_dict: {edge_type: edge_indices}
    # - y_contact_states, y_contact_forces, y_com: labels
    #
    # The Dataset internally uses FeatureExtractor to:
    # - Load Numpy memmap timestep data
    # - Extract features for each node type (using Spec to know what's needed)
    # - Assemble into graph samples
    #
    # REVERSE CONSTRAINT (CRITICAL):
    # Dataset is NOT independent - it's fully constrained by:
    # 1. Spec: "JOINT nodes need 10 input channels, BASE needs 6"
    # 2. Model (indirectly): "I accept x_dict with these exact dimensions"
    # Dataset's FeatureExtractor must:
    # - Know which raw data fields to extract (from Spec)
    # - Extract exactly the right number of features per node type
    # - Match Model's expected input_channels exactly
    # This is a HARD constraint: mismatch = runtime error or wrong predictions.
    logger.info("Step 4: Creating dataset...")
    dataset_path = Path(train_config.dataset_path)
    robot_type = train_config.robot_type
    dataset = BaseDataset.create_dataset(
        dataset_path=dataset_path,
        morphology=morphology,
        spec=spec,
        robot_type=robot_type,
        history_length=train_config.history_length,
    )
    logger.info(f"✓ Dataset created: {len(dataset)} samples")

    # Create DataLoaders for training and validation
    train_size = int(len(dataset) * (1 - train_config.val_split_ratio))
    val_size = len(dataset) - train_size
    # Keep split reproducible (and optionally independent from training seed).
    split_generator = torch.Generator().manual_seed(args.split_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(  # type: ignore
        dataset, [train_size, val_size], generator=split_generator
    )

    # =================================================================
    # STEP 4.5: Create Symmetry Expansion (Optional, MS-HGNN only)
    # =================================================================
    # Input:
    # - symmetry_type from config (e.g., "K4", "C2", "S4")
    # - spec (provides symmetry edge types for validation)
    #
    # Output: Symmetry object with expand_batch() method
    #
    # Purpose:
    # MS-HGNN models require multi-base graph representations where the
    # same robot structure is duplicated and transformed under different
    # symmetry operations (e.g., identity, transversal, sagittal, rotational).
    #
    # This happens at BATCH TIME via custom collate_fn:
    # - Dataset yields single-base HeteroData samples (standard)
    # - collate_fn expands each sample to multi-base representation
    # - Model receives expanded batch with symmetry edges
    #
    # MI-HGNN models skip this step entirely (no symmetry expansion).
    #
    # ARCHITECTURAL BENEFIT:
    # By handling symmetry expansion in collate_fn rather than dataset:
    # - Dataset remains simple and symmetry-agnostic
    # - No memory bloat from pre-expanded dataset
    # - Easy to toggle symmetry on/off via config
    # - Symmetry logic is isolated and testable
    logger.info("Step 4.5: Creating symmetry expansion (only for MS-HGNN)...")

    collate_fn = None  # Default: no custom collate (MI-HGNN path)

    if train_config.model_type == ModelType.MS_HGNN:
        symmetry = BaseSymmetry.create_symmetry(
            train_config.symmetry_type,
            spec.symmetry_edge_mapping(),
            spec.symmetry_permutation_mapping(),
        )
        logger.info(f"✓ Symmetry created: {train_config.symmetry_type.value} group")
        collate_fn = symmetry.create_collate_fn()
        logger.info(
            f" ✓ Custom collate function created for {train_config.symmetry_type}"
        )

    # Create DataLoaders with optional symmetry collate_fn
    train_loader = DataLoader(  # type: ignore
        train_dataset,  # type: ignore
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(  # type: ignore
        val_dataset,  # type: ignore
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    logger.info(f"  - Train samples: {len(train_dataset)}")  # type: ignore
    logger.info(f"  - Val samples: {len(val_dataset)}")  # type: ignore

    # =================================================================
    # STEP 5: Create Model Architecture (Layer 4)
    # =================================================================
    # Input:
    # - Spec (tells model: input_channels per node, edge types)
    # - TrainConfig (hidden_channels, num_layers, activation)
    #
    # Output: Neural network module
    # Defines:
    # - Encoder: input_channels → hidden_channels
    # - Graph convolution layers
    # - Decoder: hidden_channels → output_channels
    #
    # Model does NOT know about:
    # - Training (that's LitModel's job)
    # - Loss computation (that's LitModel's job)
    # - Data loading (that's Dataset's job)
    #
    # CONSTRAINT SOURCE (CRITICAL):
    # This Model is the ROOT CONSTRAINT generator for the entire pipeline.
    # Its input/output requirements propagate BACKWARD through all layers:
    # - Model needs x_dict with specific keys and shapes
    # - This constrains what Dataset must extract
    # - This constrains what Spec must declare
    # - This constrains what Morphology must provide
    # - This constrains what Parser must extract
    # Left-to-right execution, but RIGHT-to-LEFT dependency.
    # See ARCHITECTURE_CONSTRAINTS.md for full dependency diagram.
    logger.info("Step 5: Creating model architecture...")
    model = BaseModel.create_model(
        train_config=train_config,
        spec=spec,
        # coefficients=coefficients
    )
    logger.info(f"✓ Model created: {train_config.model_type}")
    logger.info(f"  - Hidden channels: {train_config.hidden_channels}")
    logger.info(f"  - Num layers: {train_config.num_layers}")

    # =================================================================
    # STEP 6: Wrap Model in Lightning Training Wrapper (Layer 5)
    # =================================================================
    # Input:
    # - Model (the architecture)
    # - Spec (tells LitModel: loss functions, output types, metrics)
    #
    # Output: Lightning module that handles:
    # - Forward pass through model
    # - Loss computation for each output type
    # - Metric computation
    # - Logging to TensorBoard/W&B
    # - Training/validation/test steps
    #
    # LitModel does NOT know about:
    # - Data loading (that's Dataset's job)
    # - Architecture design (that's Model's job)
    # - Orchestration (that's train.py's job)
    logger.info("Step 6: Creating Lightning training module...")
    lit_model = BaseLitModel.create_lit_model(
        model=model, spec=spec, train_config=train_config
    )
    logger.info(f"✓ Lightning module wrapped")

    # =================================================================
    # STEP 7: Configure PyTorch Lightning Trainer
    # =================================================================
    # Setup:
    # - GPU/CPU device
    # - Logger (TensorBoard/W&B)
    # - Callbacks (model checkpoints)
    # - Precision (16-bit mixed, 32-bit, etc.)
    logger.info("Step 7: Configuring trainer...")

    # Determine device
    if train_config.accelerator == "gpu" and torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        logger.info("✓ Using GPU")
    else:
        accelerator = "cpu"
        devices = "auto"
        logger.info("✓ Using CPU")

    precision = cast(
        Literal["16-mixed", "bf16-mixed", "32-true", "64-true"] | None,
        train_config.precision,
    )

    # Get checkpoint path if resuming (CLI override > config).
    checkpoint_path = args.resume_from_checkpoint or train_config.resume_from_checkpoint

    # Create logger
    config_stem = Path(args.config_path).stem
    default_run_name = f"{config_stem}_seed{args.seed:02d}"
    run_name = args.run_name or default_run_name
    wandb_save_dir = (PROJECT_ROOT / "wandb_logs").resolve()
    wandb_logger: Logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group,
        id=args.wandb_run_id,
        resume=args.wandb_resume,
        save_dir=str(wandb_save_dir),
        log_model=False,
    )

    checkpoint_root = Path(args.checkpoint_dir)
    if not checkpoint_root.is_absolute():
        checkpoint_root = (PROJECT_ROOT / checkpoint_root).resolve()
    checkpoint_dir = checkpoint_root / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="{epoch}-{val_total_loss:.2f}",
            monitor="val_total_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
    ]

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_val=train_config.gradient_clip_val,
        gradient_clip_algorithm=train_config.gradient_clip_algorithm,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        log_every_n_steps=10,
    )
    logger.info(f"✓ Trainer configured for {train_config.max_epochs} epochs")

    # =================================================================
    # STEP 8: Execute Training
    # =================================================================
    # The trainer now orchestrates the training loop:
    # 1. Iterates through DataLoader batches
    # 2. For each batch:
    #    a. LitModel.training_step() is called
    #    b. Model.forward() produces predictions
    #    c. Loss is computed using Spec's loss functions
    #    d. Metrics are computed using Spec's metrics
    #    e. Logging happens
    # 3. After each epoch, validation step runs
    # 4. Checkpoints are saved based on validation loss
    logger.info("=" * 50)
    logger.info("STARTING TRAINING")
    logger.info("=" * 50)

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,  # type: ignore
        val_dataloaders=val_loader,  # type: ignore
        ckpt_path=checkpoint_path,
    )

    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train OM-HGNN model")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="om-hgnn",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name (optional)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit W&B run name",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group name to cluster multi-seed runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed for reproducible training",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed for train/val split generator (fix this across seeds for fair comparison)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints"),
        help="Root directory where checkpoints are written (subfolder uses run name)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to resume from (overrides YAML resume field)",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Stable W&B run ID (enables resume on restarted sweeps)",
    )
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default="allow",
        choices=["allow", "must", "never", "auto"],
        help="W&B resume policy for --wandb-run-id",
    )
    return parser.parse_args()


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


if __name__ == "__main__":
    main()
