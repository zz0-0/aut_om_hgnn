"""Standalone evaluation and latency benchmarking for trained OM-HGNN checkpoints.

Loads a YAML training config and a Lightning checkpoint, rebuilds the exact
spec/parser/dataset pipeline, aggregates metrics over the validation split, and
optionally writes per-sample predictions and a batch-1 latency profile.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch_geometric.data import Batch  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore

from src.config.train_config import TrainConfig
from src.config.batch_schema import HeteroDataBatch
from src.config.train_enum import ModelType, OutputType, Stage
from src.graph.spec.base_spec import BaseSpec
from src.graph.parser.base_parser import BaseParser
from src.graph.symmetry.base_symmetry import BaseSymmetry
from src.data.dataset.base_dataset import BaseDataset
from src.model.architecture.base_model import BaseModel
from src.model.training.base_lit_model import BaseLitModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained OM-HGNN checkpoint."
    )
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override the dataset path (e.g., for blind external-controller evaluation).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "all"],
        default="val",
        help="'val' uses the trajectory-level validation split; 'all' evaluates every sample.",
    )
    parser.add_argument("--output-dir", type=str, default="evaluation")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--benchmark-latency", action="store_true")
    parser.add_argument(
        "--latency-batch-sizes", type=int, nargs="+", default=[1, 32, 128]
    )
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=50)
    return parser.parse_args()


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


def build_pipeline(config: TrainConfig):
    spec = BaseSpec.create_spec(config.spec_type, config.symmetry_type)
    parser = BaseParser.create_parser(
        config.robot_type,
        config.model_type,
        spec,
        config.parser_path,
        symmetry_edges=config.symmetry_edges,
    )
    morphology = parser.parse()
    dataset = BaseDataset.create_dataset(
        dataset_path=config.dataset_path,
        morphology=morphology,
        spec=spec,
        robot_type=config.robot_type,
        history_length=config.history_length,
    )
    return spec, morphology, dataset


def load_model(
    config: TrainConfig,
    spec: BaseSpec,
    morphology: Any,
    checkpoint_path: Path,
    device: torch.device,
):
    model = BaseModel.create_model(train_config=config, spec=spec)
    lit_model = BaseLitModel.create_lit_model(
        model=model, spec=spec, train_config=config
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = lit_model.load_state_dict(state_dict, strict=False)
    model_missing = [key for key in missing if key.startswith("model.")]
    model_unexpected = [key for key in unexpected if key.startswith("model.")]
    if model_missing or model_unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the configured model. "
            f"Missing model keys: {model_missing}; unexpected: {model_unexpected}"
        )
    lit_model.to(device)
    lit_model.eval()
    return lit_model, model


def benchmark_latency(
    model: BaseModel,
    dataset: BaseDataset,
    device: torch.device,
    batch_sizes: list[int],
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for batch_size in batch_sizes:
        samples = [dataset[index % len(dataset)] for index in range(batch_size)]
        batch = cast(HeteroDataBatch, Batch.from_data_list(samples).to(device))
        x_dict = batch.x_dict
        edge_dict = batch.edge_index_dict

        def run_forward() -> None:
            model(x_dict, edge_dict)

        with torch.no_grad():
            for _ in range(warmup):
                run_forward()
            if device.type == "cuda":
                torch.cuda.synchronize()

            timings: list[float] = []
            for _ in range(repeats):
                start = time.perf_counter()
                run_forward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)

        timings_np = np.asarray(timings, dtype=np.float64)
        results[str(batch_size)] = {
            "mean_ms": float(timings_np.mean()),
            "p50_ms": float(np.percentile(timings_np, 50)),
            "p95_ms": float(np.percentile(timings_np, 95)),
            "per_sample_ms": float(timings_np.mean() / batch_size),
        }
    return results


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)

    config = TrainConfig.build_from(args.config_path)
    if args.dataset_path is not None:
        config.dataset_path = Path(args.dataset_path)
    spec, morphology, dataset = build_pipeline(config)

    if args.split == "all":
        val_dataset = dataset  # type: ignore[assignment]
    else:
        _, val_indices = dataset.trajectory_split_indices(
            config.val_split_ratio, args.split_seed
        )
        val_dataset = torch.utils.data.Subset(dataset, val_indices.tolist())  # type: ignore

    collate_fn = None
    if config.model_type == ModelType.MS_HGNN:
        symmetry = BaseSymmetry.create_symmetry(
            config.symmetry_type,
            spec.symmetry_edge_mapping(),
            morphology.symmetry_permutation_dict,
        )
        collate_fn = symmetry.create_collate_fn(augment=False)

    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    lit_model, model = load_model(
        config, spec, morphology, Path(args.checkpoint_path), device
    )
    output_types = (
        list(config.output_types) if config.output_types else [config.output_type]
    )
    primary_output_type = output_types[0]
    metrics_by_task: dict[OutputType, dict] = {}
    for output_type in output_types:
        task_metrics = spec.metric_functions(
            output_type,
            robot_mass=config.robot_mass,
            foot_contact_area=config.foot_contact_area,
        )[Stage.TEST]
        metrics_by_task[output_type] = {
            name: metric.to(device) for name, metric in task_metrics.items()
        }

    num_feet = len(morphology.node_type_usd_node_dict.get("foot", []))
    total_grf_metrics = None
    if OutputType.GROUND_REACTION_FORCE in output_types:
        total_grf_metrics = spec.metric_functions(
            OutputType.TOTAL_GROUND_REACTION_FORCE,
            robot_mass=config.robot_mass,
            foot_contact_area=config.foot_contact_area,
        )[Stage.TEST]
        for name, metric in total_grf_metrics.items():
            total_grf_metrics[name] = metric.to(device)

    predicted_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    total_pred_batches: list[np.ndarray] = []
    total_target_batches: list[np.ndarray] = []
    lut_batches: list[np.ndarray] = []
    loss_sums: dict[OutputType, float] = {ot: 0.0 for ot in output_types}
    loss_batches: dict[OutputType, int] = {ot: 0 for ot in output_types}
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = cast(HeteroDataBatch, batch.to(device))
            predictions = model(batch.x_dict, batch.edge_index_dict)
            num_batches += 1
            primary_pred = primary_target = None

            for output_type in output_types:
                target_key = f"y_{output_type.value.lower()}"
                if not hasattr(batch, target_key):
                    continue
                pred = predictions[output_type]
                target = batch[target_key]
                pred, target = reshape_for_output(output_type, pred, target)
                if pred.shape != target.shape:
                    raise RuntimeError(
                        f"Prediction/target shape mismatch for {output_type.value}: "
                        f"{tuple(pred.shape)} vs {tuple(target.shape)}"
                    )
                loss = spec.loss_function(output_type)(pred, target)
                loss_sums[output_type] += float(loss)
                loss_batches[output_type] += 1
                for name, metric in metrics_by_task[output_type].items():
                    metric.update(pred.float(), target.float())

                if output_type == OutputType.GROUND_REACTION_FORCE:
                    total_pred = pred.reshape(-1, num_feet, 3).sum(dim=1)
                    total_target = target.reshape(-1, num_feet, 3).sum(dim=1)
                    if total_grf_metrics is not None:
                        for name, metric in total_grf_metrics.items():
                            metric.update(total_pred.float(), total_target.float())
                    if args.save_predictions:
                        total_pred_batches.append(total_pred.detach().cpu().numpy())
                        total_target_batches.append(
                            total_target.detach().cpu().numpy()
                        )

                if output_type == primary_output_type and args.save_predictions:
                    primary_pred = pred.detach().cpu().numpy()
                    primary_target = target.detach().cpu().numpy()

            if args.save_predictions:
                if primary_pred is not None:
                    predicted_batches.append(primary_pred)
                    target_batches.append(primary_target)
                start = batch_idx * loader.batch_size
                end = min(start + int(batch.num_graphs), len(val_dataset))
                if args.split == "all":
                    indices = np.arange(start, end, dtype=np.int64)
                else:
                    indices = np.asarray(
                        val_dataset.indices[start:end],  # type: ignore[attr-defined]
                        dtype=np.int64,
                    )
                lut_batches.append(dataset.base_sample_lut[indices].astype(np.int32))

            if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
                break
            if (batch_idx + 1) % 100 == 0:
                logging.info("Evaluated %d batches", batch_idx + 1)

    results: dict[str, Any] = {
        "config": str(args.config_path),
        "checkpoint": str(args.checkpoint_path),
        "dataset": str(config.dataset_path),
        "split": args.split,
        "output_types": [output_type.value for output_type in output_types],
        "num_batches": num_batches,
        "mean_batch_loss": {
            output_type.value: loss_sums[output_type]
            / max(loss_batches[output_type], 1)
            for output_type in output_types
        },
        "metrics": {
            output_type.value: {
                name: float(metric.compute())
                for name, metric in metrics_by_task[output_type].items()
            }
            for output_type in output_types
        },
    }

    if total_grf_metrics is not None:
        results["derived_total_grf"] = {
            name: float(metric.compute())
            for name, metric in total_grf_metrics.items()
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    logging.info("Metrics: %s", json.dumps(results["metrics"], indent=2))

    if args.save_predictions and predicted_batches:
        payload = {
            "pred": np.concatenate(predicted_batches, axis=0),
            "target": np.concatenate(target_batches, axis=0),
            "sample_lut": np.concatenate(lut_batches, axis=0),
        }
        if total_pred_batches:
            payload["total_pred"] = np.concatenate(total_pred_batches, axis=0)
            payload["total_target"] = np.concatenate(total_target_batches, axis=0)
        np.savez_compressed(output_dir / "predictions.npz", **payload)

    if args.benchmark_latency:
        latency = benchmark_latency(
            model,
            dataset,
            device,
            args.latency_batch_sizes,
            args.latency_warmup,
            args.latency_repeats,
        )
        (output_dir / "latency.json").write_text(json.dumps(latency, indent=2))
        logging.info("Latency: %s", json.dumps(latency, indent=2))


if __name__ == "__main__":
    main()
