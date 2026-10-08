"""Numerical symmetry-consistency measurement for MS-HGNN models.

Measures ``model(T_g x)`` against ``T_g model(x)`` for every symmetry group
element, where ``T_g`` applies the same node permutation and feature/label
reflection that the training-time augmentation uses.

Note: MS-HGNN is symmetry-*aware* (symmetric message passing + symmetry-based
augmentation), but its dense encoder/decoder/base-transform layers do not
commute with the reflection representation, so strict equivariance is not
expected. This script quantifies the deviation for reporting.
"""

import argparse
import json
from pathlib import Path

import torch

from src.config.train_config import TrainConfig
from src.config.train_enum import ModelType
from src.graph.spec.base_spec import BaseSpec
from src.graph.parser.base_parser import BaseParser
from src.graph.symmetry.base_symmetry import (
    BaseSymmetry,
    FEATURE_TYPE_WIDTH,
    OUTPUT_LABEL_LAYOUT,
)
from src.data.dataset.base_dataset import BaseDataset
from src.evaluate import load_model, build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MS-HGNN equivariance check.")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--fail-on-deviation", action="store_true")
    parser.add_argument("--output-dir", type=str, default="evaluation/equivariance")
    return parser.parse_args()


def transform_prediction(
    symmetry: BaseSymmetry,
    prediction: torch.Tensor,
    node_type: str,
    feature_types: list,
    combo: tuple[str, ...],
) -> torch.Tensor:
    flat = prediction.reshape(1, -1)
    transformed = symmetry.transform_label(flat, node_type, feature_types, combo)
    width = sum(FEATURE_TYPE_WIDTH[feature_type] for feature_type in feature_types)
    if width > 0 and flat.shape[1] % width == 0:
        return transformed.reshape(prediction.shape)
    return transformed


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    config = TrainConfig.build_from(args.config_path)
    if config.model_type != ModelType.MS_HGNN:
        raise SystemExit("Equivariance check only applies to MS_HGNN configs.")

    spec, morphology, dataset = build_pipeline(config)
    symmetry = BaseSymmetry.create_symmetry(
        config.symmetry_type,
        spec.symmetry_edge_mapping(),
        morphology.symmetry_permutation_dict or spec.symmetry_permutation_mapping(),
    )

    if args.checkpoint_path is not None:
        _, model = load_model(
            config, spec, morphology, Path(args.checkpoint_path), device
        )
    else:
        from src.model.architecture.base_model import BaseModel

        model = BaseModel.create_model(train_config=config, spec=spec).to(device)
        model.eval()

    node_type = spec.output_node_type(config.output_type)
    label_key = f"y_{config.output_type.value.lower()}"
    feature_types = OUTPUT_LABEL_LAYOUT[label_key][1]

    samples = [
        dataset[index]
        for index in range(min(args.num_samples, len(dataset)))
    ]

    results: dict[str, dict[str, float]] = {}
    all_passed = True
    for combo in symmetry.combination:
        max_abs_error = 0.0
        max_reference = 0.0
        for sample in samples:
            x_dict = {key: value.to(device) for key, value in sample.x_dict.items()}
            edge_index_dict = {
                key: value.to(device)
                for key, value in sample.edge_index_dict.items()
            }
            with torch.no_grad():
                baseline = model(x_dict, edge_index_dict)[config.output_type]

            transformed_sample = symmetry.apply_symmetry_transform(sample, combo)
            transformed_x = {
                key: value.to(device)
                for key, value in transformed_sample.x_dict.items()
            }
            with torch.no_grad():
                transformed_prediction = model(
                    transformed_x, edge_index_dict
                )[config.output_type]

            expected = transform_prediction(
                symmetry, baseline, node_type, feature_types, combo
            ).to(device)
            max_abs_error = max(
                max_abs_error, float((transformed_prediction - expected).abs().max())
            )
            max_reference = max(max_reference, float(expected.abs().max()))

        tolerance = args.tolerance * max(max_reference, 1.0)
        relative_error = max_abs_error / max(max_reference, 1.0)
        passed = max_abs_error <= tolerance
        all_passed = all_passed and passed
        name = "∘".join(combo) if combo else "identity"
        results[name] = {
            "max_abs_error": max_abs_error,
            "max_reference": max_reference,
            "relative_error": relative_error,
            "passed": passed,
        }
        print(
            f"{name:>20s}: max_abs_error={max_abs_error:.3e} "
            f"max_reference={max_reference:.3e} passed={passed}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(args.config_path),
        "checkpoint": str(args.checkpoint_path),
        "output_type": config.output_type.value,
        "tolerance": args.tolerance,
        "passed": all_passed,
        "elements": results,
    }
    (output_dir / "equivariance.json").write_text(json.dumps(payload, indent=2))
    if args.fail_on_deviation and not all_passed:
        raise SystemExit("Equivariance check FAILED.")


if __name__ == "__main__":
    main()
