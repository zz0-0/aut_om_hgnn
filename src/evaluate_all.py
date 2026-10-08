"""Evaluate every multi-task config x seed and aggregate mean/std metrics.

For internal-data configs this evaluates in-domain (trajectory validation split)
and blind on the external-controller dataset. For external-data configs it
evaluates in-domain on the external dataset and blind on the internal dataset.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path("src/config/yaml")
CHECKPOINT_ROOT = Path("checkpoints/all_configs_seed_sweep")
OUTPUT_ROOT = Path("evaluation")
INTERNAL_DATASETS = {
    "g1": "src/data/dataset/datasets_memmap/g1_locomotion_memmap_friction",
    "go2": "src/data/dataset/datasets_memmap/go2_locomotion_memmap_friction",
}
EXTERNAL_DATASETS = {
    "g1": "src/data/dataset/datasets_memmap/g1_locomotion_memmap_friction_external",
    "go2": "src/data/dataset/datasets_memmap/go2_locomotion_memmap_friction_external",
}
LOSS_RE = re.compile(r"val_total_loss=([0-9.]+)\.ckpt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all configs and seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--pattern", type=str, default="*_multi_*.yaml")
    parser.add_argument(
        "--exclude-substring",
        type=str,
        nargs="+",
        default=None,
        help="Skip config filenames containing any of these substrings (e.g. _ext_ sa_ se_).",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--blind-max-batches",
        type=int,
        default=600,
        help="Cap the number of blind-evaluation batches (None = full dataset).",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--skip-blind",
        action="store_true",
        help="Skip blind evaluation on the cross-controller datasets.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def robot_of(stem: str) -> str:
    return "g1" if stem.startswith("g1_") else "go2"


def best_checkpoint(run_dir: Path) -> Path | None:
    candidates = []
    for path in run_dir.glob("epoch=*-val_total_loss=*.ckpt"):
        match = LOSS_RE.search(path.name)
        if match is not None:
            candidates.append((float(match.group(1)), path))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    last = run_dir / "last.ckpt"
    return last if last.exists() else None


def run_single(
    config_path: Path,
    checkpoint: Path,
    dataset_path: str | None,
    split: str,
    output_dir: Path,
    args: argparse.Namespace,
    max_batches: int | None = None,
) -> bool:
    command = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--config-path",
        str(config_path),
        "--checkpoint-path",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--split",
        split,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if dataset_path is not None:
        command += ["--dataset-path", dataset_path]
    if max_batches is not None:
        command += ["--max-batches", str(max_batches)]
    result = subprocess.run(command)
    return result.returncode == 0


def aggregate(stem: str, mode: str, seed_dirs: list[Path], output: Path) -> None:
    collected: list[dict] = []
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "metrics.json"
        if metrics_path.exists():
            collected.append(json.loads(metrics_path.read_text()))
    if not collected:
        return

    summary: dict = {"config": stem, "mode": mode, "num_seeds": len(collected)}
    task_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for record in collected:
        for task, metrics in record.get("metrics", {}).items():
            task_summary = task_metrics.setdefault(task, {})
            for name, value in metrics.items():
                stats = task_summary.setdefault(name, {"values": []})
                stats["values"].append(float(value))
    summary["metrics"] = {
        task: {
            name: {
                "mean": sum(stats["values"]) / len(stats["values"]),
                "std": (
                    (sum((v - sum(stats["values"]) / len(stats["values"])) ** 2 for v in stats["values"]) / len(stats["values"])) ** 0.5
                ),
                "n": len(stats["values"]),
            }
            for name, stats in metrics.items()
        }
        for task, metrics in task_metrics.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    all_config_paths = sorted(CONFIG_DIR.glob(args.pattern))
    if args.exclude_substring:
        all_config_paths = [
            path
            for path in all_config_paths
            if not any(token in path.name for token in args.exclude_substring)
        ]
    if not all_config_paths:
        raise SystemExit(f"No configs matched {args.pattern}")

    if args.aggregate_only:
        _aggregate_all(all_config_paths, args)
        print("Aggregation complete.")
        return

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Invalid shard configuration.")
    config_paths = [
        path
        for index, path in enumerate(all_config_paths)
        if index % args.num_shards == args.shard_index
    ]

    for config_path in config_paths:
        stem = config_path.stem
        robot = robot_of(stem)
        is_external = "_ext_" in stem
        for seed in args.seeds:
            run_dir = CHECKPOINT_ROOT / stem / f"{stem}_seed{seed:02d}"
            checkpoint = best_checkpoint(run_dir)
            if checkpoint is None:
                print(f"[skip] no checkpoint: {run_dir}")
                continue

            indomain_dir = OUTPUT_ROOT / stem / f"seed{seed:02d}" / "indomain"
            blind_dir = OUTPUT_ROOT / stem / f"seed{seed:02d}" / "blind"
            indomain_dataset = (
                EXTERNAL_DATASETS[robot] if is_external else INTERNAL_DATASETS[robot]
            )
            blind_dataset = (
                INTERNAL_DATASETS[robot] if is_external else EXTERNAL_DATASETS[robot]
            )

            if args.overwrite or not (indomain_dir / "metrics.json").exists():
                print(f"[eval] {stem} seed{seed:02d} in-domain ({indomain_dataset})")
                run_single(
                    config_path,
                    checkpoint,
                    indomain_dataset,
                    "val",
                    indomain_dir,
                    args,
                    max_batches=args.max_batches,
                )
            if not args.skip_blind and (
                args.overwrite or not (blind_dir / "metrics.json").exists()
            ):
                print(f"[eval] {stem} seed{seed:02d} blind ({blind_dataset})")
                run_single(
                    config_path,
                    checkpoint,
                    blind_dataset,
                    "all",
                    blind_dir,
                    args,
                    max_batches=args.blind_max_batches,
                )

    if args.num_shards == 1:
        _aggregate_all(all_config_paths, args)
    print(f"Shard {args.shard_index}/{args.num_shards} evaluation complete.")


def _aggregate_all(
    config_paths: list[Path], args: argparse.Namespace
) -> None:
    summary_root = OUTPUT_ROOT / "summary"
    modes = ("indomain",) if args.skip_blind else ("indomain", "blind")
    for config_path in config_paths:
        stem = config_path.stem
        for mode in modes:
            seed_dirs = [
                OUTPUT_ROOT / stem / f"seed{seed:02d}" / mode
                for seed in args.seeds
            ]
            aggregate(stem, mode, seed_dirs, summary_root / f"{stem}__{mode}.json")


if __name__ == "__main__":
    main()
