"""Run the full experiment matrix: all matched configs x all seeds.

This script is restart-safe:
- Tracks per-(config, seed) status in a state JSON.
- Skips completed pairs on rerun.
- Resumes failed/interrupted pairs from last.ckpt when available.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run src.train over all configs and seeds"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="src/config/yaml",
        help="Directory containing YAML configs",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.yaml",
        help="Glob pattern to select configs",
    )
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-supergroup",
        type=str,
        default="all_configs_seed_sweep",
        help="Top-level experiment label used for state/checkpoint namespace",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Seed list, e.g. --seeds 0 1 2 3 4 5 6 7 8 9",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Fixed train/val split seed for fair comparison",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="checkpoints",
        help="Root directory for all checkpoints and state",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Optional explicit state JSON path",
    )
    parser.add_argument(
        "--limit-configs",
        type=int,
        default=0,
        help="Optional cap on number of configs (0 = all)",
    )
    parser.add_argument(
        "--reset-completed",
        action="store_true",
        help="Ignore previous state and rerun completed pairs",
    )
    parser.add_argument(
        "--delete-interrupted-wandb",
        action="store_true",
        help="Best-effort delete interrupted W&B run before retry",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining pairs if one pair fails",
    )
    return parser.parse_args()


def _safe_slug(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    slug = "".join(ch if ch in allowed else "-" for ch in text)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:120] if slug else "run"


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"runs": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _maybe_delete_wandb_run(
    *, entity: str | None, project: str, run_id: str, enabled: bool
) -> None:
    if not enabled:
        return
    if entity is None:
        print(
            "[warn] --delete-interrupted-wandb requested but --wandb-entity is missing; skipping delete"
        )
        return
    try:
        import wandb  # type: ignore

        api: Any = wandb.Api()
        run: Any = api.run(f"{entity}/{project}/{run_id}")
        run.delete(delete_artifacts=False)
        print(f"[info] Deleted interrupted W&B run: {entity}/{project}/{run_id}")
    except Exception as exc:
        print(f"[warn] Could not delete W&B run {run_id}: {exc}")


def _build_key(config_stem: str, seed: int) -> str:
    return f"{config_stem}::seed{seed:02d}"


def main() -> None:
    args = parse_args()

    config_dir = Path(args.config_dir)
    config_paths = sorted(config_dir.glob(args.pattern))
    if args.limit_configs > 0:
        config_paths = config_paths[: args.limit_configs]

    if len(config_paths) == 0:
        print(f"No config files matched pattern '{args.pattern}' in {config_dir}")
        sys.exit(1)

    super_slug = _safe_slug(args.wandb_supergroup)
    checkpoint_root = Path(args.checkpoint_root)
    default_state = checkpoint_root / super_slug / "all_configs_multiseed_state.json"
    state_path = Path(args.state_file) if args.state_file else default_state
    state = _load_state(state_path)
    state.setdefault("runs", {})

    python_exe = Path(sys.executable)
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "online")

    total_pairs = len(config_paths) * len(args.seeds)

    print("=" * 88)
    print("OM-HGNN all-configs x all-seeds training")
    print(f"Config dir: {config_dir}")
    print(f"Matched configs: {len(config_paths)}")
    print(f"Seeds: {args.seeds}")
    print(f"Total pairs: {total_pairs}")
    print(f"W&B supergroup: {args.wandb_supergroup}")
    print(f"Split seed (fixed): {args.split_seed}")
    print(f"State file: {state_path}")
    print("=" * 88)

    start = time.perf_counter()
    attempted = 0
    skipped_completed = 0
    failures: List[Tuple[str, int, int]] = []

    pair_index = 0
    for config_path in config_paths:
        config_stem = config_path.stem
        config_group = f"{args.wandb_supergroup}__{config_stem}"
        config_slug = _safe_slug(config_stem)

        for seed in args.seeds:
            pair_index += 1
            key = _build_key(config_stem, seed)
            existing = state["runs"].get(key, {})

            if not args.reset_completed and existing.get("status") == "completed":
                skipped_completed += 1
                print(
                    f"[{pair_index}/{total_pairs}] {config_stem} seed={seed} already completed, skipping"
                )
                continue

            run_name = f"{config_stem}_seed{seed:02d}"
            run_id = _safe_slug(
                f"{args.wandb_supergroup}-{config_stem}-seed-{seed:02d}"
            )
            run_checkpoint_dir = checkpoint_root / super_slug / config_slug / run_name
            last_ckpt = run_checkpoint_dir / "last.ckpt"
            resume_ckpt = str(last_ckpt) if last_ckpt.exists() else None

            if existing.get("status") in {"failed", "interrupted"}:
                _maybe_delete_wandb_run(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    run_id=run_id,
                    enabled=args.delete_interrupted_wandb,
                )

            cmd: List[str] = [
                str(python_exe),
                "-m",
                "src.train",
                "--config-path",
                str(config_path),
                "--wandb-project",
                args.wandb_project,
                "--run-name",
                run_name,
                "--wandb-group",
                config_group,
                "--wandb-run-id",
                run_id,
                "--wandb-resume",
                "allow",
                "--seed",
                str(seed),
                "--split-seed",
                str(args.split_seed),
                "--checkpoint-dir",
                str(checkpoint_root / super_slug / config_slug),
            ]
            if resume_ckpt:
                cmd.extend(["--resume-from-checkpoint", resume_ckpt])
            if args.wandb_entity:
                cmd.extend(["--wandb-entity", args.wandb_entity])

            attempted += 1
            print(
                f"[{pair_index}/{total_pairs}] config={config_stem} seed={seed} "
                f"resume={'yes' if resume_ckpt else 'no'}"
            )

            state["runs"][key] = {
                "status": "running",
                "config_path": str(config_path),
                "config_stem": config_stem,
                "seed": seed,
                "run_name": run_name,
                "run_id": run_id,
                "wandb_group": config_group,
                "checkpoint_dir": str(run_checkpoint_dir),
                "last_checkpoint": str(last_ckpt),
                "updated_at": int(time.time()),
            }
            _save_state(state_path, state)

            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                failures.append((config_stem, seed, result.returncode))
                state["runs"][key]["status"] = "failed"
                state["runs"][key]["exit_code"] = result.returncode
                state["runs"][key]["updated_at"] = int(time.time())
                _save_state(state_path, state)
                if not args.continue_on_error:
                    break
            else:
                state["runs"][key]["status"] = "completed"
                state["runs"][key]["exit_code"] = 0
                state["runs"][key]["updated_at"] = int(time.time())
                _save_state(state_path, state)

        if failures and not args.continue_on_error:
            break

    elapsed = time.perf_counter() - start
    print("=" * 88)
    print(f"Finished in {elapsed / 60.0:.1f} min")
    print(f"Pairs requested: {total_pairs}")
    print(f"Pairs attempted this launch: {attempted}")
    print(f"Pairs skipped (already completed): {skipped_completed}")
    print(f"Failures: {len(failures)}")
    if failures:
        for config_stem, seed, code in failures:
            print(f"  - config={config_stem} seed={seed} exit={code}")
        sys.exit(2)


if __name__ == "__main__":
    main()
