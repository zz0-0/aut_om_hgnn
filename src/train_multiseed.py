"""Run multi-seed OM-HGNN training with grouped Weights & Biases logging.

This launcher keeps configuration/hyperparameters fixed and only changes seed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run src.train across multiple seeds")
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="W&B group; defaults to '<config_stem>_seed_sweep'",
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
        help="Fixed train/val split seed for fair across-seed comparison",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Optional run name prefix; defaults to config stem",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Optional JSON state path; defaults to checkpoints/<group>/multiseed_state.json",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="checkpoints",
        help="Root checkpoint directory used by src.train",
    )
    parser.add_argument(
        "--reset-completed",
        action="store_true",
        help="Ignore previous state and rerun all seeds",
    )
    parser.add_argument(
        "--delete-interrupted-wandb",
        action="store_true",
        help="Best-effort delete interrupted runs in W&B before retrying that seed",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining seeds when one seed fails",
    )
    return parser.parse_args()


def _safe_slug(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    slug = "".join(ch if ch in allowed else "-" for ch in text)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:120] if slug else "run"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: Path, state: dict[str, Any]) -> None:
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

        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        run.delete(delete_artifacts=False)
        print(f"[info] Deleted interrupted W&B run: {entity}/{project}/{run_id}")
    except Exception as exc:
        print(f"[warn] Could not delete W&B run {run_id}: {exc}")


def main() -> None:
    args = parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    run_prefix = args.run_prefix or config_path.stem
    wandb_group = args.wandb_group or f"{config_path.stem}_seed_sweep"
    group_slug = _safe_slug(wandb_group)
    checkpoint_root = Path(args.checkpoint_root)
    default_state = checkpoint_root / group_slug / "multiseed_state.json"
    state_path = Path(args.state_file) if args.state_file else default_state
    state = _load_state(state_path)
    state.setdefault("runs", {})

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "online")

    python_exe = Path(sys.executable)

    print("=" * 72)
    print("OM-HGNN multi-seed training")
    print(f"Config: {config_path}")
    print(f"Seeds: {args.seeds}")
    print(f"W&B group: {wandb_group}")
    print(f"Split seed (fixed): {args.split_seed}")
    print(f"State file: {state_path}")
    print("=" * 72)

    start = time.perf_counter()
    failures: list[tuple[int, int]] = []
    attempted = 0
    skipped_completed = 0

    for idx, seed in enumerate(args.seeds, start=1):
        seed_key = str(seed)
        existing = state["runs"].get(seed_key, {})
        if not args.reset_completed and existing.get("status") == "completed":
            skipped_completed += 1
            print(f"[{idx}/{len(args.seeds)}] seed={seed} already completed, skipping")
            continue

        run_name = f"{run_prefix}_seed{seed:02d}"
        run_id = _safe_slug(f"{wandb_group}-seed-{seed:02d}")
        run_checkpoint_dir = checkpoint_root / group_slug / run_name
        last_ckpt = run_checkpoint_dir / "last.ckpt"
        resume_ckpt = str(last_ckpt) if last_ckpt.exists() else None

        # If this seed had an interrupted status and user requests cleanup,
        # delete stale W&B run first; next launch recreates/resumes cleanly.
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
            wandb_group,
            "--wandb-run-id",
            run_id,
            "--wandb-resume",
            "allow",
            "--seed",
            str(seed),
            "--split-seed",
            str(args.split_seed),
            "--checkpoint-dir",
            str(checkpoint_root / group_slug),
        ]
        if resume_ckpt:
            cmd.extend(["--resume-from-checkpoint", resume_ckpt])
        if args.wandb_entity:
            cmd.extend(["--wandb-entity", args.wandb_entity])

        attempted += 1
        print(
            f"[{idx}/{len(args.seeds)}] seed={seed} run={run_name} "
            f"resume={'yes' if resume_ckpt else 'no'}"
        )

        state["runs"][seed_key] = {
            "status": "running",
            "run_name": run_name,
            "run_id": run_id,
            "checkpoint_dir": str(run_checkpoint_dir),
            "last_checkpoint": str(last_ckpt),
            "updated_at": int(time.time()),
        }
        _save_state(state_path, state)

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            failures.append((seed, result.returncode))
            state["runs"][seed_key]["status"] = "failed"
            state["runs"][seed_key]["exit_code"] = result.returncode
            state["runs"][seed_key]["updated_at"] = int(time.time())
            _save_state(state_path, state)
            if not args.continue_on_error:
                break
        else:
            state["runs"][seed_key]["status"] = "completed"
            state["runs"][seed_key]["exit_code"] = 0
            state["runs"][seed_key]["updated_at"] = int(time.time())
            _save_state(state_path, state)

    elapsed = time.perf_counter() - start
    print("=" * 72)
    print(f"Finished in {elapsed / 60.0:.1f} min")
    print(f"Seeds requested: {len(args.seeds)}")
    print(f"Seeds attempted this launch: {attempted}")
    print(f"Seeds skipped (already completed): {skipped_completed}")
    print(f"Failures: {len(failures)}")
    if failures:
        for seed, code in failures:
            print(f"  - seed={seed} exit={code}")
        sys.exit(2)


if __name__ == "__main__":
    main()
