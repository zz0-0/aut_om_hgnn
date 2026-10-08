#!/bin/bash
# Sim-online validation for the headline configs (MI/MS, H=150) with the internal policy.
# Waits for the full offline schedule to finish, then runs each case at 50 Hz and writes JSON.
set -u

GNN="/home/ryz5920/Github Project/aut_om_hgnn"
IL="/home/ryz5920/Github Project/aut_unitree_g1_isaaclab"
IL_PY="$IL/.venv/bin/python"
GNN_PY="$GNN/.venv/bin/python"
OUT="$GNN/evaluation/online"
STEPS=1500
SEED=0
mkdir -p "$OUT"

G1_TASK="G1-Rough-Locomotion-Dataset-v0"
GO2_TASK="Go2-Rough-Locomotion-Dataset-v0"
G1_CKPT="/home/ryz5920/Project/archive/aut_unitree_isaaclab_v1/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/g1_rough/2026-01-15_12-00-00/model_2999.pt"
GO2_CKPT="/home/ryz5920/Project/archive/aut_unitree_isaaclab_v1/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/go2_rough/2026-01-15_08-44-46/model_2999.pt"
FRICTION_ARGS=(--joint_friction_mu_range 0.05 0.30 --joint_friction_viscous_range 0.01 0.05)

resolve_checkpoint() {
    "$GNN_PY" -c "
import sys
from pathlib import Path
from src.evaluate_all import best_checkpoint
stem = sys.argv[1]
seed = int(sys.argv[2])
run_dir = Path('checkpoints/all_configs_seed_sweep') / stem / f'{stem}_seed{seed:02d}'
checkpoint = best_checkpoint(run_dir)
print(checkpoint.resolve() if checkpoint is not None else '')
" "$1" "$2"
}

run_case() {
    local stem="$1"
    local task="$2"
    local policy_tag="$3"
    shift 3
    local checkpoint
    checkpoint="$(cd "$GNN" && resolve_checkpoint "$stem" "$SEED")"
    if [ -z "$checkpoint" ]; then
        echo "[online] missing checkpoint for $stem; skipping"
        return
    fi
    echo "[online] $stem with $policy_tag"
    cd "$IL" || exit 1
    OMNI_KIT_ACCEPT_EULA=YES timeout 1800 "$IL_PY" \
        scripts/reinforcement_learning/rsl_rl/eval_online_estimation.py \
        --task "$task" --num_envs 1 --steps "$STEPS" \
        "$@" \
        --gnn_config_path "$GNN/src/config/yaml/${stem}.yaml" \
        --gnn_checkpoint_path "$checkpoint" \
        --gnn_repo "$GNN" \
        --output "$OUT/${stem}__${policy_tag}.json" \
        "${FRICTION_ARGS[@]}" --headless > "$OUT/${stem}__${policy_tag}.log" 2>&1
}

echo "[online] starting online validation"

for stem in g1_bhmg_multi_mi_150 g1_bhmg_multi_ms_150; do
    run_case "$stem" "$G1_TASK" internal --policy_checkpoint "$G1_CKPT" --external_action_scale 0.25
done

for stem in go2_qhmg_multi_mi_150 go2_qhmg_multi_ms_150; do
    run_case "$stem" "$GO2_TASK" internal --policy_checkpoint "$GO2_CKPT" --external_action_scale 0.25
done

echo "[online] COMPLETE"
