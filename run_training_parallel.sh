#!/bin/bash
# Four parallel internal-only training streams on one GPU (MI/MS x {H150,H1}):
#   A: Go2 mi_150 + mi_nohist     C: Go2 ms_nohist
#   B: Go2 ms_150                 D: G1 ms_150 + ms_nohist
# Each stream has its own state file to avoid races; checkpoints stay under the
# shared supergroup so the evaluation scripts find them.
# Env overrides: SEEDS (default "1 2"), WANDB_PROJECT (default om-hgnn-v4).
set -u

GNN="/home/ryz5920/Github Project/aut_om_hgnn"
GNN_PY="$GNN/.venv/bin/python"
STATE="$GNN/checkpoints/all_configs_seed_sweep"
SEEDS="${SEEDS:-1 2}"
PROJECT="${WANDB_PROJECT:-om-hgnn-v4}"

mkdir -p "$STATE"

# shellcheck disable=SC2086
"$GNN_PY" -m src.train_all_configs_multiseed \
    --config-dir /tmp/opencode/go2_a_yaml \
    --pattern '*.yaml' \
    --seeds $SEEDS \
    --wandb-project "$PROJECT" \
    --wandb-supergroup all_configs_seed_sweep \
    --state-file "$STATE/state_go2_a.json" \
    --continue-on-error \
    > /tmp/opencode/stream_go2a.log 2>&1 &
PID_A=$!

# shellcheck disable=SC2086
"$GNN_PY" -m src.train_all_configs_multiseed \
    --config-dir /tmp/opencode/go2_b_yaml \
    --pattern '*.yaml' \
    --seeds $SEEDS \
    --wandb-project "$PROJECT" \
    --wandb-supergroup all_configs_seed_sweep \
    --state-file "$STATE/state_go2_b.json" \
    --continue-on-error \
    > /tmp/opencode/stream_go2b.log 2>&1 &
PID_B=$!

# shellcheck disable=SC2086
"$GNN_PY" -m src.train_all_configs_multiseed \
    --config-dir /tmp/opencode/go2_c_yaml \
    --pattern '*.yaml' \
    --seeds $SEEDS \
    --wandb-project "$PROJECT" \
    --wandb-supergroup all_configs_seed_sweep \
    --state-file "$STATE/state_go2_c.json" \
    --continue-on-error \
    > /tmp/opencode/stream_go2c.log 2>&1 &
PID_C=$!

# shellcheck disable=SC2086
"$GNN_PY" -m src.train_all_configs_multiseed \
    --config-dir /tmp/opencode/g1_valid_yaml \
    --pattern '*.yaml' \
    --seeds $SEEDS \
    --wandb-project "$PROJECT" \
    --wandb-supergroup all_configs_seed_sweep \
    --state-file "$STATE/state_g1.json" \
    --continue-on-error \
    > /tmp/opencode/stream_g1.log 2>&1 &
PID_D=$!

echo "[parallel] streams started: go2a=$PID_A go2b=$PID_B go2c=$PID_C g1=$PID_D"
for pid in "$PID_A" "$PID_B" "$PID_C" "$PID_D"; do
    wait "$pid"
    echo "[parallel] stream pid=$pid exit=$?"
done
echo "[parallel] all streams finished"
