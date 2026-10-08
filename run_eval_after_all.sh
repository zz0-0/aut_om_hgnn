#!/bin/bash
# Wait for the training streams, then evaluate every internal config x seed
# (internal-controller protocol), aggregate, and run the sim-online validation.
set -u

GNN="/home/ryz5920/Github Project/aut_om_hgnn"
GNN_PY="$GNN/.venv/bin/python"

while pgrep -f "run_training_parallel.sh" >/dev/null 2>&1; do
    echo "[eval-wait] waiting for training streams..."
    sleep 300
done

cd "$GNN" || exit 1
echo "[eval-wait] running evaluation for seeds 0..9 (2 shards in parallel)..."
"$GNN_PY" -m src.evaluate_all --seeds 0 1 2 3 4 5 6 7 8 9 --skip-blind --num-shards 2 --shard-index 0 \
    > /tmp/opencode/eval_shard0.log 2>&1 &
SHARD0=$!
"$GNN_PY" -m src.evaluate_all --seeds 0 1 2 3 4 5 6 7 8 9 --skip-blind --num-shards 2 --shard-index 1 \
    > /tmp/opencode/eval_shard1.log 2>&1 &
SHARD1=$!
wait "$SHARD0"
wait "$SHARD1"

echo "[eval-wait] aggregating metrics..."
"$GNN_PY" -m src.evaluate_all --seeds 0 1 2 3 4 5 6 7 8 9 --skip-blind --aggregate-only

echo "[eval-wait] running sim-online validation..."
bash run_online_validation.sh

echo "[eval-wait] COMPLETE"
