#!/usr/bin/env bash
# run_grpo_demo.sh — one command: baseline pass@1 -> GRPO train -> re-eval, with
# a clean BEFORE/AFTER summary for the paper. Local GPU path; for a free T4 use
# training/colab_grpo.ipynb (Run All) instead.
#
#   cd oss/neurarch-arch-bench && ./training/run_grpo_demo.sh
#
# Tunables (env): STEPS (300), TRAIN_COUNT (512), EVAL_COUNT (64),
# TRAIN_SEED (123), EVAL_SEED (999), MODEL (Qwen/Qwen2.5-1.5B-Instruct).
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-300}; TRAIN_COUNT=${TRAIN_COUNT:-512}; EVAL_COUNT=${EVAL_COUNT:-64}
TRAIN_SEED=${TRAIN_SEED:-123}; EVAL_SEED=${EVAL_SEED:-999}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
OUT=out/grpo-arch; LOG=out/grpo-demo.log; mkdir -p out

echo "Starting reward server..."; node env-server.mjs >out/env-server.log 2>&1 &
ENV_PID=$!; trap 'kill $ENV_PID 2>/dev/null' EXIT
sleep 2

echo "== 1/3 baseline pass@1 (held-out seed $EVAL_SEED) =="
python training/train_grpo.py --eval-only --seed "$EVAL_SEED" --count "$EVAL_COUNT" --model "$MODEL" | tee out/eval_before.txt

echo "== 2/3 GRPO training (seed $TRAIN_SEED, $STEPS steps) =="
python training/train_grpo.py --steps "$STEPS" --count "$TRAIN_COUNT" --seed "$TRAIN_SEED" --lora --model "$MODEL" --out "$OUT" 2>&1 | tee "$LOG"

echo "== 3/3 re-eval trained policy (same held-out seed $EVAL_SEED) =="
python training/train_grpo.py --eval-only --seed "$EVAL_SEED" --count "$EVAL_COUNT" --model "$OUT/checkpoint-final" | tee out/eval_after.txt

echo ""
echo "================ BEFORE / AFTER (paste this back) ================"
echo "-- before --"; grep -E "pass@1|parse failures|mean reward" out/eval_before.txt
echo "-- after  --"; grep -E "pass@1|parse failures|mean reward" out/eval_after.txt
echo "reward curve: $LOG  (grep 'reward' for per-step means)"
echo "================================================================="
