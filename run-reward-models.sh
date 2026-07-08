#!/usr/bin/env bash
# run-reward-models.sh — sweep reward_anchor.mjs over many models, one row each.
#
# Set the API keys you have as env vars (leave the rest unset to skip them),
# then run:  ./run-reward-models.sh
# Results stream to the console AND append to reward-anchor-results.txt.
#
# Tunables (env):  COUNT (default 30 tasks -> 60 examples), DELAY ms between
# calls (default 800; raise for rate-limited providers).
#
# Get keys:
#   ANTHROPIC_API_KEY  console.anthropic.com/settings/keys
#   XAI_API_KEY        console.x.ai
#   OPENAI_API_KEY     platform.openai.com/api-keys
#   OPENROUTER_API_KEY openrouter.ai/keys   (one key -> many open models)
#   DEEPSEEK_API_KEY   platform.deepseek.com/api_keys
#   GROQ_API_KEY       console.groq.com/keys   (free tier)
set -uo pipefail
COUNT=${COUNT:-30}
DELAY=${DELAY:-800}
OUT=reward-anchor-results.txt
: > "$OUT"

run() {  # run <provider> <MODEL_ENV=value> <label>
  echo "=== $3 ===" | tee -a "$OUT"
  env "$2" node reward_anchor.mjs --provider="$1" --count="$COUNT" --delay="$DELAY" 2>&1 | tee -a "$OUT"
  echo "" | tee -a "$OUT"
}

# --- closed frontier (one key each) ---
[ -n "${ANTHROPIC_API_KEY:-}" ] && run claude "ANTHROPIC_MODEL=claude-sonnet-4-6" "claude-sonnet-4-6 (frontier)"
[ -n "${XAI_API_KEY:-}" ]       && run grok   "XAI_MODEL=grok-4"                  "grok-4 (frontier)"
[ -n "${OPENAI_API_KEY:-}" ]    && run openai "OPENAI_MODEL=gpt-4o"               "gpt-4o (frontier)"

# --- open weights via OpenRouter (one key, several models) ---
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  for M in qwen/qwen-2.5-72b-instruct mistralai/mistral-large meta-llama/llama-3.3-70b-instruct google/gemma-2-27b-it; do
    run openrouter "OPENROUTER_MODEL=$M" "$M (open)"
  done
fi

# --- cheap / very cheap (deepseek-chat is fast; deepseek-reasoner/R1 is slow) ---
[ -n "${DEEPSEEK_API_KEY:-}" ] && run deepseek "DEEPSEEK_MODEL=deepseek-chat"        "deepseek-chat (open)"
[ -n "${GROQ_API_KEY:-}" ]     && run groq     "GROQ_MODEL=llama-3.3-70b-versatile" "llama-3.3-70b (groq, free)"

echo "Done. Every row is also in $OUT — paste it back to fill the paper table."
