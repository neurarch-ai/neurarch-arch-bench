# How to produce the RL training result (the #1 reviewer fix)

The self-assessment's top objection: the paper calls this an *RL environment* but
shows no policy *trained* with RL against it. This guide runs the shipped GRPO
loop end to end and gives you the exact numbers to drop into the paper. Even a
modest, honest lift converts "environment (proposed)" into "environment
(demonstrated)."

The whole thing fits a single free Colab T4 (default model is Qwen2.5-1.5B with
LoRA). Budget ~1-2 hours wall-clock for a short run.

## 0. Setup (once)

```bash
cd oss/neurarch-arch-bench
pip install "trl>=0.14" transformers datasets accelerate peft
node env-server.mjs        # reward server on http://localhost:8737 — leave running in one terminal
```

`env-server.mjs` is zero-dependency node; it serves `GET /tasks` (seeded splits)
and `POST /grade` (the same deterministic reward the paper uses).

## 1. Baseline pass@1 on a held-out split (no training)

In a second terminal:

```bash
python training/train_grpo.py --eval-only --seed 999 --count 64
```

Expected output (numbers illustrative; record YOURS):

```
model=Qwen/Qwen2.5-1.5B-Instruct
split: seed=999 count=64
pass@1: 7/64 = 0.109
parse failures: 12/64
mean reward: 0.31
```

A small instruct model is weak at emitting valid graph edits, so a low baseline
(single digits to ~20%) and some parse failures are expected and are the point:
there is headroom to learn.

## 2. Train with GRPO against the verifier

```bash
python training/train_grpo.py --steps 300 --count 512 --seed 123 --lora
# saves a LoRA adapter to out/grpo-arch/checkpoint-final
```

Watch the reward: TRL logs mean reward per step. The parse-floor reward (a small
negative below every server reward) means "emit valid JSON" is learned first, so
you should see parse failures fall early, then pass rate climb. Save the reward
curve (the TRL log / `trainer_state.json`) — it is Figure material.

## 3. Re-evaluate the trained policy on the SAME held-out split

```bash
python training/train_grpo.py --eval-only --seed 999 --count 64 \
    --model out/grpo-arch/checkpoint-final
```

Expected shape of the result (illustrative):

```
pass@1: 18/64 = 0.281      # up from 0.109
parse failures: 2/64       # down from 12
mean reward: 0.63          # up from 0.31
```

The held-out seed (999) differs from the training seed (123), so this is
generalization, not memorization. Report all three deltas.

## 4. What goes in the paper

- A short **RL results** subsection: baseline → post-training pass@1 on the
  held-out split, plus the parse-failure drop and mean-reward rise.
- A **reward-vs-step figure** from the TRL log.
- One honest sentence on scale (small model, LoRA, N steps) so the claim is
  "demonstrated at small scale," not oversold.

Paste your three numbers (baseline, post-training, and a few reward-curve points)
back and I will write the subsection and the figure.

## Notes

- No GPU locally? The repo's Colab notebook runs steps 0-3 on a free T4; open it,
  run all, copy the printed numbers.
- Bigger lift, more compute: raise `--steps` and `--count`; try `--num-generations 8`
  (GRPO group size) and a slightly higher `--lr`. Keep the eval seed fixed at 999.
- Everything is seeded and reproducible; the reward is the same verifier as the
  benchmark, so the RL result and the leaderboard are directly comparable.

---

## Companion: a second model for the amplification ablation (cheap)

The self-assessment's #2 fix — show the 82→100 lift is not Claude-specific — needs
one more model's per-round pass rates. Cheapest capable option via OpenRouter:

```bash
OPENROUTER_MODEL=deepseek/deepseek-chat \
  node amplify.mjs --providers=openrouter --generate=60 --seed=7 --turns=3 --out=amp-deepseek.json
```

This runs single-shot vs up-to-3 repair rounds on 60 tasks (~a few dollars).
Paste the printed summary (or the `amp-deepseek.json`) back and I will compute the
per-k cumulative pass (k=1/2/3) and add a second row to the ablation table, so it
reads "two models, the gap closes in one repair round for both."
