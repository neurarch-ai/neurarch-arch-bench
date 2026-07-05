# Training: GRPO on a verifiable architecture-design reward

This folder turns the benchmark into an RL training environment. A policy
reads a spec plus a serialized start graph, emits graph-edit actions, and gets
a reward from the same deterministic verifier the leaderboard uses. No human
labels, no LLM judge, no GPU on the reward side: grading is a sub-millisecond
pure function behind a zero-dependency HTTP server.

## Pieces

| File | Role |
| --- | --- |
| `../env-server.mjs` | HTTP reward service: `GET /tasks` (seeded splits), `POST /grade` (shaped reward). Plain node >= 18, zero deps. |
| `../generate.mjs` | Deterministic task generator, 8 families (design-from-spec and edit-in-place), unlimited scale. Same (count, seed) always yields the same split. |
| `train_grpo.py` | TRL GRPOTrainer against the server, plus an `--eval-only` mode for pass@1 on any split. |
| `colab_grpo.ipynb` | The whole loop on a free Colab T4: baseline eval, train, reward curve, held-out eval. |

## Quick start (local GPU box)

```bash
# terminal 1: the reward server
node ../env-server.mjs

# terminal 2: baseline, train, re-eval
pip install "trl>=0.14" transformers datasets accelerate peft
python train_grpo.py --eval-only --seed 999 --count 64          # before
python train_grpo.py --steps 300 --count 512 --seed 123 --lora --bf16
python train_grpo.py --eval-only --seed 999 --count 64 \
    --model out/grpo-arch/checkpoint-final                      # after
```

The train split (seed 123) and eval split (seed 999) are disjoint by
construction, and neither ships in the repo: they are minted on demand by the
seeded generator, so the eval split is contamination-resistant.

## Reward

Computed server-side (see `env-server.mjs` header for the exact formula):
passing the task's constraints is worth ~1.0, a valid-but-failing graph earns
a dense partial signal from the 0..100 health score, malformed edits cost a
little, and a completion that is not valid JSON gets a flat -0.5 in the
trainer, below every server reward. So the learning order is typically:
valid JSON first, then valid graphs, then constraint satisfaction.

## Grounding study: does the verdict track reality?

`dump_grounding_set.mjs` emits clean reference architectures plus systematically
corrupted variants (broken attention divisibility, linear width mismatch,
severed mid-graph connection); `grounding.py` builds every graph as an actual
PyTorch model, trains it briefly on a synthetic fixed-teacher task, and
cross-tabulates the verifier's verdict against physical outcomes.

```bash
node dump_grounding_set.mjs --count=20 --seed=123 --out=grounding_set.jsonl
python grounding.py --set grounding_set.jsonl --steps 60   # CPU, ~2 min
```

Latest run (two seeds merged: 123 + 777, count=40 each, 264 graphs, torch
2.8, 2026-07-04; reproduce with `--set gset-123.jsonl gset-777.jsonl`):

| Verifier verdict | n | constructs | forward ok | trains |
| --- | --- | --- | --- | --- |
| PASS (clean) | 80 | 100% | 100% | 90% |
| BLOCKED | 96 | 75% | 0% | 0% |
| not blocked, corrupted | 88 | 100% | 0% | 0% |

Read it honestly: a blocker predicts runtime failure perfectly at this scale
(96/96 forward failures), and clean graphs overwhelmingly construct and make
training progress. The third row is the transparent rubric's blind spot on
purpose: linear in/out mismatches pass this repo's simple rubric but crash
PyTorch (the richer verifier in the Neurarch product does flag them as shape
issues). "Trains" means the loss fell by at least 20% in 60 steps; it is a
trainability probe, not a claim about final model quality.

**Score magnitude is NOT a quality ranking.** Among the 80 clean graphs,
Spearman rho between the 0..100 score and relative loss decrease was -0.581:
higher-scoring (deeper, more structured) models make slower per-step progress
on the short synthetic probe than tiny MLPs. So the grounded claim is the
pass/blocked boundary; treat the score as a validity margin. Ranking design
QUALITY needs real training outcomes, which is exactly the open roadmap item.

## What this is and is not

- It IS a demonstration that the verifier's reward is learnable and that the
  environment plugs into a standard RL stack (TRL GRPO) with ~200 lines of
  glue.
- It is NOT a claim that the resulting policy designs architectures that
  train to good accuracy. The verifier checks structural validity, budgets,
  connectivity, and required components; grounding the score against real
  training runs is tracked separately in the benchmark's POSITIONING notes.

## Hardware notes

- Default model is `Qwen/Qwen2.5-1.5B-Instruct`. With `--lora` it fits a
  Colab free-tier T4. Any HF causal LM with a chat template works via
  `--model`.
- The reward server adds no GPU load; grading 8 generations per step is
  network-bound at localhost latency (microseconds of compute per grade).
- GRPO group size (`--num-generations`, default 8) is the main memory knob
  after model size.
