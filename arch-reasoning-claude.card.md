---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- reasoning
- neural-architecture-design
- verified
- rejection-sampling
- rlvr
pretty_name: Verified architecture-design reasoning traces (claude-sonnet-4-6)
size_categories:
- n<1K
---

# arch-reasoning-claude

**306 verified (spec -> reasoning -> design) traces for neural-architecture
design, minted by claude-sonnet-4-6 and filtered by a deterministic verifier.**

Each row pairs a natural-language architecture spec with the model's reasoning
and its final structured design (a batch of typed graph edits). Every row's
design was applied to the task's start graph and re-graded by the deterministic
verifier of [neurarch-arch-bench](https://github.com/neurarch-ai/neurarch-arch-bench):
no trace enters this set unless its design provably solves its spec. No human
judge, no LLM judge.

## Provenance and yield

- Model: `claude-sonnet-4-6` (Anthropic API), rejection-sampled.
- Tasks: 500 procedurally generated (generator v1, seed 20260708).
- Mint-time yield: 327/451 = 72.5% verified (49 API failures excluded).
- Released rows: **306**, each re-verified under the repository's current
  rubric (v2), which adds the linear-width consistency check of the paper's
  Algorithm 1. The 21 mint-time-verified rows that fail v2's stricter width
  check are excluded; they are exactly the blind-spot class the paper's
  grounding study documents.

## Schema

One JSON object per line: `task_id`, `spec`, `observation` (serialized start
graph), `reasoning`, `actions` (the verified edit batch), `verified: true`,
`source`, and a chat-format `messages` array ready for SFT.

## Reproduce / extend

```bash
git clone https://github.com/neurarch-ai/neurarch-arch-bench
ANTHROPIC_API_KEY=... node training/reasoning_traces.mjs --provider=claude --count=500 --delay=800 --out=my-traces
```

A private seed produces reasoning data whose designs never appeared on the
public web. Companion set: [arch-reasoning-grok](https://huggingface.co/datasets/neurarch-ai/arch-reasoning-grok).
