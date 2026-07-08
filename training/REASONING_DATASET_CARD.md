# Verified architecture-design reasoning traces

Produced by `training/reasoning_traces.mjs`. Each row is a
`(spec -> reasoning -> design)` triple where **the design's final graph is
re-graded by the same deterministic verifier the benchmark uses, and only
passing traces are kept**. No LLM judge, anywhere. This is verified reasoning
data of the shape RLVR / reasoning-model post-training consumes, for a domain
(neural architecture design) no code-only corpus contains.

## Why this exists

The scarce input for training reasoning models on verifiable rewards is
verified reasoning: problem -> chain of thought -> checked answer. Code and math
have it (compilers, unit tests, proof checkers). Architecture design did not,
because "is this network sound and within budget" is only a pure function when
the design is a typed, structured graph. This dataset is that function applied
at scale to model reasoning.

## Schema (one JSON object per line)

| field | meaning |
| --- | --- |
| `task_id` | generator task id (family + index) |
| `spec` | natural-language design spec |
| `observation` | serialized starting graph (input->output stub, or a broken graph for repair tasks) |
| `reasoning` | step-by-step reasoning over the constraints (required layers, shapes, divisibility, budgets) |
| `actions` | the structured edits that produce the design |
| `verified` | always `true` — the resulting graph was re-graded and passed |
| `source` | `reference-derived` (keyless) or `<provider>:verified` (rejection-sampled) |
| `messages` | chat-format `{system, user, assistant}`, reasoning inside `<reasoning>...</reasoning>` |

## Two modes

- **`<provider>:verified` (premium).** A model is asked to reason then act; we
  apply its actions and grade them; only designs that PASS are written
  (rejection sampling). The `keptRate` in `<out>.stats.json` is the model's
  verified-solve rate. This is the real product: model-generated reasoning that
  the verifier certifies.
- **`reference-derived` (keyless).** The reasoning is composed from the task's
  own constraints and its reference design (passing by construction). A
  structural scaffold and smoke test; no API key.

## Reproduce

```bash
# keyless scaffold (no key)
node reasoning_traces.mjs --count=500 --seed=20260708 --out=arch-reasoning

# premium, rejection-sampled, model-generated reasoning (needs a key)
ANTHROPIC_API_KEY=... node reasoning_traces.mjs --provider=claude --count=500 --tries=2 --out=arch-reasoning-claude
```

## Provenance and safety

Rows carry structural specs, graphs, reasoning, and structured edits only. No
weights, no training data, no PII. Every design is verified deterministically,
so contamination is bounded by construction: a private seed yields traces whose
designs never existed on the public web. Licensing of larger, model-specific
trace sets is covered in `../LABS.md`.
