# neurarch-arch-design-v1

Design neural-network architectures as structured graph edits, graded by a
**deterministic verifier** — no human judge, no LLM judge, no GPU. Tasks are
procedurally generated from a seed, so train/eval/held-out splits are just
different seeds and a held-out split never appears on the public web.

This is the **verifiers v1-native** taskset (`verifiers>=0.2`, taskset/harness
split, July 2026 API). The legacy v0 package (`../neurarch_arch_design`) is
frozen.

## Run it

```bash
# 1. Start the zero-dependency verifier server (node >= 18) from the repo root:
node env-server.mjs

# 2. Evaluate with the tool-less null harness (the task is text-in / JSON-out):
uv run eval --taskset.id neurarch-arch-design-v1 --harness.id null \
    --taskset.count 60 --taskset.seed 999 -n 60
```

Config knobs: `--taskset.count` (split size), `--taskset.seed` (split
identity), `--taskset.task.env-url` (verifier server, default
`http://localhost:8737`).

## Rewards and metrics

| Name | Kind | Meaning |
|---|---|---|
| `arch_reward` | reward (w=1.0) | Server's shaped reward: pass ~1.0..1.5, dense partial credit for valid-but-failing graphs, -0.5 for unparseable plans |
| `task_pass` | metric | Raw pass rate, undiluted by shaping |
| `parse_ok` | metric | Share of completions that parsed as a JSON action plan |

## Why this environment

- **Verifiable reward, zero judge cost**: grading is microseconds of CPU.
- **Anti-gaming**: per-task forbidden-action constraints, structural success
  predicates, and satisfiability-proven references (see `../../ANTI_GAMING.md`
  and `../../ROBUSTNESS.md`).
- **Measured**: a frontier model goes 79% -> 100% with the verifier in the
  loop; SFT on environment-minted pairs lifts a 1.5B model 14% -> 69% on
  strictly-unseen tasks (see `../../RESULTS.md` and the tech report).

## Smoke test (no verifiers install needed)

```bash
node env-server.mjs &   # from the repo root
python3 environments/neurarch_arch_design_v1/test_v1_smoke.py
```

Stubs the `verifiers.v1` surface and exercises load + grading end-to-end
against the live server.
