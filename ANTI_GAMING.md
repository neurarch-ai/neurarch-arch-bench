# Anti-gaming: how this benchmark resists lazy and degenerate policies

Reward-hacking resistance is the first question anyone training against a
verifiable environment should ask. This document lists the degenerate
strategies we have tried against this repo's grader, the specific constraint
that blocks each one, and the test that pins that defense. It ends with the
blind spots we know about and keep on purpose, with measurements, because a
transparent rubric plus an honest gap report beats an opaque rubric claiming
completeness.

Everything below refers to code in this repository only (`bench.mjs`,
`generate.mjs`, `tasks.json`). No LLM judge sits anywhere in the grading path,
so there is nothing to persuade: a submission passes on structure or it fails.

## The verifier is bracketed from both sides

- **Upper bound**: every task (curated and generated) ships a reference
  solution graded by the same `gradeTask`. Tests assert every reference passes
  (`generate.test.ts` runs 500 generated cases), so the rubric cannot drift
  into rejecting legitimate solutions.
- **Lower bound**: the `degenerate solutions must fail` suite in
  `bench.test.ts` asserts that empty plans, disconnected showpieces,
  over-budget graphs, and forbidden-action rebuilds fail with the expected
  finding, so the rubric cannot drift into rubber-stamping lazy policies.
- **No vacuous tasks**: every edit-in-place start graph is asserted to FAIL
  its own task before the fix, so "do nothing" is never a solution.

## Strategy → defense → pinned by

| # | Gaming strategy | Defense | Pinned by |
|---|---|---|---|
| 1 | Emit zero actions | Design-from-spec starts are bare stubs failing `minComponents` and `mustContainTypes`; repair starts carry a real blocker | `bench.test.ts` degenerate suite; `generate.test.ts` starts-fail-untouched |
| 2 | Rebuild the graph wholesale on a "surgical repair" task | `forbidActionTypes` rejects `replace_model` / `clear_canvas` on generated edit-in-place families regardless of the resulting graph's quality; `maxActions` caps edit economy | `generate.test.ts` repair-rejects-replace_model; degenerate suite |
| 3 | Shrink everything to nothing under a param budget | Budgets are bands: `minParams` (the floor) exists exactly so the trivial minimum fails; the param-grow family has a floor AND a ceiling | generated param-grow references + under-band failure category |
| 4 | Place required layer types as floating nodes, wire nothing | `mustReachOutput` reachability check plus the disconnected-graph structural blocker | `bench.test.ts` disconnected-showpiece case |
| 5 | Attention params that fit the budget but violate divisibility | `embedDim % numHeads` and `numHeads % numKVHeads` are hard blockers under `forbidBlockers` | `bench.test.ts` verifier suite |
| 6 | Blow the budget with a structurally perfect graph | `maxParams` fails independently of score | degenerate suite over-budget case |
| 7 | Memorize the public task set | The procedural generator mints held-out splits from a seed (`--generate=N --seed=S`); fresh tasks never appear on the public web and randomize dims, depths, and budgets per instance | `generate.test.ts` determinism + reference-satisfiability |
| 8 | Farm the dense score without solving the task | `minScore` is one constraint among many; types, reachability, budgets, and action rules all bind independently | grader structure (`gradeTask`) |

## Known blind spots (kept deliberately, and measured)

This rubric is optimized for transparency and zero dependencies: the whole
grading path is one readable file. That costs coverage, and we measured the
cost instead of hiding it. From the grounding study
([training/README.md](training/README.md), 264 graphs, torch 2.8):

| Verifier verdict | n | constructs | forward ok | trains |
| --- | --- | --- | --- | --- |
| PASS (clean) | 80 | 100% | 100% | 90% |
| BLOCKED | 96 | 75% | 0% | 0% |
| not blocked, corrupted | 88 | 100% | 0% | 0% |

1. **Linear in/out width mismatches pass this rubric and crash PyTorch**
   (the third row: 88 corrupted graphs, 100% construct, 0% forward). The
   richer verifier inside the Neurarch product flags these as shape issues;
   this repo keeps the simple rubric and documents the gap. If you train
   against this environment, treat "passes the rubric" as "structurally
   valid", not "torch-runnable".
2. **The 0..100 score is a validity margin, not a quality ranking.** Among
   clean graphs, score does not correlate positively with training progress
   (measured negative rank correlation in the grounding set). Use `pass` as
   the reward signal; do not optimize score beyond the `minScore` gate.
3. **No serving-physics constraints.** `kvBytesPerToken` ships as a metric
   (used by the MCP server) but no task in this repo grades on KV or latency
   budgets. The product benchmark adds those constraint families.

## Reproduce

```bash
npx vitest run                 # brackets: references pass, degenerates fail
node leaderboard.mjs --providers=reference   # oracle pass rate: 100%, no API key
python training/grounding.py   # the blind-spot measurement itself
```
