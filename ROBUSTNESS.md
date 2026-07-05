# Anti-gaming report

Reward-hacking resistance is the quality bar frontier labs cite first when
they evaluate RL environments. This document is the evidence artifact for THIS
repo's grader: every degenerate strategy we have tried against it, the defense
that blocks it, and the pinned test that fails if anyone weakens that defense.
It ends with the residual risks we know about and have not closed, because a
red-team report that claims zero gaps is itself a red flag.

Scope: the transparent grader in [`bench.mjs`](./bench.mjs), the 10-family
generator in [`generate.mjs`](./generate.mjs), the reward shaping in
[`env-server.mjs`](./env-server.mjs), and the LLM task proposer in
[`propose.mjs`](./propose.mjs). The Neurarch product runs a richer verifier;
results in this repo are defined solely by the rubric here, and the one place
that difference matters is listed under residual risks, with measurements.

## How the grader is bracketed

Two suites pin it from both sides, so it can drift neither into a rubber
stamp nor into an impossible bar:

- **Upper bound**: every curated task has a reference solution
  (`solutions.json`) and every generated task carries its own; tests assert
  each passes its task (`bench.test.ts`, `generate.test.ts`, including a
  500-case sweep). A grader change that rejects legitimate solutions fails CI.
- **Lower bound**: degenerate graphs that MUST fail, each pinned to its
  specific failure (`bench.test.ts`), and every edit-in-place start graph is
  asserted to fail its own task untouched (no vacuous tasks).

Grading is a pure function: no LLM judge anywhere, so there is no persuasion
surface, and no hidden state, so there is nothing to probe across calls.

## Gaming strategies and their defenses

### 1. Do nothing (the empty plan)

Design-from-spec tasks start from a bare input->output stub that fails
`minComponents` and `mustContainTypes`; edit-in-place tasks start from graphs
broken by construction. Pinned by the "start graphs fail their own task
before the fix" test and the calibration harness's `--policy=noop` self-test
(must pass ~0%).

### 2. Wholesale rebuild masquerading as a surgical fix

`forbidActionTypes` on every edit-in-place family rejects a plan that used
`replace_model` / `clear_canvas` regardless of how good the resulting graph
is; `maxActions` caps repairs so brute-force rebuilds via many small actions
also fail. Pinned by the "repair tasks reject replace_model solutions" test.

### 3. Shrink to nothing under a budget

Budgets are bands, not ceilings: `minParams` (floor) plus `minComponents`
plus `mustContainTypes` mean a trimmed graph must still be a real network of
the required shape, and the grow family inverts the trap with a floor AND a
ceiling. Pinned by the generated-reference sweep exercising both sides of the
band.

### 4. Disconnected showpiece graphs

Adding every required layer type as floating nodes passes the type checklist
but fails `mustReachOutput` (input-to-output reachability) and the
disconnected-graph blocker in `findBlockers`.

### 5. Divisibility traps

`embedDim % numHeads != 0` and `numHeads % numKVHeads != 0` are hard blockers
in `findBlockers`; the repair family is built around exactly these defects,
and its start graphs are asserted broken.

### 6. Reward farming against the server

The env server's shaped reward has no progress term (it is a pure function of
the current graph), so break-then-fix oscillation has nothing to farm; each
malformed or unapplicable action costs `applyError` penalty, so action spam is
negative, not free; unparseable completions are scored below every server
reward by the trainers. Multi-turn episodes cap at `maxTurns` and end on pass.

### 7. Memorizing the public task set

Splits are minted on demand from a seed, so evaluation can always run on
tasks that have never existed anywhere public; families randomize dims,
depths, budgets, and head counts per instance. Same seed = identical split,
so private evaluation is still reproducible.

### 8. Poisoning via LLM-authored tasks

`propose.mjs` accepts nothing without a satisfiability proof: the proposal's
own reference must pass the grader, and safety-net constraints
(`forbidBlockers`, `minScore`, `mustReachOutput`) are always enforced on top
of whatever the proposer wrote, so a malicious or sloppy proposal cannot mint
a rubber-stamp task or an impossible one.

### 9. Gaming the score rubric instead of the task

The 0..100 rubric is deliberately transparent (one readable function), so it
is the most gameable component by design; that is why `minScore` is only one
of several independent constraints and every task also carries structural
requirements the rubric does not control.

## Difficulty calibration

An ungameable environment is still useless at ~0% pass (no learning signal)
or ~100% (nothing to learn). Labs cite a floor of roughly 2-3% pass rate on
the hard end (one success per 64-128 rollouts). `calibrate.mjs` measures
per-family pass rates with Wilson 95% intervals and flags each family's band:

```bash
node calibrate.mjs --policy=reference                 # self-test: must be 100%
node calibrate.mjs --policy=noop                      # self-test: must be ~0%
XAI_API_KEY=... node calibrate.mjs --provider=grok --per-family=16
```

The two keyless policies bracket the harness itself: if `reference` ever
fails or `noop` ever passes, the harness (not the model) is broken.

## Residual risks (open, tracked, not hidden)

1. **Linear width mismatches pass this rubric.** The transparent grader does
   not chase shapes through the graph, so a plan whose linear `inFeatures`
   disagrees with the upstream width can pass here and crash in PyTorch. This
   is measured, not suspected: in the 264-graph grounding study
   ([training/README.md](./training/README.md)), such graphs passed the rubric
   and failed 100% of real forward passes. The Neurarch product's shape
   propagator catches this class; porting a minimal width-chase into
   `findBlockers` is the highest-value hardening item.
2. **Score magnitude is not a quality ranking.** Same study: among clean
   graphs, Spearman rho between the rubric score and real training progress
   was -0.581. Treat pass/blocked as the grounded claim and the score as a
   validity margin; quality calibration against real runs is the open
   flywheel (`training/grounding_at_scale.py`).
3. **No torch execution in the reward path.** Shape-class failures are
   caught statistically (via the grounding study), not per-rollout; a sampled
   async forward-pass audit is the planned shape, kept out of the sub-ms path.
4. **Reference style bias.** Generated references use `replace_model` for
   design-from-spec families; grading is on the resulting graph, so this
   biases imitation style, not correctness.

## Reproduce every claim here

```bash
npx vitest run            # upper + lower bound suites
node calibrate.mjs --policy=reference
node calibrate.mjs --policy=noop
```

All without API keys or GPUs.
