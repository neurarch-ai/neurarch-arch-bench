# neurarch-arch-bench

**A verifiable benchmark for agents that design neural-network architectures.**

A model is shown a spec ("design a CNN for CIFAR-10 under 50M params") and a starting graph. It emits a set of graph edits. We apply them and grade the resulting architecture against programmatic constraints: is it structurally valid, connected, within budget, does it contain the right kinds of layers. No human judge, no LLM judge, no GPU. Deterministic and reproducible.

```
spec + graph  ──▶  model emits edit actions  ──▶  apply  ──▶  verify + grade
```

## Why a benchmark like this

The scarce ingredient in RL and agent evaluation is a **verifier**: a function that, given a candidate, says how good it is, cheaply and without a human. Code and math have one (a compiler, a unit test). Architecture design did not have a public one. This is that verifier, plus a task set, plus a leaderboard runner.

It measures something narrow and real: **can a model turn a natural-language spec into a valid, connected, budget-respecting neural network**, expressed as edits to a structured graph. A model that hallucinates a layer that doesn't wire up, picks an attention head count that doesn't divide the embedding dim, or blows the parameter budget, fails, and the failure is machine-checkable.

## Run it

Zero dependencies, no build step.

**Try it with no API key.** The `reference` provider replays a known-good solution for every task, so the whole thing runs out of the box and prints the oracle (upper-bound) leaderboard:

```bash
node leaderboard.mjs --providers=reference
```

```
[PASS] reference cnn-cifar              score= 80 params=19936
[PASS] reference text-encoder           score= 63 params=3971584
...
-- Leaderboard --
  reference  8/8 passed  avg score 74
```

**Rank real models.** Bring an API key for whichever you want to measure:

```bash
XAI_API_KEY=xai-...        node leaderboard.mjs --providers=grok
ANTHROPIC_API_KEY=sk-...   node leaderboard.mjs --providers=claude
GEMINI_API_KEY=AIza...     node leaderboard.mjs --providers=gemini

# rank several at once, subset of tasks, markdown output for pasting
node leaderboard.mjs --providers=grok,claude,gemini --only=cnn-cifar,text-encoder --format=md

# pin a specific model, dump JSON
XAI_MODEL=grok-3 LEADERBOARD_OUT=board.json node leaderboard.mjs --providers=grok
```

A failing row shows exactly why:

```
[FAIL] grok     text-encoder           score= 18 params=0
         - structural blocker: attn: embedDim 100 not divisible by numHeads 7
```

See [RESULTS.md](./RESULTS.md) for the oracle baseline and how to contribute model rows.

## The action space

A policy returns `{ "actions": [ ... ] }` using the structured edit vocabulary:

| Action | Effect |
| --- | --- |
| `add_component` | add a layer, optionally spliced after an existing node |
| `add_connection` / `delete_connection` | wire / unwire two layers |
| `update_params` / `scale_params` | change layer hyperparameters |
| `delete_component` / `delete_components_matching` | remove layers (graph self-heals) |
| `replace_model` | rebuild the whole graph |

This is the same vocabulary the [Neurarch](https://neurarch.com) editor applies, so a policy that scores well here speaks a language a real tool can execute.

## How grading works

Each task carries machine-checkable constraints. A submission **passes** only if all hold:

- `forbidBlockers` — no hard structural failures (non-divisible attention heads, GQA head/kv mismatch, disconnected graph, empty graph)
- `mustReachOutput` — input actually reaches output
- `maxParams` — rough parameter estimate within budget
- `mustContainTypes` / `mustContainTypesAny` — required layer families present
- `minComponents` / `maxActions` — depth and edit-economy bounds
- `minScore` — a transparent 0..100 rubric (valid baseline + depth + nonlinearities + normalization + budget respect)

The verifier (`bench.mjs`) is deliberately simple and fully transparent: the entire grading rubric is in one readable file, so anyone can audit, reproduce, or extend it. (The Neurarch product uses a richer internal verifier; this benchmark's results are defined solely by the rubric in this repo.)

## Use it as a library / environment

```js
import { loadBenchmark, buildFixture, applyActions, gradeTask, scoreModel } from 'neurarch-arch-bench';

const bench = loadBenchmark();
const task = bench.tasks.find(t => t.id === 'cnn-cifar');
const start = buildFixture(bench, task.start);

const { model } = applyActions(start, myPolicyActions);  // your agent's edits
const result = gradeTask(task, model, myPolicyActions.length);
// { pass, score, params, blockers, failures }
```

`applyActions` + `scoreModel` are pure and sub-millisecond, so they double as an RL environment: state is the graph, action is an edit batch, reward is `scoreModel(...).score` (dense) or `gradeTask(...).pass` (sparse terminal).

## Contributing tasks

Add an entry to `tasks.json` with a `spec`, a `start` fixture, and `constraints`, plus a known-good reference solution in `solutions.json` keyed by the task id. The test suite asserts every task is solvable by its reference, so a task that's accidentally impossible (an over-tightened constraint) fails CI instead of silently breaking the benchmark. Keep constraints machine-checkable. PRs that add tasks, harder budgets, or new layer-family requirements are welcome.

## License

MIT. Built by [neurarch-ai](https://neurarch.com).
