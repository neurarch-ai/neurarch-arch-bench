# A Verifiable Environment for Neural Architecture Design

**Draft, {DATE}.** Author: Xin Gao (Neurarch). Correspondence via
[neurarch.com](https://neurarch.com). Code and data:
`github.com/neurarch-ai/neurarch-arch-bench` (MIT).

> Working draft. All results marked *measured* are reproduced by the commands
> in this repository without API keys or GPUs, except the amplification study
> (Section 6), which requires provider API keys and is marked *pending* until
> a run is pasted in. Do not cite without the maintainer's confirmation.

## Abstract

Reinforcement learning and agent evaluation both depend on a *verifier*: a
function that scores a candidate cheaply and without a human. Code and
mathematics have compilers and unit tests; neural architecture design has had
no public verifier. We present an environment in which an agent designs a
neural network by emitting edits to a typed, structured graph, and a
deterministic, sub-millisecond verifier grades the result against structural,
budget, and serving-physics constraints, with no human or LLM in the scoring
loop. The environment ships (i) a procedural task generator across ten
families with satisfiability and non-vacuity proofs, (ii) anti-gaming
constraints that reject wholesale rebuilds masquerading as surgical fixes,
(iii) a difficulty-calibration harness that reports per-family pass rates
against the pass-rate floor labs cite for usable RL environments, and (iv) a
grounding study relating the verifier's verdict to real PyTorch behavior.
Across 264 architectures, verifier blockers predict forward-pass failure with
perfect precision (96/96), while graphs the verifier passes construct and
train in 90% of cases; we also report, honestly, that the 0-100 health score
is a validity margin rather than a quality ranking (within-family Spearman
correlation with training progress is weak). Because grading is a pure
function, the same environment serves as a leaderboard, an RL training gym,
and a source of verified supervised data; we release all three.

## 1. Introduction

The scarce ingredient in modern agent training and evaluation is not the
policy but the verifier. SWE-Bench [Jimenez et al., 2024] grades a code patch
by running the project's own test suite; tau-bench [Yao et al., 2024] grades a
tool-using dialogue by diffing the resulting database state against a goal.
Neither requires a human judge or a second model, and this is precisely why
they are trusted and why they can be used as training signals.

Neural architecture design has lacked such a verifier, for a structural
reason: whether a proposed network is even runnable (attention head counts
that divide the embedding dimension, linear widths that match upstream shapes,
a graph whose input reaches its output, a parameter count inside a budget) is
a pure function only when architectures are represented as typed, structured
graphs rather than free-form code. Coding agents emit code, not graphs, so
their output carries no shape-level verifiability. We close this gap by
building the environment around a typed graph and the deterministic checks
that a production editor already runs on it.

Our contributions are: (1) a design-from-spec and edit-in-place environment
with a sub-millisecond programmatic verifier; (2) a procedural task generator
with satisfiability proofs, non-vacuity proofs, and anti-gaming constraints;
(3) a calibration harness that measures difficulty against the labs' usable
band, with keyless self-tests that bracket the harness itself; (4) a grounding
study connecting verdicts to real PyTorch behavior, reported with its
negative results intact; and (5) three downstream uses of the same verifier
(leaderboard, RL gym, verified SFT data), all released.

## 2. The environment

**State.** A typed graph of a neural network: components drawn from a
vocabulary of 182 layer types, each carrying parameters, connected by directed
edges. **Action.** A batch of structured edits from a fixed vocabulary:
`add_component`, `add_connection`, `update_params`, `delete_component`,
`replace_model`, and others, the same vocabulary a production editor executes.
**Reward.** A grading function returning a 0-100 health score plus a list of
hard blockers.

The grader composes three checks, each pure and sub-millisecond: a structural
score with hard blockers (disconnected input/output, attention divisibility,
parameter bloat), a guardrail pass over the proposed edits, and a shape
propagator that catches shape bugs an edit would introduce. No LLM grades
anything, so there is no persuasion surface and no stochasticity in the
reward.

A task pairs a natural-language specification with a starting graph and
machine-checkable constraints, e.g. *"This encoder fails validation: embedDim
(192) is not divisible by numHeads (7). Repair the attention configuration in
place with at most 2 actions. Do not rebuild the model from scratch."*

## 3. Task generation

Curated tasks are a seed. A deterministic generator mints splits of any size
from a seed across ten families: six design-from-spec (dense classifier,
autoencoder, convolutional classifier, transformer encoder, grouped-query
attention encoder, two-tower retrieval) and four edit-in-place (repair a
broken attention configuration, trim an oversized model under a parameter
budget, grow an undersized model into a two-sided parameter band, insert
normalization). Because tasks are synthesized rather than scraped, a held-out
split need never appear on the public web; contamination resistance is a
property of construction, not obscurity.

**Satisfiability and non-vacuity (measured).** Every generated task carries
its own reference solution. A 500-case sweep asserts that each reference
passes its own task (no unsatisfiable tasks), and every edit-in-place start
graph is asserted to fail its own task untouched (no vacuous tasks). Both run
in CI without keys.

**Anti-gaming.** Edit-in-place families forbid `replace_model` and
`clear_canvas`, so an agent that cannot diagnose a defect cannot pass by
regenerating the whole network; budgets are two-sided bands rather than
ceilings, so "delete everything" fails a trim task; a KV-cache budget is
always paired with a required-attention constraint, so amputating attention
cannot zero the cache. Nine such strategies, their defenses, and the pinned
tests that fail if any defense is weakened are enumerated in the repository's
red-team report (`ROBUSTNESS.md`). An LLM-driven task proposer accepts nothing
without a satisfiability proof: the proposer's own reference must pass the
grader under always-enforced safety-net constraints, so LLM authorship, unsafe
for human- or LLM-judged benchmarks, is safe here.

## 4. Difficulty calibration

An ungameable environment is still useless at 0% pass (no learning signal) or
near 100% (nothing to learn). Practitioners cite a floor near 2-3% pass rate
on the hard end (one success per 64-128 rollouts) for a usable RL environment.
The calibration harness measures per-family pass rates with Wilson 95%
intervals and flags each family's band. Two keyless self-test policies bracket
the harness itself: a `reference` policy that replays known-good solutions
(measured: 100% pass) and a `noop` policy that submits empty plans (measured:
0% pass). If either self-test deviates, the harness rather than the model is
broken, and CI fails.

**Measured (claude-sonnet-4-6, public benchmark, n=12/family).** Single-shot pass rate per family: eight families saturate (100%, a curriculum tier for a frontier model) and two are hard: transformer encoder (25%) and insert-norm (0%). Deterministic grading, reproducible from the seed.

## 5. Grounding study

Does the verifier's verdict correspond to reality? We generated clean
reference architectures and systematically corrupted variants (broken
attention divisibility, mismatched linear width, a severed mid-graph edge),
built every graph as a PyTorch module, and trained each briefly on a synthetic
fixed-teacher task.

**Result (measured, 264 graphs, two seeds).** Verifier blockers predicted a
forward-pass failure with perfect precision: 96 of 96 blocked graphs failed to
run. Graphs the verifier passed constructed in 80 of 80 cases and made
training progress in 90%. The one systematic miss is honest and instructive:
the transparent rubric in this repository does not chase linear widths through
the graph, so a width mismatch can pass the rubric and crash PyTorch; the
richer verifier in the production system flags this class. We treat the
pass/blocked boundary as the grounded claim.

**Negative result (measured).** Within clean graphs, the 0-100 score's
magnitude does *not* rank training quality: Spearman correlation between the
score and relative loss decrease was -0.58, because deeper, more structured
models make slower per-step progress on the short synthetic probe than tiny
ones. The score is therefore a validity margin, not a quality ranking. A
learned quality head, trained on (architecture, verdict, real training curve)
triples that the environment can mint at scale, is the natural next step and
the point at which the verifier would become a calibrated predictor rather
than a gate.

## 6. Verifier-in-the-loop amplification

*Pending a run.* Because grading is free, the verifier's failure messages can
be fed back to a policy for repair rounds. We measure, per provider, the pass
rate of a single shot versus the pass rate when up to k repair rounds are
allowed, holding tasks, model, and prompt fixed so that the only variable is
the feedback loop. The framing is deliberately pro-model: the environment's
value is the lift it adds to any model.

| Model | single-shot pass | with verifier feedback | lift |
| --- | --- | --- | --- |
| claude-sonnet-4-6 | 82% | 100% | +18 pts |

The lift is entirely on the two families a frontier model finds hard: insert-norm 0% -> 100% and transformer encoder 25% -> 100% (the verifier's failure messages let the model repair them), while the eight curriculum families are already at 100%. n=12 per family, 117 graded (three transient network errors excluded).

## Auditing LLM reward models with the verifier

Frontier labs increasingly use a strong model *as* a reward model to optimize
objectives that are not directly verifiable (xAI reported this for Grok 4.1).
The known failure mode is that an LLM reward model drifts: it approves answers
that are actually broken. In a domain with a verifiable reward that drift is
*measurable*. `reward_anchor.mjs` builds a verifier-labeled set (a reference
design passes, its unsolved starting graph fails) and, given an LLM acting as a
reward model, reports its agreement with the verifier and, crucially, its
**false-positive rate**: how often it approves a design the verifier proves is
broken. That number is what a lab needs to trust an LLM judge in this domain,
and it can only be produced where a ground-truth verifier exists. A use of the
environment beyond training or evaluation: a calibration anchor for the LLM
reward models labs deploy where no verifier is available.

| model | tier | agreement | false-pos. | false-neg. |
| --- | --- | --- | --- | --- |
| qwen-2.5-72b-instruct | open | 100.0% | **0.0%** | 0.0% |
| grok-4.5 | frontier | 95.0% | **0.0%** | 5.0% |
| claude-sonnet-4-6 | frontier | 93.3% | **0.0%** | 6.7% |
| mistral-large | open | 93.3% | **0.0%** | 6.7% |
| deepseek-chat (V3) | open | 93.3% | **0.0%** | 6.7% |
| grok-4.20-reasoning | frontier | 93.3% | **0.0%** | 6.7% |
| grok-4 | frontier | 91.7% | **0.0%** | 8.3% |
| grok-4-fast | frontier | 91.7% | **0.0%** | 8.3% |
| gemma-2-27b-it | open | 90.0% | **0.0%** | 10.0% |
| llama-3.3-70b-instruct | open | 90.0% | **0.0%** | 10.0% |
| deepseek-r1 (reasoning) | open | 90.0% | **0.0%** | 10.0% |
| deterministic verifier (ours) | --- | 100% | **0%** | **0%** |

Across eleven reward models (n=60 each) spanning closed-frontier and open-weights, not one approves a broken design (0% false positive across all eleven). The only failure mode is over-conservatism (0-10% false negative). The RLVR-corrupting mode (rewarding a broken design) does not appear; the verifier matches the best (0/0) for free.

## 7. Related work

**Verifiable agent benchmarks.** SWE-Bench and tau-bench established the
programmatic-verifier pattern in code and tool-use; SWE-Gym [2024] showed that
a few thousand verifiable tasks suffice to train agents to a double-digit
absolute improvement on SWE-Bench, and did so as free, open research. We
target the same pattern in a domain that lacked a public verifier.

**RL environments.** A growing market supplies labs with RL environments;
reward-hacking resistance is the quality bar most consistently cited, and
usable environments are expected to expose a hard band near a 2-3% pass floor.
We build directly against both criteria.

**Neural architecture search.** Classical NAS automates architecture design
under a proxy objective; our environment differs in providing a
human-legible, per-edit, sub-millisecond verifier over a typed graph, and in
serving as a training and evaluation substrate rather than a single search
procedure. The fit-to-budget procedure in the accompanying product is a
deterministic, constraint-satisfying special case.

## 8. Limitations

The transparent rubric released here does not verify linear-width consistency
(Section 5); the health-score magnitude is not a quality ranking (Section 5);
the reward path does not execute PyTorch per rollout, so shape-class failures
are caught statistically rather than per-instance; and generated references
lean on `replace_model` for design-from-spec families, biasing imitation
style. Each is documented with its measurement or its planned mitigation in
`ROBUSTNESS.md`.

## 9. Conclusion

Representing architectures as typed graphs turns "is this network sound, and
what will it cost to serve" into a pure function. That function is
simultaneously a leaderboard, an RL reward, and a verified-data generator, and
it grounds against real PyTorch behavior on the claim that matters. We release
the environment, the generator, the calibration and red-team harnesses, and
the training loops, so that architecture design can join code and mathematics
as a domain with a public verifier.

## Reproducibility

```bash
git clone https://github.com/neurarch-ai/neurarch-arch-bench && cd neurarch-arch-bench
npx vitest run                                  # satisfiability + non-vacuity + anti-gaming
node calibrate.mjs --policy=reference           # harness self-test (must be 100%)
node calibrate.mjs --policy=noop                # harness self-test (must be 0%)
node dump_grounding_set.mjs --count=40 --seed=123 --out=g.jsonl
python training/grounding_at_scale.py --set g.jsonl && python training/analyze_grounding.py
```

## References

Placeholder; to be completed with formal citations for SWE-Bench (Jimenez et
al., 2024), tau-bench (Yao et al., 2024), SWE-Gym (2024), and the RL
environment market analyses before submission.
