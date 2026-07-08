# For labs and post-training teams

Everything in this repository is MIT and stays that way: the grader, the
10-family generator, the reward server, the GRPO/STaR training loops, the
calibration and red-team harnesses, and the public splits. Run all of it
without talking to us.

This page lists what we provide beyond the public repo, for teams using the
environment seriously. The short version: architecture design is a
verifiable-reward domain (RLVR), and the verifier is a tool a reasoning model
can call mid-thought (tool-integrated reasoning), so it plugs into how frontier
models are already trained.

## Private, never-published evaluation seeds

Splits are deterministic functions of a seed. A seed we generate for you and
never publish gives you an evaluation set that has never existed anywhere
public (no contamination argument to have), while staying fully reproducible
inside your infra. Includes rotation on your schedule.

## Custom task families

The generator's families are built from a typed graph vocabulary of 182 layer
types and a verifier that enforces structural, budget, and serving-physics
constraints (KV cache per token, parameter bands, action-economy limits).
We build families to your specification: your architecture priors, your
serving budgets, your hardware targets, with the same guarantees as the
public families (satisfiability proofs, broken-start proofs, anti-gaming
constraints; see [ROBUSTNESS.md](./ROBUSTNESS.md)).

## Difficulty calibration against your models

`calibrate.mjs` is self-serve. On the public benchmark against claude-sonnet-4-6
(n=12 per family), eight families saturate (100% single-shot, a curriculum tier)
and two are hard: transformer-encoder (25% single-shot) and insert-norm (0%).
Verifier-in-the-loop feedback lifts overall pass 82% -> 100% (+18 pts), entirely
on those two (insert-norm 0->100, transformer 25->100). Deterministic grading
makes it reproducible from the seed. What we add:
hardening families until the hard band sits where YOUR training runs need it
(labs typically cite a 2-3% pass floor), calibrated against your model's actual
pass rates rather than public-model proxies. A frontier model trivially solves
simple chains and single-edit repairs; making those hard for a specific model
(coupled defects, multi-branch shape agreement, tighter serving bands) is done
against that model, not a proxy.

## Grounding data

The flywheel in `training/grounding_at_scale.py` produces (architecture
fingerprint, verifier verdict, real training curve) triples. We license
aggregated triple datasets and can run targeted grounding studies on
architecture families you care about.

## Verified reasoning traces

Beyond verified SFT rows and grounding triples, `training/reasoning_traces.mjs`
mints `(spec -> reasoning -> verified design)` triples, rejection-sampled against
the deterministic verifier so only passing designs are kept. We license
model-specific and private-seed trace sets, and can target the families your
training run needs.

## Exclusivity

Task families or seed ranges can be exclusive to one lab for a defined
window. The public benchmark stays public; exclusivity applies to what we
build for you.

## Integration

The verifier is already consumable four ways: HTTP reward server
(`env-server.mjs`), Prime Intellect verifiers package
(`environments/neurarch_arch_design/`), MCP tools (`mcp-server.mjs`), and
plain library import. We support wiring any of them into your stack.

## Contact

Open a GitHub issue on this repo, or reach us through
[neurarch.com](https://neurarch.com).
