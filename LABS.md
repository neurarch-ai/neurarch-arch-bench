# For labs and post-training teams

Everything in this repository is MIT and stays that way: the grader, the
10-family generator, the reward server, the GRPO/STaR training loops, the
calibration and red-team harnesses, and the public splits. Run all of it
without talking to us.

This page lists what we provide beyond the public repo, for teams using the
environment seriously.

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

`calibrate.mjs` is self-serve. Calibrated against claude-sonnet-4-6 (stable
across three runs), the public families split into three in the RL-learnable
band (two-tower 0%, GQA encoder 25-56%, transformer encoder 60-62%) and nine a
frontier model saturates, which serve as an easy curriculum tier. What we add:
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
