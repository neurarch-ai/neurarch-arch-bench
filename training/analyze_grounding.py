#!/usr/bin/env python3
"""Analyze grounding triples: is the verifier score calibrated against real
training outcomes, overall and within each family?

Within-family analysis is the one that matters: across families, "harder task
distribution" confounds "better architecture" (a transformer's achievable
loss and an MLP's are not comparable), which is exactly the confound behind
the naive negative correlation the short probe reported.

  python analyze_grounding.py --triples triples.jsonl
"""
import argparse
import json
import math
from collections import defaultdict

from grounding import spearman


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="triples.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.triples) if l.strip()]
    ok = [r for r in rows if r.get("ok") and isinstance(r.get("outcome", {}).get("bestLoss"), float)
          and math.isfinite(r["outcome"]["bestLoss"])]
    print(f"{len(ok)}/{len(rows)} triples usable")
    if len(ok) < 3:
        return

    def corr(sel, label):
        if len(sel) < 3:
            print(f"  {label:<12} n={len(sel):<4} (too few)")
            return
        scores = [r["verifier"]["score"] for r in sel]
        # Negate loss so positive rho means "higher score, better outcome".
        quality = [-r["outcome"]["bestLoss"] for r in sel]
        rho = spearman(scores, quality)
        print(f"  {label:<12} n={len(sel):<4} rho(score, -bestLoss) = {rho:+.3f}")

    print("\n== Score vs achievable quality ==")
    corr(ok, "ALL (raw)")
    by_family = defaultdict(list)
    for r in ok:
        by_family[r["fingerprint"]["family"]].append(r)
    for fam in sorted(by_family):
        corr(by_family[fam], fam)

    # Params as a sanity covariate: is score just tracking size?
    params = [r["fingerprint"]["params"] for r in ok]
    scores = [r["verifier"]["score"] for r in ok]
    print(f"\n  rho(score, params) = {spearman(scores, [float(p) for p in params]):+.3f}  "
          "(high = the score mostly measures size, not design)")

    print("\nReading guide: within-family positive rho = the score ranks better designs")
    print("higher on comparable tasks; raw cross-family rho is confounded by task")
    print("difficulty and is reported only for honesty. These triples are also the")
    print("training set for a learned quality head (score calibration), the step that")
    print("turns the verifier from a validity gate into a calibrated predictor.")


if __name__ == "__main__":
    main()
