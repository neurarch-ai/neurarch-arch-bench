#!/usr/bin/env python3
"""Grounding at scale: mint (architecture, verifier score, REAL training run)
triples on free GPU time.

The short grounding probe (grounding.py, 60 steps) established that the
verifier's pass/blocked boundary predicts physical failure. This script goes
after the harder question with proper runs: train every CLEAN generated
architecture near to convergence on a fixed task distribution and record the
whole curve, producing the dataset that lets the 0..100 score be CALIBRATED
against achievable quality instead of hand-tuned.

Each output row is a triple:
  fingerprint  {taskId, family, seed, params, components}
  verifier     {score, blockers}
  outcome      {curve: [loss@checkpoints], finalLoss, bestLoss, aucLoss,
                steps, converged}

Run anywhere with torch (Colab T4: ~80 archs in ~30-40 min):
  node dump_grounding_set.mjs --count=40 --seed=123 --out=gset.jsonl
  python grounding_at_scale.py --set gset.jsonl --steps 800 --out triples.jsonl
  python analyze_grounding.py --triples triples.jsonl

Scale path beyond one machine: the Neurarch product's free-GPU loop
(Colab/Kaggle notebooks that POST per-epoch curves back with an HMAC token)
collects the same triples from real user runs; this script is the
self-serve batch engine for study-sized datasets.

Honest scope: quality here = achievable loss on a synthetic fixed-teacher
task distribution (dataset-free, so any generated input shape works). It is
a real optimization outcome, not a benchmark-accuracy claim.
"""
import argparse
import json
import math

import torch

from grounding import GraphModel, make_batch


def train_full(model, graph, steps, batch, ckpt_every, lr, seed):
    """Train to (near) convergence; return curve + summary stats."""
    import torch.nn.functional as F
    torch.manual_seed(seed)
    inputs, feat = make_batch(graph, batch, seed=seed)
    with torch.no_grad():
        out = model(inputs)
    out_dim = out.shape[-1]
    teacher = torch.randn(feat.shape[-1], max(out_dim, 1))
    if out_dim >= 2:
        labels = (feat @ teacher).argmax(dim=-1)
        loss_fn = lambda o: F.cross_entropy(o if o.dim() == 2 else o.mean(dim=1), labels)
    else:
        target = torch.tanh(feat @ teacher)
        loss_fn = lambda o: F.mse_loss(o if o.dim() == 2 else o.mean(dim=1), target)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve, best = [], float("inf")
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(inputs))
        loss.backward()
        opt.step()
        v = loss.item()
        best = min(best, v)
        if step % ckpt_every == 0 or step == steps - 1:
            curve.append(round(v, 5))
        # Early stop on plateau: last 3 checkpoints within 0.1% of best.
        if len(curve) >= 6 and all(abs(c - best) / max(best, 1e-9) < 1e-3 for c in curve[-3:]):
            break
    finite = [c for c in curve if math.isfinite(c)]
    auc = sum(finite) / len(finite) if finite else float("nan")
    return {
        "curve": curve,
        "finalLoss": curve[-1] if curve else float("nan"),
        "bestLoss": best if math.isfinite(best) else float("nan"),
        "aucLoss": round(auc, 5) if math.isfinite(auc) else None,
        "steps": (len(curve) - 1) * ckpt_every if curve else 0,
        "converged": len(curve) >= 6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="sets", nargs="+", default=["grounding_set.jsonl"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-seed", type=int, default=7, help="fixed task distribution across archs")
    ap.add_argument("--out", default="triples.jsonl")
    args = ap.parse_args()

    entries = []
    for p in args.sets:
        with open(p) as f:
            entries.extend(json.loads(l) for l in f if l.strip())
    clean = [e for e in entries if e["variant"] == "clean"]
    print(f"{len(clean)} clean architectures (of {len(entries)} rows; corrupted variants "
          f"belong to grounding.py's boundary study, not the calibration set)")

    written = 0
    with open(args.out, "w") as out:
        for i, e in enumerate(clean):
            row = {
                "fingerprint": {
                    "taskId": e["taskId"],
                    "family": e["taskId"].replace("gen-", "").rsplit("-", 1)[0],
                    "params": e["params"],
                    "components": len(e["graph"]["components"]),
                },
                "verifier": {"score": e["verifierScore"], "blockers": e["verifierBlockers"]},
            }
            try:
                model = GraphModel(e["graph"])
                row["outcome"] = train_full(model, e["graph"], args.steps, args.batch,
                                            args.ckpt_every, args.lr, args.data_seed)
                row["ok"] = True
            except Exception as err:
                row["ok"] = False
                row["error"] = str(err)[:160]
            out.write(json.dumps(row) + "\n")
            out.flush()  # crash-safe: Colab disconnects lose nothing finished
            written += 1
            tag = "OK " if row.get("ok") else "ERR"
            fl = row.get("outcome", {}).get("finalLoss")
            print(f"[{tag}] {i + 1}/{len(clean)} {e['taskId']:<16} score={e['verifierScore']:>3}"
                  f"{f' finalLoss={fl:.4f}' if isinstance(fl, float) and math.isfinite(fl) else ''}")

    print(f"\nWrote {written} triples to {args.out}")
    print("Next: python analyze_grounding.py --triples", args.out)


if __name__ == "__main__":
    main()
