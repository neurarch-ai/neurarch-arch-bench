#!/usr/bin/env python3
"""quality_head — turn the environment's triples into a calibrated quality
predictor, and measure it against the health score it is meant to replace.

The grounding study reports a negative result: the 0-100 health score is a
validity margin, not a quality ranking. That is honest but it leaves the
obvious next question unanswered. The score cannot rank designs; can anything
trained on what the environment mints?

This fits a ridge regression on (architecture -> real training outcome) pairs
the environment produces for free, and scores it the only way that means
anything: leave-one-family-out. The head never sees the family it is tested on,
so it cannot win by memorising that convolutional graphs converge slowly, and
family identity is deliberately absent from the feature set for the same
reason. On each held-out family we report the rank correlation of the learned
head and, on exactly the same graphs, of the health score, so the comparison is
head to head.

Target. Not final loss: the synthetic probe is easy enough that a third of
clean graphs reach exactly zero, so final loss barely varies and nothing can
rank it. The target is the area under the loss curve, i.e. how fast the design
gets there, which does vary and is the thing a practitioner would want
predicted before spending GPU hours.

  python training/quality_head.py --triples triples.jsonl --graphs grounding_set.jsonl

Dependencies: numpy only, deliberately.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

# Broad classes, so the feature vector says "this graph is mostly attention"
# rather than naming layer types the held-out family might monopolise.
CLASSES = {
    "linear": "dense", "embedding": "dense",
    "conv2d": "conv", "conv1d": "conv",
    "multiHeadAttention": "attn", "groupedQueryAttention": "attn",
    "layerNorm": "norm", "batchNorm1d": "norm", "rmsNorm": "norm",
    "relu": "act", "gelu": "act", "silu": "act",
    "concatenate": "merge", "add": "merge",
}


def spearman(a, b):
    """Rank correlation with average ranks for ties; undefined when a side is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        # average ties
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def depth(graph):
    """Longest path from any input, by Kahn order; cycles cannot occur here."""
    comps = {c["id"]: c for c in graph["components"]}
    indeg = defaultdict(int)
    out = defaultdict(list)
    for cn in graph["connections"]:
        out[cn["from"]].append(cn["to"])
        indeg[cn["to"]] += 1
    d = {cid: 0 for cid in comps}
    queue = [cid for cid in comps if indeg[cid] == 0]
    seen = 0
    while queue:
        cid = queue.pop()
        seen += 1
        for nxt in out[cid]:
            d[nxt] = max(d[nxt], d[cid] + 1)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return max(d.values()) if d else 0


def features(graph, params, score):
    comps = graph["components"]
    n = max(1, len(comps))
    counts = defaultdict(int)
    widths = []
    for c in comps:
        counts[CLASSES.get(c["type"], "other")] += 1
        p = c.get("params") or {}
        for k in ("outFeatures", "embedDim", "outChannels", "embeddingDim", "normalizedShape", "numFeatures"):
            if isinstance(p.get(k), (int, float)):
                widths.append(float(p[k]))
    widths = widths or [1.0]
    return [
        math.log10(max(params, 1)),
        len(comps),
        len(graph["connections"]),
        depth(graph),
        math.log10(max(widths)),
        math.log10(sum(widths) / len(widths)),
        math.log10(max(widths) / max(1.0, min(widths)) + 1.0),  # width dynamic range
        counts["dense"] / n, counts["conv"] / n, counts["attn"] / n,
        counts["norm"] / n, counts["act"] / n, counts["merge"] / n,
        score / 100.0,
    ]


FEATURE_NAMES = ["log10 params", "components", "edges", "depth", "log10 max width",
                 "log10 mean width", "log width range", "frac dense", "frac conv",
                 "frac attn", "frac norm", "frac act", "frac merge", "health score"]


def ridge_fit(X, y, lam):
    """Closed-form ridge on standardised features; the intercept is not penalised."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Z = np.hstack([np.ones((len(Z), 1)), Z])
    P = np.eye(Z.shape[1]) * lam
    P[0, 0] = 0.0
    w = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    return (w, mu, sd)


def ridge_predict(model, X):
    w, mu, sd = model
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    return Z @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="triples.jsonl")
    ap.add_argument("--graphs", default="grounding_set.jsonl",
                    help="the dumped set, for the structural features triples do not carry")
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--out", default=None, help="write per-family results as JSON")
    args = ap.parse_args()

    graphs = {}
    for line in open(args.graphs):
        r = json.loads(line)
        if r.get("variant", "clean") == "clean":
            graphs[r["taskId"]] = r

    rows = []
    for line in open(args.triples):
        t = json.loads(line)
        if not t.get("ok"):
            continue
        g = graphs.get(t["fingerprint"]["taskId"])
        auc = t["outcome"].get("aucLoss")
        if g is None or not auc or auc <= 0:
            continue
        rows.append({
            "family": t["fingerprint"]["family"],
            "x": features(g["graph"], t["fingerprint"]["params"], t["verifier"]["score"]),
            "y": math.log10(auc),
            "score": t["verifier"]["score"],
        })

    if len(rows) < 40:
        raise SystemExit(f"only {len(rows)} usable triples; need more before this means anything")

    fams = sorted({r["family"] for r in rows})
    print(f"{len(rows)} clean triples across {len(fams)} families\n")
    print(f"{'held-out family':<18}{'n':>5}{'score rho':>12}{'head rho':>11}{'delta':>9}")

    per_family, head_all, score_all = [], [], []
    for fam in fams:
        tr = [r for r in rows if r["family"] != fam]
        te = [r for r in rows if r["family"] == fam]
        if len(te) < 4:
            continue
        Xtr = np.array([r["x"] for r in tr]); ytr = np.array([r["y"] for r in tr])
        Xte = np.array([r["x"] for r in te]); yte = np.array([r["y"] for r in te])
        pred = ridge_predict(ridge_fit(Xtr, ytr, args.lam), Xte)
        # Lower aucLoss is better, and the health score is "higher is better", so
        # the score is compared against -y to put both on the same orientation.
        rho_head = spearman(pred, yte)
        rho_score = spearman([r["score"] for r in te], -yte)
        per_family.append({"family": fam, "n": len(te),
                           "rho_score": rho_score, "rho_head": rho_head})
        head_all.append(rho_head); score_all.append(rho_score)
        d = ("" if math.isnan(rho_head) or math.isnan(rho_score)
             else f"{abs(rho_head) - abs(rho_score):+.2f}")
        print(f"{fam:<18}{len(te):>5}{rho_score:>12.2f}{rho_head:>11.2f}{d:>9}")

    hv = [abs(v) for v in head_all if not math.isnan(v)]
    sv = [abs(v) for v in score_all if not math.isnan(v)]
    print(f"\nmean |rho| over held-out families: health score {np.mean(sv):.2f}, "
          f"learned head {np.mean(hv):.2f}  (n={len(hv)} families)")
    print("The head never sees the family it is scored on, and family identity is not a feature.")

    # Which features carry it, fitted once on everything (reported, not tuned on).
    X = np.array([r["x"] for r in rows]); y = np.array([r["y"] for r in rows])
    w, _, _ = ridge_fit(X, y, args.lam)
    order = np.argsort(-np.abs(w[1:]))[:5]
    print("\ntop standardised weights: " +
          ", ".join(f"{FEATURE_NAMES[i]} {w[i+1]:+.2f}" for i in order))

    if args.out:
        json.dump({"per_family": per_family,
                   "mean_abs_rho_score": float(np.mean(sv)),
                   "mean_rho_head": float(np.mean(hv)),
                   "n_triples": len(rows), "lambda": args.lam},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
