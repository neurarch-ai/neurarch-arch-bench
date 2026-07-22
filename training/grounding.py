#!/usr/bin/env python3
"""Grounding study: does the verifier's verdict track real trainability?

Reads the graphs dumped by dump_grounding_set.mjs, builds each one as an
actual PyTorch model, trains it briefly on a synthetic task, and measures:

  constructs   — the modules can be instantiated at all
  forward_ok   — a forward pass runs without shape errors
  trains       — loss decreases by >= 20% over a short budget

Then it groups those outcomes by the verifier's verdict (pass / blocker /
score) and prints the confusion. The claim under test is narrow and honest:
a verifier "blocker" should predict construction or forward failure, and
clean graphs should overwhelmingly construct and train. This measures
TRAINABILITY (does optimization make progress), not final model quality.

  node dump_grounding_set.mjs --count=40 --seed=123 --out=grounding_set.jsonl
  python grounding.py --set grounding_set.jsonl --steps 60 --out grounding_results.csv

CPU is fine: these are tiny nets and short budgets (a few minutes total).
"""
import argparse
import csv
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Graph -> PyTorch ─────────────────────────────────────────────────────────
# Conventions mirror the benchmark's estimator: a linear fed a 4D conv map
# global-average-pools spatial dims first; fed a 3D sequence, mean-pools the
# sequence. Attention divisibility is asserted at CONSTRUCTION (as any real
# implementation would), so a broken config fails exactly where it would fail
# in a real codebase.

class GQA(nn.Module):
    """Minimal grouped-query attention. Asserts the two divisibility rules."""

    def __init__(self, embed_dim, num_heads, num_kv_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embedDim % numHeads != 0"
        assert num_heads % num_kv_heads == 0, "numHeads % numKVHeads != 0"
        self.h, self.kv, self.d = num_heads, num_kv_heads, embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, self.kv * self.d)
        self.v = nn.Linear(embed_dim, self.kv * self.d)
        self.o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(B, L, self.kv, self.d).transpose(1, 2).repeat_interleave(self.h // self.kv, dim=1)
        v = self.v(x).view(B, L, self.kv, self.d).transpose(1, 2).repeat_interleave(self.h // self.kv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x, need_weights=False)[0]


def build_module(comp):
    """Instantiate one graph node as an nn.Module (or a marker for passthrough)."""
    t, p = comp["type"], comp.get("params") or {}
    if t == "linear":
        return nn.Linear(int(p["inFeatures"]), int(p["outFeatures"]))
    if t == "relu":
        return nn.ReLU()
    if t == "conv2d":
        k = int(p.get("kernelSize", 3))
        return nn.Conv2d(int(p["inChannels"]), int(p["outChannels"]), k, padding=k // 2)
    if t == "embedding":
        return nn.Embedding(int(p["numEmbeddings"]), int(p["embeddingDim"]))
    if t == "multiHeadAttention":
        return MHA(int(p["embedDim"]), int(p["numHeads"]))
    if t == "groupedQueryAttention":
        return GQA(int(p["embedDim"]), int(p["numHeads"]), int(p.get("numKVHeads", p["numHeads"])))
    if t == "layerNorm":
        shape = p.get("normalizedShape")
        return nn.LayerNorm(shape if isinstance(shape, list) else [int(shape)]) if shape else nn.Identity()
    if t == "batchNorm1d":
        return nn.BatchNorm1d(int(p["numFeatures"]))
    if t in ("input", "output", "concatenate"):
        return None  # handled structurally
    return nn.Identity()  # unknown types pass through, matching the serializer's tolerance


class GraphModel(nn.Module):
    """Executes the benchmark graph in topological order, multi-input aware."""

    def __init__(self, graph):
        super().__init__()
        self.comps = {c["id"]: c for c in graph["components"]}
        self.preds = {cid: [] for cid in self.comps}
        for cn in graph["connections"]:
            if cn["to"] in self.preds:
                self.preds[cn["to"]].append(cn["from"])
        self.order = self._toposort(graph)
        self.mods = nn.ModuleDict()
        for cid, c in self.comps.items():
            m = build_module(c)
            if m is not None:
                self.mods[cid] = m
        self.input_ids = [c["id"] for c in graph["components"] if c["type"] == "input"]
        outs = [c["id"] for c in graph["components"] if c["type"] == "output"]
        if not outs:
            raise ValueError("no output node")
        self.output_id = outs[0]

    def _toposort(self, graph):
        indeg = {cid: len(self.preds[cid]) for cid in self.comps}
        queue = [cid for cid, d in indeg.items() if d == 0]
        succ = {cid: [] for cid in self.comps}
        for cn in graph["connections"]:
            if cn["from"] in succ:
                succ[cn["from"]].append(cn["to"])
        order = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        return order

    def forward(self, inputs):
        """inputs: dict input_name -> tensor."""
        vals = {}
        for cid in self.order:
            c = self.comps[cid]
            t = c["type"]
            if t == "input":
                vals[cid] = inputs[c["name"]]
                continue
            srcs = [vals[p] for p in self.preds[cid] if p in vals]
            if not srcs:
                raise RuntimeError(f"node {c['name']} has no computed input (disconnected)")
            if t == "concatenate":
                vals[cid] = torch.cat(srcs, dim=-1)
                continue
            # A non-merging node fed by parents of differing width is not a
            # graph this builder may quietly repair. Taking srcs[0] silently
            # discards an edge the designer drew, and which edge survives is
            # decided by list order: the same graph then runs or raises a shape
            # error depending on the sequence its connections happen to be
            # emitted in. Refusing here keeps "it ran" a statement about the
            # graph rather than about the ordering.
            widths = {s.shape[-1] for s in srcs}
            if len(widths) > 1:
                raise RuntimeError(
                    f"node {c['name']} merges parents of differing widths "
                    f"({', '.join(str(w) for w in sorted(widths))}); graph is under-determined")
            x = srcs[0]
            if t == "output":
                vals[cid] = x
                continue
            m = self.mods[cid]
            if isinstance(m, nn.Linear) and x.dim() == 4:
                x = x.mean(dim=(2, 3))          # implicit GAP, the estimator's convention
            elif isinstance(m, nn.Linear) and x.dim() == 3:
                x = x.mean(dim=1)               # sequence mean-pool before a head
            elif isinstance(m, nn.BatchNorm1d) and x.dim() == 3:
                x = m(x.transpose(1, 2)).transpose(1, 2)
                vals[cid] = x
                continue
            vals[cid] = m(x)
        if self.output_id not in vals:
            raise RuntimeError("output node unreachable")
        return vals[self.output_id]


# ── Synthetic tasks ──────────────────────────────────────────────────────────
# One fixed-teacher task per input modality so "loss decreases" is meaningful.

def make_batch(graph, batch=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    inputs, feats = {}, []
    for c in graph["components"]:
        if c["type"] != "input":
            continue
        shape = c["params"]["shape"]
        if len(shape) == 2:                      # [1, D] vector
            x = torch.randn(batch, shape[1], generator=g)
            # embeddings need token ids, detect a downstream embedding
            if any(cc["type"] == "embedding" for cc in graph["components"]):
                vocab = min(int(next(cc["params"]["numEmbeddings"] for cc in graph["components"] if cc["type"] == "embedding")), 30000)
                x = torch.randint(0, vocab, (batch, shape[1]), generator=g)
            inputs[c["name"]] = x
            feats.append(x.float().reshape(batch, -1) if x.dtype != torch.long else F.one_hot(x % 32, 32).float().reshape(batch, -1))
        elif len(shape) == 4:                    # [1, 3, S, S] image
            x = torch.randn(batch, shape[1], shape[2], shape[3], generator=g)
            inputs[c["name"]] = x
            feats.append(x.reshape(batch, -1))
    return inputs, torch.cat(feats, dim=-1)


def train_briefly(model, graph, steps=60, batch=64):
    """Return (initial_loss, final_loss). Raises on forward/shape errors."""
    inputs, feat = make_batch(graph, batch)
    with torch.no_grad():
        out = model(inputs)
    out_dim = out.shape[-1]
    torch.manual_seed(7)
    if out_dim >= 2:                              # classification against a fixed linear teacher
        teacher = torch.randn(feat.shape[-1], out_dim)
        labels = (feat @ teacher).argmax(dim=-1)
        loss_fn = lambda o: F.cross_entropy(o if o.dim() == 2 else o.mean(dim=1), labels)
    else:                                         # scalar head (retrieval score): regress teacher scores
        teacher = torch.randn(feat.shape[-1], 1)
        target = torch.tanh(feat @ teacher)
        loss_fn = lambda o: F.mse_loss(o if o.dim() == 2 else o.mean(dim=1), target)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = None
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(inputs))
        if initial is None:
            initial = loss.item()
        loss.backward()
        opt.step()
    return initial, loss.item()


# ── Study ────────────────────────────────────────────────────────────────────

def spearman(xs, ys):
    """Spearman rank correlation, pure python (no scipy dependency)."""
    n = len(xs)
    if n < 3:
        return float("nan")
    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="sets", nargs="+", default=["grounding_set.jsonl"],
                    help="one or more JSONL sets (merge multiple seeds for a bigger study)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", default="grounding_results.csv")
    args = ap.parse_args()

    rows = []
    entries = []
    for path in args.sets:
        with open(path) as f:
            entries.extend(json.loads(line) for line in f if line.strip())

    for e in entries:
        row = {
            "taskId": e["taskId"], "variant": e["variant"],
            "verifierScore": e["verifierScore"],
            "verifierBlocked": len(e["verifierBlockers"]) > 0,
            "constructs": False, "forward_ok": False, "trains": False,
            "initial_loss": "", "final_loss": "", "error": "",
        }
        try:
            model = GraphModel(e["graph"])
            row["constructs"] = True
            initial, final = train_briefly(model, e["graph"], steps=args.steps)
            row["forward_ok"] = True
            row["initial_loss"], row["final_loss"] = round(initial, 4), round(final, 4)
            row["trains"] = math.isfinite(final) and final < 0.8 * initial
        except Exception as err:
            row["error"] = str(err)[:160]
        rows.append(row)
        tag = "OK " if row["trains"] else ("FWD" if row["forward_ok"] else ("CON" if row["constructs"] else "ERR"))
        print(f"[{tag}] {e['taskId']:<16} {e['variant']:<16} score={e['verifierScore']:>3} "
              f"blocked={row['verifierBlocked']} {row['error']}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # The confusion that matters: verifier verdict vs. physical outcome.
    print("\n== Verifier verdict vs. reality ==")
    def bucket(pred):
        sel = [r for r in rows if pred(r)]
        n = len(sel)
        if not n:
            return "n=0"
        c = sum(r["constructs"] for r in sel)
        fo = sum(r["forward_ok"] for r in sel)
        tr = sum(r["trains"] for r in sel)
        return f"n={n:<3} constructs {c/n:>5.0%}  forward_ok {fo/n:>5.0%}  trains {tr/n:>5.0%}"
    print(f"  verifier PASS (clean):      {bucket(lambda r: r['variant'] == 'clean')}")
    print(f"  verifier BLOCKED:           {bucket(lambda r: r['verifierBlocked'])}")
    print(f"  not blocked, corrupted:     {bucket(lambda r: not r['verifierBlocked'] and r['variant'] != 'clean')}")

    # Second question: among CLEAN graphs, does the score's magnitude track
    # optimization progress, or is it only the pass/blocked boundary that is
    # grounded? Honest either way: a weak correlation here means the score is
    # a validity gate, not a quality ranking, and we say so.
    clean = [r for r in rows if r["variant"] == "clean" and r["forward_ok"]
             and isinstance(r["initial_loss"], float) and r["initial_loss"] > 0]
    if len(clean) >= 3:
        scores = [float(r["verifierScore"]) for r in clean]
        decrease = [(r["initial_loss"] - r["final_loss"]) / r["initial_loss"] for r in clean]
        rho = spearman(scores, decrease)
        print(f"\n== Score magnitude vs. training progress (clean graphs, n={len(clean)}) ==")
        print(f"  Spearman rho(verifier score, relative loss decrease) = {rho:.3f}")
        print("  Interpretation: the pass/blocked boundary is the grounded claim;")
        print("  treat the 0..100 score as a validity margin, not a quality ranking,")
        print("  unless this correlation is strong across seeds.")

    print(f"\nWrote {args.out}")
    print("Read it as: blockers should predict construction/forward failure;")
    print("clean graphs should construct and train. The 'not blocked, corrupted'")
    print("row exposes what this transparent rubric does NOT catch (e.g. linear")
    print("in/out mismatches, which the richer product verifier does flag).")


if __name__ == "__main__":
    main()
