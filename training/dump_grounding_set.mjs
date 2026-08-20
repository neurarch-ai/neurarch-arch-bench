#!/usr/bin/env node
/**
 * dump_grounding_set — build the dataset for the grounding study.
 *
 * The open question every cheap verifier must answer: does the score actually
 * track reality? This script samples generated architectures and emits, for
 * each one, the clean reference build plus systematically corrupted variants:
 *
 *   clean            — the reference solution applied (verifier: pass)
 *   divisibility     — attention numHeads set to a non-divisor (verifier: blocker)
 *   linear-mismatch  — one linear's inFeatures doubled (transparent rubric does
 *                      NOT catch this; PyTorch will. An honest probe of this
 *                      rubric's limits — the richer product verifier flags it.)
 *   disconnect       — a middle connection removed (verifier: connectivity)
 *
 * grounding.py then builds every graph in PyTorch, trains it briefly, and
 * correlates the verifier's verdict with constructability / trainability.
 *
 *   node dump_grounding_set.mjs --count=40 --seed=123 --out=grounding_set.jsonl
 */
import fs from 'node:fs';
import path from 'node:path';
import { applyActions, gradeTask, scoreModel } from '../bench.mjs';
import { generateCases } from '../generate.mjs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));
const COUNT = Math.max(1, parseInt(args.count ?? '40', 10) || 40);
const SEED = parseInt(args.seed ?? '123', 10) || 123;
const OUT = args.out ?? 'grounding_set.jsonl';
// --clean-only: the quality-head study trains clean graphs to convergence and
// has no use for corrupted ones, which never finish a forward pass.
const CLEAN_ONLY = args['clean-only'] === 'true' || args['clean-only'] === true;

function record(task, model, variant) {
  const { score, blockers, params } = scoreModel(model, task.budget ?? {});
  const grade = gradeTask(task, model, 0, []);
  return {
    taskId: task.id,
    variant,
    verifierScore: score,
    verifierBlockers: blockers,
    verifierPass: grade.pass, // grade of the GRAPH; action-count checks don't apply here
    params,
    graph: { name: model.name, components: model.components, connections: model.connections },
  };
}

/** Corruptions are structural functions of the clean graph. Return null when
 *  a corruption doesn't apply (e.g. no attention layer to break). */
const CORRUPTIONS = {
  divisibility(model) {
    const next = structuredClone(model);
    const attn = next.components.find(c =>
      (c.type === 'multiHeadAttention' || c.type === 'groupedQueryAttention') &&
      typeof c.params?.embedDim === 'number');
    if (!attn) return null;
    attn.params.numHeads = 7; // no embedDim in the generator is divisible by 7
    return next;
  },
  'linear-mismatch'(model) {
    const next = structuredClone(model);
    // Pick a linear that has a predecessor which is not an input (a mid-chain
    // linear), so the mismatch is a real wiring bug, not an input-width choice.
    const linears = next.components.filter(c => c.type === 'linear' && typeof c.params?.inFeatures === 'number');
    const mid = linears.find(l => {
      const preds = next.connections.filter(cn => cn.to === l.id).map(cn => cn.from);
      return preds.some(p => next.components.find(c => c.id === p)?.type !== 'input');
    });
    if (!mid) return null;
    mid.params.inFeatures = mid.params.inFeatures * 2;
    return next;
  },
  disconnect(model) {
    const next = structuredClone(model);
    // Remove an edge into a non-output node so a middle layer loses its input.
    const inner = next.connections.find(cn => {
      const to = next.components.find(c => c.id === cn.to);
      const from = next.components.find(c => c.id === cn.from);
      return to && to.type !== 'output' && from && from.type !== 'input';
    });
    if (!inner) return null;
    next.connections = next.connections.filter(cn => cn !== inner);
    // Rebind ports to match.
    const byId = new Map(next.components.map(c => [c.id, c]));
    for (const c of next.components) { c.inputs = []; c.outputs = []; }
    for (const cn of next.connections) {
      byId.get(cn.from)?.outputs.push(cn.to);
      byId.get(cn.to)?.inputs.push(cn.from);
    }
    return next;
  },
  // Three families bounded the residue; a reviewer is right that three is not
  // "no defect outside the checks exists". These four widen the sample toward
  // the defects that actually show up in generated designs: a normalisation
  // layer sized to the wrong axis, a convolution whose channel count disagrees
  // with its predecessor, a pooling stride that annihilates the spatial map,
  // and an embedding whose vocabulary no longer covers its own indices.
  'norm-dim'(model) {
    const next = structuredClone(model);
    // The generator mints layerNorm (normalizedShape) and batchNorm1d
    // (numFeatures); both are the same defect, a norm sized to the wrong axis.
    const norm = next.components.find(c =>
      (c.type === 'layerNorm' && typeof c.params?.normalizedShape === 'number') ||
      (c.type === 'batchNorm1d' && typeof c.params?.numFeatures === 'number'));
    if (!norm) return null;
    const key = norm.type === 'layerNorm' ? 'normalizedShape' : 'numFeatures';
    norm.params[key] = norm.params[key] + 13;   // never a width the graph carries
    return next;
  },
  'conv-channels'(model) {
    const next = structuredClone(model);
    const mid = next.components.filter(c =>
      c.type === 'conv2d' && typeof c.params?.inChannels === 'number').find(l => {
        const preds = next.connections.filter(cn => cn.to === l.id).map(cn => cn.from);
        return preds.some(p => next.components.find(c => c.id === p)?.type !== 'input');
      });
    if (!mid) return null;
    mid.params.inChannels = mid.params.inChannels * 3 + 1;
    return next;
  },
  'kernel-oversize'(model) {
    const next = structuredClone(model);
    // conv2d carries no stride in this generator, so the spatial-collapse defect
    // is expressed as a kernel wider than the feature map it slides over.
    const conv = next.components.find(c =>
      c.type === 'conv2d' && typeof c.params?.kernelSize === 'number');
    if (!conv) return null;
    conv.params.kernelSize = 99;
    return next;
  },
  'embedding-vocab'(model) {
    const next = structuredClone(model);
    const emb = next.components.find(c =>
      c.type === 'embedding' && typeof c.params?.numEmbeddings === 'number');
    if (!emb) return null;
    emb.params.numEmbeddings = 1;   // indices immediately run past the table
    return next;
  },
};

const lines = [];
for (const { task, start, reference } of generateCases(COUNT, SEED)) {
  const applied = applyActions(start, reference);
  if (applied.errors.length) {
    console.error(`skip ${task.id}: reference did not apply cleanly`);
    continue;
  }
  const clean = applied.model;
  lines.push(record(task, clean, 'clean'));
  if (CLEAN_ONLY) continue;
  for (const [variant, corrupt] of Object.entries(CORRUPTIONS)) {
    const corrupted = corrupt(clean);
    if (corrupted) lines.push(record(task, corrupted, variant));
  }
}

fs.writeFileSync(path.resolve(OUT), lines.map(l => JSON.stringify(l)).join('\n') + '\n');
const byVariant = {};
for (const l of lines) byVariant[l.variant] = (byVariant[l.variant] ?? 0) + 1;
console.log(`Wrote ${lines.length} graphs to ${OUT}`);
console.log(Object.entries(byVariant).map(([v, n]) => `  ${v}: ${n}`).join('\n'));
