/**
 * neurarch-arch-bench — core engine.
 *
 * A verifiable benchmark for agents that DESIGN neural-network architectures.
 * State is a structured graph; a policy emits edit actions; we apply them and
 * grade the resulting graph against programmatic constraints. No human judge,
 * no LLM judge, no GPU — just a deterministic, transparent verifier.
 *
 * This file is intentionally self-contained (plain ESM, zero dependencies) so
 * anyone can clone the repo and run `node leaderboard.mjs` with no build step.
 * The verifier here is a deliberately simple, fully transparent rubric; the
 * Neurarch product uses a richer one internally, but the benchmark's grading is
 * defined entirely by what's in this file so results are reproducible by anyone.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── Benchmark loading ───────────────────────────────────────────────────────

export function loadBenchmark() {
  return JSON.parse(fs.readFileSync(path.join(__dirname, 'tasks.json'), 'utf8'));
}

/** Reference (oracle) solutions: a known-good action sequence per task id.
 *  Used by the 'reference' leaderboard provider (no API key) and by the
 *  solvability tests. The graded model never sees these. */
export function loadSolutions() {
  const raw = JSON.parse(fs.readFileSync(path.join(__dirname, 'solutions.json'), 'utf8'));
  delete raw._comment;
  return raw;
}

/** Hydrate a raw fixture (compact JSON) into a full graph with rebound ports. */
export function buildFixture(bench, name) {
  const f = bench.fixtures[name];
  if (!f) throw new Error(`unknown fixture: ${name}`);
  const components = f.components.map((c, i) => ({
    id: c.id,
    type: c.type,
    name: c.name,
    params: { ...(c.params ?? {}) },
    inputs: [],
    outputs: [],
  }));
  const byId = new Map(components.map(c => [c.id, c]));
  const connections = f.connections.map((cn, i) => ({ id: `c${i}`, from: cn.from, to: cn.to }));
  for (const cn of connections) {
    byId.get(cn.from)?.outputs.push(cn.to);
    byId.get(cn.to)?.inputs.push(cn.from);
  }
  return { name: f.name, components, connections };
}

// ─── Headless action applier ─────────────────────────────────────────────────
// The structural subset of the editor's action vocabulary, as a pure function.
// Deterministic ids (counter, never Date.now/Math.random) so a graph is a
// reproducible function of (start, actions).

const STRUCTURAL = new Set([
  'add_component', 'delete_component', 'delete_components_matching',
  'update_params', 'scale_params', 'add_connection', 'delete_connection',
  'rename_component', 'replace_model', 'clear_canvas',
]);

export function applyActions(model, actions) {
  const next = structuredClone(model);
  let components = next.components;
  let connections = next.connections;
  const errors = [];
  const skipped = [];
  let applied = 0;

  const used = new Set([...components.map(c => c.id), ...connections.map(c => c.id)]);
  let counter = 0;
  const newId = (p) => { let id; do { id = `${p}-b${counter++}`; } while (used.has(id)); used.add(id); return id; };
  const resolveId = (name) => components.find(c => c.name === name)?.id ?? null;
  const edgeExists = (from, to) => connections.some(cn => cn.from === from && cn.to === to);
  const mkConn = (from, to) => ({ id: newId('c'), from, to });
  const mkComp = (type, name, params) => ({ id: newId('n'), type, name, params: { ...(params ?? {}) }, inputs: [], outputs: [] });

  const deleteById = (id) => {
    const preds = connections.filter(cn => cn.to === id).map(cn => cn.from);
    const succs = connections.filter(cn => cn.from === id).map(cn => cn.to);
    connections = connections.filter(cn => cn.from !== id && cn.to !== id);
    for (const p of preds) for (const s of succs) if (p !== s && !edgeExists(p, s)) connections.push(mkConn(p, s));
    components = components.filter(c => c.id !== id);
  };

  for (const a of Array.isArray(actions) ? actions : []) {
    if (!a || !STRUCTURAL.has(a.type)) { if (a?.type) skipped.push(a.type); continue; }
    try {
      switch (a.type) {
        case 'add_component': {
          const node = mkComp(a.componentType, a.name, a.params);
          components.push(node);
          if (a.afterName) {
            const afterId = resolveId(a.afterName);
            if (!afterId) errors.push(`add_component: afterName "${a.afterName}" not found`);
            else { for (const cn of connections) if (cn.from === afterId) cn.from = node.id; connections.push(mkConn(afterId, node.id)); }
          }
          applied++; break;
        }
        case 'delete_component': {
          const id = resolveId(a.name);
          if (!id) { errors.push(`delete_component: "${a.name}" not found`); break; }
          deleteById(id); applied++; break;
        }
        case 'delete_components_matching': {
          let re; try { re = new RegExp(a.namePattern); } catch { errors.push(`bad regex ${a.namePattern}`); break; }
          const ids = components.filter(c => re.test(c.name)).map(c => c.id);
          for (const id of ids) deleteById(id);
          if (ids.length) applied++; break;
        }
        case 'update_params': {
          const comp = components.find(c => c.name === a.name);
          if (!comp) { errors.push(`update_params: "${a.name}" not found`); break; }
          comp.params = { ...comp.params, ...a.params }; applied++; break;
        }
        case 'scale_params': {
          let re = null;
          if (a.namePattern) { try { re = new RegExp(a.namePattern); } catch { errors.push(`bad regex`); break; } }
          const minV = typeof a.minValue === 'number' ? a.minValue : 1;
          let touched = 0;
          for (const c of components) {
            if (re && !re.test(c.name)) continue;
            const cur = c.params[a.paramKey];
            if (typeof cur === 'number') { c.params[a.paramKey] = Math.max(minV, Math.round(cur * a.factor)); touched++; }
          }
          if (touched) applied++; break;
        }
        case 'add_connection': {
          const from = resolveId(a.fromName), to = resolveId(a.toName);
          if (!from || !to) { errors.push(`add_connection: endpoint not found`); break; }
          if (!edgeExists(from, to)) connections.push(mkConn(from, to)); applied++; break;
        }
        case 'delete_connection': {
          const from = resolveId(a.fromName), to = resolveId(a.toName);
          if (!from || !to) { errors.push(`delete_connection: endpoint not found`); break; }
          connections = connections.filter(cn => !(cn.from === from && cn.to === to)); applied++; break;
        }
        case 'rename_component': {
          const comp = components.find(c => c.name === a.name);
          if (!comp) { errors.push(`rename_component: "${a.name}" not found`); break; }
          comp.name = a.newName; applied++; break;
        }
        case 'replace_model': {
          const nameToId = new Map();
          const built = (a.components ?? []).map(c => { const n = mkComp(c.componentType, c.name, c.params); nameToId.set(c.name, n.id); return n; });
          const builtConns = [];
          for (const cn of a.connections ?? []) { const f = nameToId.get(cn.from), t = nameToId.get(cn.to); if (f && t) builtConns.push(mkConn(f, t)); }
          components = built; connections = builtConns; applied++; break;
        }
        case 'clear_canvas': { components = []; connections = []; applied++; break; }
      }
    } catch (err) { errors.push(`${a.type}: ${err.message}`); }
  }

  // Rebind ports.
  const byId = new Map(components.map(c => [c.id, c]));
  for (const c of components) { c.inputs = []; c.outputs = []; }
  for (const cn of connections) { byId.get(cn.from)?.outputs.push(cn.to); byId.get(cn.to)?.inputs.push(cn.from); }
  next.components = components; next.connections = connections;
  return { model: next, applied, errors, skipped };
}

// ─── Verifier (transparent rubric) ───────────────────────────────────────────

const NONLINEARITIES = new Set(['relu', 'gelu', 'silu', 'tanh', 'sigmoid', 'leakyRelu', 'elu', 'mish', 'swish']);
const NORMS = new Set(['batchNorm1d', 'batchNorm2d', 'layerNorm', 'groupNorm', 'rmsNorm', 'instanceNorm2d']);
const WEIGHTED = new Set(['linear', 'conv1d', 'conv2d', 'conv3d', 'embedding', 'multiHeadAttention', 'attention', 'transformerBlock']);
const num = (p, k) => (typeof p?.[k] === 'number' ? p[k] : 0);

/** Rough per-layer parameter estimate. Approximate by design. */
export function estimateParams(model) {
  let sum = 0;
  for (const c of model.components) {
    const p = c.params ?? {};
    switch (c.type) {
      case 'linear': sum += num(p, 'inFeatures') * num(p, 'outFeatures'); break;
      case 'embedding': sum += num(p, 'numEmbeddings') * num(p, 'embeddingDim'); break;
      case 'conv1d': case 'conv2d': case 'conv3d':
        sum += num(p, 'inChannels') * num(p, 'outChannels') * Math.max(1, num(p, 'kernelSize')) ** 2; break;
      case 'multiHeadAttention': case 'attention': {
        const d = num(p, 'embedDim') || num(p, 'hiddenDim') || num(p, 'dModel'); sum += 4 * d * d; break;
      }
      case 'transformerBlock': {
        const d = num(p, 'hiddenDim') || num(p, 'embedDim') || num(p, 'dModel'); sum += 12 * d * d; break;
      }
      case 'mixtureOfExperts': {
        // E experts of (D->H->D) + a D->E router. New type: no existing task
        // or dataset uses it, so adding this changes no frozen measurement.
        const d = num(p, 'embedDim'); const h = num(p, 'hiddenDim');
        const e = Math.max(1, num(p, 'numExperts'));
        sum += e * (2 * d * h) + d * e; break;
      }
      default: break;
    }
  }
  return sum;
}

/** KV cache bytes per generated token at fp16-by-default. Canonical rules:
 *  full attention caches 2 x dim, GQA caches 2 x kvHeads x headDim, MLA caches
 *  the compressed latent (+ RoPE keys). 0 for attention-free graphs. */
export function kvBytesPerToken(model, bytesPerValue = 2) {
  let sum = 0;
  for (const c of model.components) {
    const p = c.params ?? {};
    const pos = (v, fb = 0) => { const x = Number(v); return Number.isFinite(x) && x > 0 ? x : fb; };
    switch (c.type) {
      case 'multiHeadAttention': case 'selfAttention': case 'causalAttention':
      case 'attention': case 'transformerBlock': {
        const dim = pos(p.embedDim ?? p.hiddenDim ?? p.dModel);
        sum += 2 * dim * bytesPerValue;
        break;
      }
      case 'groupedQueryAttention': {
        const heads = pos(p.numHeads);
        const kv = pos(p.numKVHeads, heads);
        const dim = pos(p.embedDim);
        const headDim = pos(p.headDim, heads > 0 ? dim / heads : 0);
        sum += 2 * kv * headDim * bytesPerValue;
        break;
      }
      case 'mla': {
        sum += (pos(p.kvLatentDim) + pos(p.ropeHeadDim)) * bytesPerValue;
        break;
      }
      default: break;
    }
  }
  return Math.round(sum);
}

/** True if some output node is reachable from some input node. */
export function inputReachesOutput(model) {
  const adj = new Map();
  for (const c of model.components) adj.set(c.id, []);
  for (const cn of model.connections) adj.get(cn.from)?.push(cn.to);
  const inputs = model.components.filter(c => c.type === 'input').map(c => c.id);
  const outputs = new Set(model.components.filter(c => c.type === 'output').map(c => c.id));
  if (!inputs.length || !outputs.size) return false;
  const seen = new Set(inputs); const stack = [...inputs];
  while (stack.length) { const u = stack.pop(); if (outputs.has(u)) return true; for (const v of adj.get(u) ?? []) if (!seen.has(v)) { seen.add(v); stack.push(v); } }
  return false;
}

/** Types that carry their input width through unchanged. */
const PASSTHROUGH = new Set([
  ...NONLINEARITIES, ...NORMS,
  'dropout', 'softmax', 'residual', 'output',
]);

/** Width-bearing families. Attention declares one width for both sides: its
 *  embedDim is simultaneously what it consumes and what it emits. */
const ATTENTION_LIKE = new Set([
  'multiHeadAttention', 'attention', 'selfAttention', 'causalAttention',
  'groupedQueryAttention', 'transformerBlock', 'mla', 'mixtureOfExperts',
]);

const attnWidth = p => num(p, 'embedDim') || num(p, 'hiddenDim') || num(p, 'dModel') || null;

/** The width a node emits, or null when it is not statically known. */
function declaredOutWidth(c) {
  const p = c.params ?? {};
  if (ATTENTION_LIKE.has(c.type)) return attnWidth(p);
  switch (c.type) {
    case 'linear': return num(p, 'outFeatures') || null;
    case 'embedding': return num(p, 'embeddingDim') || null;
    case 'conv1d': case 'conv2d': case 'conv3d': return num(p, 'outChannels') || null;
    default: return null;
  }
}

/** The input width a node explicitly declares, or null when it declares none. */
function declaredInWidth(c) {
  const p = c.params ?? {};
  if (ATTENTION_LIKE.has(c.type)) return attnWidth(p);
  switch (c.type) {
    case 'linear': return num(p, 'inFeatures') || null;
    case 'conv1d': case 'conv2d': case 'conv3d': return num(p, 'inChannels') || null;
    default: return null;
  }
}

/**
 * Propagate feature widths along the graph and report interface mismatches:
 * Algorithm 1's shapeMismatch check. Deliberately conservative — an unknown
 * upstream width (an `input` node's raw shape, an untyped layer) yields null
 * and suppresses the check downstream, so a reference solution can never be
 * failed by a width this rubric merely could not infer. O(|V| + |E|).
 */
export function propagateWidths(model) {
  const byId = new Map(model.components.map(c => [c.id, c]));
  const parents = new Map(model.components.map(c => [c.id, []]));
  const children = new Map(model.components.map(c => [c.id, []]));
  const indeg = new Map(model.components.map(c => [c.id, 0]));
  for (const cn of model.connections) {
    if (!byId.has(cn.from) || !byId.has(cn.to)) continue;
    parents.get(cn.to).push(cn.from);
    children.get(cn.from).push(cn.to);
    indeg.set(cn.to, indeg.get(cn.to) + 1);
  }
  // Kahn order; nodes inside a cycle are simply never visited (a cycle is a
  // separate defect, and we do not want to report a width artifact for it).
  const queue = [...indeg.entries()].filter(([, d]) => d === 0).map(([id]) => id);
  const deg = new Map(indeg);
  const width = new Map();
  const mismatches = [];
  while (queue.length) {
    const id = queue.shift();
    const c = byId.get(id);
    const pw = parents.get(id).map(p => width.get(p) ?? null);
    const known = pw.filter(w => typeof w === 'number' && w > 0);
    let inW = null;
    if (c.type === 'concatenate') {
      // Only defined once every branch is known; a partial sum would be wrong.
      if (pw.length > 0 && known.length === pw.length) inW = known.reduce((a, b) => a + b, 0);
    } else if (known.length) {
      const uniq = [...new Set(known)];
      if (uniq.length > 1) {
        mismatches.push(`${c.name}: merges parents of differing widths (${uniq.join(' vs ')})`);
      }
      inW = uniq[0];
    }
    const dIn = declaredInWidth(c);
    if (dIn !== null && inW !== null && dIn !== inW) {
      mismatches.push(`${c.name}: ${c.type} declares input width ${dIn} but upstream emits ${inW}`);
    }
    const dOut = declaredOutWidth(c);
    width.set(id, dOut ?? (PASSTHROUGH.has(c.type) || c.type === 'concatenate' ? inW : null));
    for (const v of children.get(id)) {
      deg.set(v, deg.get(v) - 1);
      if (deg.get(v) === 0) queue.push(v);
    }
  }
  return { width, mismatches };
}

/** Hard structural failures — a genuinely broken graph, not a caution. */
export function findBlockers(model) {
  const blockers = [];
  if (model.components.length === 0) { blockers.push('empty graph'); return blockers; }
  for (const c of model.components) {
    const p = c.params ?? {};
    const heads = num(p, 'numHeads');
    if (heads > 0) {
      const d = num(p, 'embedDim') || num(p, 'hiddenDim') || num(p, 'dModel');
      if (d > 0 && d % heads !== 0) blockers.push(`${c.name}: embedDim ${d} not divisible by numHeads ${heads}`);
      const kv = num(p, 'numKVHeads');
      if (kv > 0 && heads % kv !== 0) blockers.push(`${c.name}: numHeads ${heads} not divisible by numKVHeads ${kv}`);
    }
    const experts = num(p, 'numExperts');
    const topK = num(p, 'topK');
    if (experts > 0 && topK > 0 && topK > experts) blockers.push(`${c.name}: topK ${topK} > numExperts ${experts}`);
  }
  if (!inputReachesOutput(model)) blockers.push('input does not reach output (disconnected)');
  // A severed middle layer leaves an orphan whose tower is dead even though
  // some other path still joins an input to an output, so the reachability
  // check above cannot see it.
  const indeg = new Map(model.components.map(c => [c.id, 0]));
  for (const cn of model.connections) if (indeg.has(cn.to)) indeg.set(cn.to, indeg.get(cn.to) + 1);
  for (const c of model.components) {
    if (c.type !== 'input' && indeg.get(c.id) === 0) blockers.push(`${c.name}: no incoming connection (orphaned)`);
  }
  for (const m of propagateWidths(model).mismatches) blockers.push(m);
  return blockers;
}

/**
 * Transparent 0..100 score. Broken graphs score low; valid graphs earn points
 * for depth, nonlinearities, normalization, and respecting a param budget.
 */
export function scoreModel(model, budget = {}) {
  const blockers = findBlockers(model);
  const params = estimateParams(model);
  if (blockers.length) {
    return { score: Math.max(0, 20 - 5 * blockers.length), blockers, params };
  }
  const types = new Set(model.components.map(c => c.type));
  const hasWeighted = model.components.some(c => WEIGHTED.has(c.type));
  let score = 40; // valid baseline
  // Depth: reward non-trivial graphs, saturating at ~12 components.
  score += Math.min(model.components.length, 12) / 12 * 25;
  // Nonlinearity present when there are weighted layers to separate.
  if (!hasWeighted || [...types].some(t => NONLINEARITIES.has(t))) score += 15;
  // Normalization.
  if ([...types].some(t => NORMS.has(t))) score += 10;
  // Budget respect.
  if (typeof budget.maxParams === 'number' && budget.maxParams > 0) {
    score += params <= budget.maxParams ? 10 : Math.max(0, 10 - (params / budget.maxParams - 1) * 10);
  } else {
    score += 10;
  }
  return { score: Math.round(Math.min(100, score)), blockers, params };
}

// ─── Grader ──────────────────────────────────────────────────────────────────

export function gradeTask(task, model, actionCount = 0, actionTypes = []) {
  const c = task.constraints ?? {};
  const { score, blockers, params } = scoreModel(model, task.budget ?? {});
  const failures = [];
  const present = new Set(model.components.map(comp => comp.type));

  if (c.forbidBlockers && blockers.length) failures.push(`structural blocker: ${blockers[0]}`);
  if (typeof c.minScore === 'number' && score < c.minScore) failures.push(`score ${score} < min ${c.minScore}`);
  if (typeof c.maxParams === 'number' && params > c.maxParams) failures.push(`params ${params} > budget ${c.maxParams}`);
  if (typeof c.minParams === 'number' && params < c.minParams) failures.push(`params ${params} < required minimum ${c.minParams}`);
  if (typeof c.maxKvBytesPerToken === 'number') {
    const kv = kvBytesPerToken(model);
    if (kv > c.maxKvBytesPerToken) failures.push(`KV ${kv} bytes/token > budget ${c.maxKvBytesPerToken}`);
  }
  if (typeof c.minComponents === 'number' && model.components.length < c.minComponents) failures.push(`${model.components.length} components < min ${c.minComponents}`);
  for (const t of c.mustContainTypes ?? []) if (!present.has(t)) failures.push(`missing layer type "${t}"`);
  if (c.mustContainTypesAny && !c.mustContainTypesAny.some(t => present.has(t))) failures.push(`needs one of: ${c.mustContainTypesAny.join(', ')}`);
  if (c.mustReachOutput && !inputReachesOutput(model)) failures.push('input does not reach output');
  if (typeof c.maxActions === 'number' && actionCount > c.maxActions) failures.push(`${actionCount} actions > max ${c.maxActions}`);
  // Repair tasks forbid replace_model / clear_canvas so a wholesale rebuild
  // can't masquerade as the asked-for surgical fix.
  for (const t of c.forbidActionTypes ?? []) if (actionTypes.includes(t)) failures.push(`used forbidden action "${t}"`);

  return { taskId: task.id, pass: failures.length === 0, score, params, blockers, failures };
}

/**
 * Failure taxonomy: map a grader failure string (or an ERROR string from the
 * harness) to a stable category. Categories make model comparisons legible:
 * "60% of grok's failures are divisibility" says more than a pass rate.
 */
export function categorizeFailure(failure) {
  const f = String(failure).toLowerCase();
  if (f.includes('no json object') || f.includes('"actions" array')) return 'parse-error';
  if (f.includes('not divisible')) return 'attention-divisibility';
  if (f.includes('bytes/token > budget')) return 'kv-over-budget';
  if (f.includes('topk') && f.includes('numexperts')) return 'moe-routing';
  if (f.includes('does not reach output') || f.includes('disconnected') || f.includes('empty graph')) return 'connectivity';
  if (f.includes('> budget')) return 'over-budget';
  if (f.includes('< required minimum')) return 'under-band';
  if (f.includes('missing layer type') || f.includes('needs one of')) return 'missing-layer-type';
  if (f.includes('forbidden action')) return 'forbidden-action';
  if (f.includes('actions > max')) return 'action-limit';
  if (f.includes('components < min')) return 'too-shallow';
  if (f.includes('score')) return 'low-score';
  return 'other';
}

/** Render a graph as the compact text a language-model policy reads. */
export function serializeModel(model) {
  const byId = new Map(model.components.map(c => [c.id, c]));
  const lines = [`Model: ${model.name}`, 'Components:'];
  if (!model.components.length) lines.push('  (empty)');
  for (const c of model.components) {
    const p = c.params && Object.keys(c.params).length ? ` params=${JSON.stringify(c.params)}` : '';
    lines.push(`  - ${c.name} (${c.type})${p}`);
  }
  lines.push('Connections:');
  if (!model.connections.length) lines.push('  (none)');
  for (const cn of model.connections) lines.push(`  - ${byId.get(cn.from)?.name ?? cn.from} -> ${byId.get(cn.to)?.name ?? cn.to}`);
  return lines.join('\n');
}
