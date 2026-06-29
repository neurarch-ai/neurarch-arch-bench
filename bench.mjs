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
      default: break;
    }
  }
  return sum;
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
  }
  if (!inputReachesOutput(model)) blockers.push('input does not reach output (disconnected)');
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

export function gradeTask(task, model, actionCount = 0) {
  const c = task.constraints ?? {};
  const { score, blockers, params } = scoreModel(model, task.budget ?? {});
  const failures = [];
  const present = new Set(model.components.map(comp => comp.type));

  if (c.forbidBlockers && blockers.length) failures.push(`structural blocker: ${blockers[0]}`);
  if (typeof c.minScore === 'number' && score < c.minScore) failures.push(`score ${score} < min ${c.minScore}`);
  if (typeof c.maxParams === 'number' && params > c.maxParams) failures.push(`params ${params} > budget ${c.maxParams}`);
  if (typeof c.minComponents === 'number' && model.components.length < c.minComponents) failures.push(`${model.components.length} components < min ${c.minComponents}`);
  for (const t of c.mustContainTypes ?? []) if (!present.has(t)) failures.push(`missing layer type "${t}"`);
  if (c.mustContainTypesAny && !c.mustContainTypesAny.some(t => present.has(t))) failures.push(`needs one of: ${c.mustContainTypesAny.join(', ')}`);
  if (c.mustReachOutput && !inputReachesOutput(model)) failures.push('input does not reach output');
  if (typeof c.maxActions === 'number' && actionCount > c.maxActions) failures.push(`${actionCount} actions > max ${c.maxActions}`);

  return { taskId: task.id, pass: failures.length === 0, score, params, blockers, failures };
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
