/**
 * generate-edge — the edge tier: design under joint on-device budgets.
 *
 * SpaceX-class deployments (satellite constellations, vehicles, hardened
 * compute) serve models under HARD joint budgets: total parameters AND
 * KV-cache bytes per token, simultaneously. This opt-in tier grades exactly
 * that with the same deterministic verifier; the core families, the frontier
 * tier, and every published number are untouched.
 *
 * Two families, round-robin (env-server `split=edge`):
 *   1. eedge   — design-from-spec: an on-device encoder under a param budget
 *                AND a KV budget that full attention cannot meet (GQA fits).
 *   2. eshrink — edit-in-place: an over-budget encoder must be brought under
 *                both budgets in at most 4 surgical actions (vocabulary trim +
 *                GQA swap is the canonical route); replace_model forbidden.
 *
 * Every case ships a reference solution; edge.test.ts asserts satisfiability,
 * non-vacuity, and anti-gaming, mirroring the core and frontier suites.
 */
import { scoreModel, kvBytesPerToken } from './bench.mjs';

export const EDGE_VERSION = 1;

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const randInt = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
const pick = (r, xs) => xs[Math.floor(r() * xs.length)];

function emptyStub(id, shape) {
  const components = [
    { id: 'i', type: 'input', name: 'input', params: { shape }, inputs: [], outputs: ['o'] },
    { id: 'o', type: 'output', name: 'output', params: {}, inputs: [], outputs: [] },
  ];
  return { name: id, components, connections: [{ id: 'c0', from: 'i', to: 'o' }] };
}

function buildGraph(id, nodes, edges) {
  const components = nodes.map((n, i) => ({
    id: `n${i}`, type: n.componentType, name: n.name, params: { ...n.params }, inputs: [], outputs: [],
  }));
  const byName = new Map(components.map(c => [c.name, c]));
  const connections = edges.map((e, i) => ({ id: `c${i}`, from: byName.get(e.from).id, to: byName.get(e.to).id }));
  for (const cn of connections) {
    components.find(c => c.id === cn.from).outputs.push(cn.to);
    components.find(c => c.id === cn.to).inputs.push(cn.from);
  }
  return { name: id, components, connections };
}

function chain(comps) {
  const conns = [];
  for (let k = 0; k + 1 < comps.length; k++) conns.push({ from: comps[k].name, to: comps[k + 1].name });
  return conns;
}

// ─── Family 1: on-device encoder under joint budgets (design-from-spec) ──────
function genEdge(i, r) {
  const heads = pick(r, [4, 8]);
  const headDim = randInt(r, 8, 32);
  const D = heads * headDim;
  const g = 2; // divides 4 and 8
  const V = 1000 * randInt(r, 2, 16);
  const C = randInt(r, 2, 100);

  const comps = [
    { componentType: 'input', name: 'input', params: { shape: [1, 64] } },
    { componentType: 'embedding', name: 'embed', params: { numEmbeddings: V, embeddingDim: D } },
    { componentType: 'groupedQueryAttention', name: 'gqa', params: { embedDim: D, numHeads: heads, numKVHeads: g } },
    { componentType: 'layerNorm', name: 'ln', params: { normalizedShape: D } },
    { componentType: 'linear', name: 'head', params: { inFeatures: D, outFeatures: C } },
    { componentType: 'output', name: 'output', params: {} },
  ];
  const conns = chain(comps);
  const graph = buildGraph('m', comps, conns);
  const paramBudget = Math.ceil(scoreModel(graph).params * 1.2);
  // GQA caches 4*g*headDim bytes/token at fp16; full attention would cache
  // 4*D = heads/g times more, over budget for every sampled config.
  const kvBudget = Math.ceil(4 * g * headDim * 1.25);

  return {
    task: {
      id: `gen-eedge-${i}`,
      spec: `Design an on-device text encoder for a hard edge deployment: a ${V}-token vocabulary embedding into ${D} dims, attention, a norm, and a ${D}->${C} linear head, under BOTH budgets at once: total params <= ${paramBudget} AND KV cache <= ${kvBudget} bytes per token at fp16. Full ${heads}-head attention would cache ${4 * D} bytes/token and cannot fit; grouped-query attention with few KV heads can.`,
      constraints: {
        forbidBlockers: true, minScore: 50, mustReachOutput: true,
        mustContainTypes: ['embedding', 'linear'],
        mustContainTypesAny: ['groupedQueryAttention', 'mla'],
        minComponents: comps.length,
        maxParams: paramBudget,
        maxKvBytesPerToken: kvBudget,
      },
    },
    start: emptyStub(`gen-eedge-${i}`, [1, 64]),
    reference: [{ type: 'replace_model', components: comps, connections: conns }],
  };
}

// ─── Family 2: shrink an over-budget encoder onto the edge (edit-in-place) ───
function genShrink(i, r) {
  const heads = pick(r, [8, 16]);
  const headDim = randInt(r, 16, 48);
  const D = heads * headDim;
  const g = pick(r, [2, 4]);
  const bigV = 1000 * randInt(r, 40, 120);
  const smallV = 1000 * randInt(r, 4, 12);
  const C = randInt(r, 2, 100);

  const start = buildGraph(`gen-eshrink-${i}`, [
    { componentType: 'input', name: 'input', params: { shape: [1, 64] } },
    { componentType: 'embedding', name: 'embed', params: { numEmbeddings: bigV, embeddingDim: D } },
    { componentType: 'multiHeadAttention', name: 'attn', params: { embedDim: D, numHeads: heads } },
    { componentType: 'layerNorm', name: 'ln', params: { normalizedShape: D } },
    { componentType: 'linear', name: 'head', params: { inFeatures: D, outFeatures: C } },
    { componentType: 'output', name: 'output', params: {} },
  ], [
    { from: 'input', to: 'embed' }, { from: 'embed', to: 'attn' },
    { from: 'attn', to: 'ln' }, { from: 'ln', to: 'head' }, { from: 'head', to: 'output' },
  ]);

  // Budgets sit between the shrunk target and the bloated start, so the start
  // fails BOTH budgets and the canonical 3-action repair passes both.
  const target = buildGraph('t', [
    { componentType: 'input', name: 'input', params: { shape: [1, 64] } },
    { componentType: 'embedding', name: 'embed', params: { numEmbeddings: smallV, embeddingDim: D } },
    { componentType: 'groupedQueryAttention', name: 'gqa', params: { embedDim: D, numHeads: heads, numKVHeads: g } },
    { componentType: 'layerNorm', name: 'ln', params: { normalizedShape: D } },
    { componentType: 'linear', name: 'head', params: { inFeatures: D, outFeatures: C } },
    { componentType: 'output', name: 'output', params: {} },
  ], [
    { from: 'input', to: 'embed' }, { from: 'embed', to: 'gqa' },
    { from: 'gqa', to: 'ln' }, { from: 'ln', to: 'head' }, { from: 'head', to: 'output' },
  ]);
  const paramBudget = Math.ceil(scoreModel(target).params * 1.15);
  const kvBudget = Math.ceil(kvBytesPerToken(target) * 1.25);

  return {
    task: {
      id: `gen-eshrink-${i}`,
      spec: `Fit this ${D}-dim encoder onto edge hardware: bring it under BOTH total params <= ${paramBudget} AND KV cache <= ${kvBudget} bytes/token at fp16, editing in place with at most 4 actions. It currently embeds a ${bigV}-token vocabulary (a ${smallV}-token vocabulary suffices for the on-device task) and its ${heads}-head full attention caches ${4 * D} bytes/token. Keep embedDim ${D} and the ${D}->${C} head. Do not rebuild the model.`,
      constraints: {
        forbidBlockers: true, minScore: 50, mustReachOutput: true,
        mustContainTypesAny: ['groupedQueryAttention', 'mla'],
        minComponents: start.components.length,
        maxParams: paramBudget,
        maxKvBytesPerToken: kvBudget,
        maxActions: 4,
        forbidActionTypes: ['replace_model', 'clear_canvas'],
      },
    },
    start,
    reference: [
      { type: 'update_params', name: 'embed', params: { numEmbeddings: smallV } },
      { type: 'add_component', componentType: 'groupedQueryAttention', name: 'gqa', afterName: 'embed', params: { embedDim: D, numHeads: heads, numKVHeads: g } },
      { type: 'delete_component', name: 'attn' },
    ],
  };
}

const FAMILIES = [genEdge, genShrink];

export function generateEdgeCases(count, seed = 1) {
  const r = rng(seed);
  const cases = [];
  for (let i = 0; i < count; i++) cases.push(FAMILIES[i % FAMILIES.length](i, r));
  return cases;
}
