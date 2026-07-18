/**
 * generate-frontier — the frontier tier: tasks shaped like the models labs
 * actually design and serve in 2026 (MoE feed-forwards, latent-attention KV
 * compression, long-context serving budgets), graded by the same deterministic
 * verifier as the core benchmark.
 *
 * This is a SEPARATE, OPT-IN tier (env-server `split=frontier`); the core
 * ten-family generator and every published number on it are untouched.
 *
 * Three families, round-robin:
 *   1. fmoe — design a Mixture-of-Experts transformer block (router validity:
 *      topK must not exceed numExperts; param band around the reference).
 *   2. fmla — design a long-context encoder under a KV-cache budget that full
 *      attention cannot meet; multi-head latent attention compresses the cache.
 *   3. fkv  — edit-in-place: retrofit an existing MHA encoder to a KV budget
 *      by converting the attention to grouped-query attention (replace_model
 *      forbidden, action-economy capped).
 *
 * Every case ships a reference solution graded by the same `gradeTask`;
 * frontier.test.ts asserts satisfiability, non-vacuity, and anti-gaming.
 */
import { scoreModel } from './bench.mjs';

export const FRONTIER_VERSION = 1;

// ─── Deterministic PRNG (mulberry32, same as the core generator) ─────────────
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
    { id: 'o', type: 'output', name: 'output', params: {}, inputs: ['i'], outputs: [] },
  ];
  return { name: id, components, connections: [{ id: 'c0', from: 'i', to: 'o' }] };
}

/** Hydrate a node list + name-edges into a full graph with rebound ports. */
function buildGraph(id, nodes, edges) {
  const components = nodes.map((n, i) => ({
    id: `n${i}`, type: n.componentType, name: n.name, params: { ...n.params }, inputs: [], outputs: [],
  }));
  const byName = new Map(components.map(c => [c.name, c]));
  const connections = edges.map((e, i) => ({ id: `c${i}`, from: byName.get(e.from).id, to: byName.get(e.to).id }));
  for (const cn of connections) {
    byName.get(components.find(c => c.id === cn.from).name).outputs.push(cn.to);
    byName.get(components.find(c => c.id === cn.to).name).inputs.push(cn.from);
  }
  return { name: id, components, connections };
}

/** Chain nodes[0]->nodes[1]->... into (comps, conns) for replace_model refs. */
function chain(comps) {
  const conns = [];
  for (let k = 0; k + 1 < comps.length; k++) conns.push({ from: comps[k].name, to: comps[k + 1].name });
  return conns;
}

// ─── Family 1: MoE transformer block (design-from-spec) ──────────────────────
function genMoe(i, r) {
  const heads = pick(r, [4, 8, 16]);
  const D = heads * randInt(r, 16, 64);
  const E = pick(r, [8, 16, 32, 64]);
  const k = randInt(r, 1, 4); // k <= 4 <= E by construction
  const H = D * randInt(r, 2, 4);
  const C = randInt(r, 2, 1000);

  const comps = [
    { componentType: 'input', name: 'input', params: { shape: [1, D] } },
    { componentType: 'multiHeadAttention', name: 'attn', params: { embedDim: D, numHeads: heads } },
    { componentType: 'layerNorm', name: 'ln1', params: { normalizedShape: D } },
    { componentType: 'mixtureOfExperts', name: 'moe', params: { embedDim: D, hiddenDim: H, numExperts: E, topK: k } },
    { componentType: 'layerNorm', name: 'ln2', params: { normalizedShape: D } },
    { componentType: 'linear', name: 'head', params: { inFeatures: D, outFeatures: C } },
    { componentType: 'output', name: 'output', params: {} },
  ];
  const conns = chain(comps);
  const measured = scoreModel(buildGraph(`m`, comps, conns)).params;

  return {
    task: {
      id: `gen-fmoe-${i}`,
      spec: `Design a Mixture-of-Experts transformer block for a ${D}-dim stream: ${heads}-head self-attention, layer norms, then a mixture-of-experts feed-forward (type "mixtureOfExperts", params embedDim/hiddenDim/numExperts/topK) with ${E} experts of hidden size ${H} routed top-${k}, and a ${D}->${C} linear head. topK must not exceed numExperts. Keep total params within 25% of the canonical layout.`,
      constraints: {
        forbidBlockers: true, minScore: 50, mustReachOutput: true,
        mustContainTypes: ['multiHeadAttention', 'mixtureOfExperts', 'layerNorm', 'linear'],
        minComponents: comps.length,
        minParams: Math.floor(measured * 0.75), maxParams: Math.ceil(measured * 1.25),
      },
    },
    start: emptyStub(`gen-fmoe-${i}`, [1, D]),
    reference: [{ type: 'replace_model', components: comps, connections: conns }],
  };
}

// ─── Family 2: latent-attention KV budget (design-from-spec) ─────────────────
function genMla(i, r) {
  const heads = pick(r, [8, 16]);
  const D = heads * randInt(r, 16, 64);
  const latent = 64 * randInt(r, 4, 9);
  const rope = pick(r, [32, 64]);
  const C = randInt(r, 2, 1000);
  // fp16 MLA cache = (latent + rope) * 2 bytes/token; budget adds 25% slack.
  // Full attention would cache 2*D*2 = 4D bytes/token, over budget for all
  // sampled D, so compression is forced, not stylistic.
  const budget = Math.ceil((latent + rope) * 2 * 1.25);

  const comps = [
    { componentType: 'input', name: 'input', params: { shape: [1, D] } },
    { componentType: 'mla', name: 'mla', params: { embedDim: D, numHeads: heads, kvLatentDim: latent, ropeHeadDim: rope } },
    { componentType: 'layerNorm', name: 'ln', params: { normalizedShape: D } },
    { componentType: 'linear', name: 'head', params: { inFeatures: D, outFeatures: C } },
    { componentType: 'output', name: 'output', params: {} },
  ];
  const conns = chain(comps);

  return {
    task: {
      id: `gen-fmla-${i}`,
      spec: `Serve a 128k-token context: the KV cache must stay under ${budget} bytes per token at fp16. Design a ${D}-dim encoder using multi-head latent attention (type "mla", params embedDim/numHeads/kvLatentDim/ropeHeadDim; a ~${latent}-dim latent with ${rope} RoPE dims fits), a layer norm, and a ${D}->${C} linear head. Full ${heads}-head attention would cache ${4 * D} bytes/token and cannot fit.`,
      constraints: {
        forbidBlockers: true, minScore: 50, mustReachOutput: true,
        mustContainTypes: ['mla'], minComponents: comps.length,
        maxKvBytesPerToken: budget,
      },
    },
    start: emptyStub(`gen-fmla-${i}`, [1, D]),
    reference: [{ type: 'replace_model', components: comps, connections: conns }],
  };
}

// ─── Family 3: GQA retrofit under a KV budget (edit-in-place) ────────────────
function genKvRetrofit(i, r) {
  const heads = pick(r, [8, 16, 32]);
  const headDim = randInt(r, 16, 64);
  const D = heads * headDim;
  const g = pick(r, [2, 4]); // divides 8/16/32
  const V = 1000 * randInt(r, 8, 32);
  const C = randInt(r, 2, 500);
  // Start caches 2*D*2 = 4D bytes/token (MHA); GQA at g KV heads caches
  // 4*g*headDim. Budget sits 25% above the GQA cost and far below MHA's.
  const budget = Math.ceil(4 * g * headDim * 1.25);

  const start = buildGraph(`gen-fkv-${i}`, [
    { componentType: 'input', name: 'input', params: { shape: [1, 128] } },
    { componentType: 'embedding', name: 'embed', params: { numEmbeddings: V, embeddingDim: D } },
    { componentType: 'multiHeadAttention', name: 'attn', params: { embedDim: D, numHeads: heads } },
    { componentType: 'layerNorm', name: 'ln', params: { normalizedShape: D } },
    { componentType: 'linear', name: 'head', params: { inFeatures: D, outFeatures: C } },
    { componentType: 'output', name: 'output', params: {} },
  ], [
    { from: 'input', to: 'embed' }, { from: 'embed', to: 'attn' },
    { from: 'attn', to: 'ln' }, { from: 'ln', to: 'head' }, { from: 'head', to: 'output' },
  ]);

  return {
    task: {
      id: `gen-fkv-${i}`,
      spec: `This ${D}-dim encoder (a ${V}-token vocabulary embedding and a ${D}->${C} linear head) must serve a 128k-token context under ${budget} bytes of KV cache per token at fp16, but its ${heads}-head full attention caches ${4 * D} bytes/token. Convert the attention to grouped-query attention in place (keep embedDim ${D} and numHeads ${heads}; choose a numKVHeads that divides ${heads} and fits the budget). At most 3 actions. Do not rebuild the model.`,
      constraints: {
        forbidBlockers: true, minScore: 50, mustReachOutput: true,
        mustContainTypesAny: ['groupedQueryAttention', 'mla'],
        minComponents: start.components.length,
        maxKvBytesPerToken: budget,
        maxActions: 3,
        forbidActionTypes: ['replace_model', 'clear_canvas'],
      },
    },
    start,
    reference: [
      { type: 'add_component', componentType: 'groupedQueryAttention', name: 'gqa', afterName: 'embed', params: { embedDim: D, numHeads: heads, numKVHeads: g } },
      { type: 'delete_component', name: 'attn' },
    ],
  };
}

const FAMILIES = [genMoe, genMla, genKvRetrofit];

export function generateFrontierCases(count, seed = 1) {
  const r = rng(seed);
  const cases = [];
  for (let i = 0; i < count; i++) cases.push(FAMILIES[i % FAMILIES.length](i, r));
  return cases;
}
