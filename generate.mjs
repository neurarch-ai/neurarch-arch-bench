/**
 * generate — procedurally synthesize benchmark tasks from templates.
 *
 * Twelve hand-authored tasks are a seed, not a benchmark. This generator mints
 * an arbitrarily large split deterministically from a seed, so a held-out
 * split can be created on demand and need never appear on the public web
 * (contamination resistance), and so an RL trainer has an effectively
 * unlimited task stream with verifiable rewards.
 *
 * Eight families:
 *
 *  design-from-spec (start = empty input->output stub):
 *   1. dense classifier        2. dense autoencoder
 *   3. conv image classifier   4. transformer encoder
 *   5. GQA encoder (numHeads % numKVHeads divisibility)
 *  edit-in-place (start = full graph; replace_model / clear_canvas forbidden):
 *   6. repair a broken attention config (start carries a real blocker)
 *   7. trim an oversized MLP under a parameter budget
 *   8. insert normalization after every hidden linear
 *
 * Every generated case ships its own reference solution graded by the same
 * `gradeTask` as the hand-authored set; the test suite asserts each reference
 * passes (no unsatisfiable tasks) and each edit-in-place start fails untouched
 * (no vacuous tasks). Zero dependencies, plain ESM, deterministic PRNG.
 */

// ─── Deterministic PRNG (mulberry32) ─────────────────────────────────────────
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

const pick = (r, xs) => xs[Math.floor(r() * xs.length)];

// ─── Graph builders ──────────────────────────────────────────────────────────

/** Hydrate an input -> output stub of a given input shape. */
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
  const idOf = new Map(components.map(c => [c.name, c.id]));
  const connections = edges.map(([from, to], i) => ({ id: `c${i}`, from: idOf.get(from), to: idOf.get(to) }));
  const byId = new Map(components.map(c => [c.id, c]));
  for (const cn of connections) {
    byId.get(cn.from)?.outputs.push(cn.to);
    byId.get(cn.to)?.inputs.push(cn.from);
  }
  return { name: id, components, connections };
}

/** Chain nodes into a straight input->...->output MLP spec. */
function mlpNodes(D, widths, C) {
  const nodes = [{ componentType: 'input', name: 'input', params: { shape: [1, D] } }];
  const edges = [];
  let prev = 'input';
  let inDim = D;
  widths.forEach((w, k) => {
    nodes.push({ componentType: 'linear', name: `fc${k + 1}`, params: { inFeatures: inDim, outFeatures: w } });
    edges.push([prev, `fc${k + 1}`]);
    nodes.push({ componentType: 'relu', name: `act${k + 1}`, params: {} });
    edges.push([`fc${k + 1}`, `act${k + 1}`]);
    prev = `act${k + 1}`;
    inDim = w;
  });
  nodes.push({ componentType: 'linear', name: 'head', params: { inFeatures: inDim, outFeatures: C } });
  edges.push([prev, 'head']);
  nodes.push({ componentType: 'output', name: 'output', params: {} });
  edges.push(['head', 'output']);
  return { nodes, edges };
}

const EDIT_FORBIDDEN = ['replace_model', 'clear_canvas'];

// ─── Design-from-spec families ───────────────────────────────────────────────

function genClassifier(i, r) {
  const D = pick(r, [16, 32, 48, 64, 128]);
  const C = pick(r, [2, 3, 10]);
  const depth = pick(r, [2, 3, 4]);
  const widths = Array.from({ length: depth }, () => pick(r, [32, 64, 128, 256]));

  const comps = [{ componentType: 'input', name: 'input', params: { shape: [1, D] } }];
  const conns = [];
  let prev = 'input';
  let inDim = D;
  widths.forEach((w, k) => {
    comps.push({ componentType: 'linear', name: `fc${k + 1}`, params: { inFeatures: inDim, outFeatures: w } });
    conns.push({ from: prev, to: `fc${k + 1}` });
    comps.push({ componentType: 'relu', name: `act${k + 1}`, params: {} });
    conns.push({ from: `fc${k + 1}`, to: `act${k + 1}` });
    prev = `act${k + 1}`;
    inDim = w;
  });
  comps.push({ componentType: 'linear', name: 'head', params: { inFeatures: inDim, outFeatures: C } });
  conns.push({ from: prev, to: 'head' });
  comps.push({ componentType: 'output', name: 'output', params: {} });
  conns.push({ from: 'head', to: 'output' });

  return {
    task: {
      id: `gen-mlp-${i}`,
      spec: `Design a dense classifier for a ${D}-feature input with ${C} output classes. Use ${depth} hidden linear layers with ReLU and a linear head. Keep it under 10M params.`,
      budget: { maxParams: 10_000_000 },
      constraints: {
        forbidBlockers: true, minScore: 50, maxParams: 10_000_000,
        mustContainTypes: ['linear', 'relu'], minComponents: comps.length, mustReachOutput: true,
      },
    },
    start: emptyStub(`gen-mlp-${i}`, [1, D]),
    reference: [{ type: 'replace_model', components: comps, connections: conns }],
  };
}

function genAutoencoder(i, r) {
  const D = pick(r, [256, 512, 784]);
  const hidden = pick(r, [128, 256]);
  const bottleneck = pick(r, [16, 32, 64]);
  const dims = [D, hidden, bottleneck, hidden, D];

  const comps = [{ componentType: 'input', name: 'input', params: { shape: [1, D] } }];
  const conns = [];
  let prev = 'input';
  for (let k = 0; k < dims.length - 1; k++) {
    const name = k < 2 ? `enc${k + 1}` : `dec${k - 1}`;
    comps.push({ componentType: 'linear', name, params: { inFeatures: dims[k], outFeatures: dims[k + 1] } });
    conns.push({ from: prev, to: name });
    prev = name;
  }
  comps.push({ componentType: 'output', name: 'output', params: {} });
  conns.push({ from: prev, to: 'output' });

  return {
    task: {
      id: `gen-ae-${i}`,
      spec: `Design a dense autoencoder for ${D}-dim input: encode down to a ${bottleneck}-dim bottleneck and decode back to ${D}. Keep it valid and connected, under 20M params.`,
      budget: { maxParams: 20_000_000 },
      constraints: {
        forbidBlockers: true, minScore: 50, maxParams: 20_000_000,
        mustContainTypes: ['linear'], minComponents: comps.length, mustReachOutput: true,
      },
    },
    start: emptyStub(`gen-ae-${i}`, [1, D]),
    reference: [{ type: 'replace_model', components: comps, connections: conns }],
  };
}

function genConvClassifier(i, r) {
  const side = pick(r, [28, 32]);
  const C = pick(r, [2, 10]);
  const stages = pick(r, [2, 3]);

  const comps = [{ componentType: 'input', name: 'input', params: { shape: [1, 3, side, side] } }];
  const conns = [];
  let prev = 'input';
  let inCh = 3;
  for (let k = 0; k < stages; k++) {
    const outCh = pick(r, [8, 16, 32]);
    comps.push({ componentType: 'conv2d', name: `conv${k + 1}`, params: { inChannels: inCh, outChannels: outCh, kernelSize: 3 } });
    conns.push({ from: prev, to: `conv${k + 1}` });
    comps.push({ componentType: 'relu', name: `act${k + 1}`, params: {} });
    conns.push({ from: `conv${k + 1}`, to: `act${k + 1}` });
    prev = `act${k + 1}`;
    inCh = outCh;
  }
  comps.push({ componentType: 'linear', name: 'head', params: { inFeatures: inCh, outFeatures: C } });
  conns.push({ from: prev, to: 'head' });
  comps.push({ componentType: 'output', name: 'output', params: {} });
  conns.push({ from: 'head', to: 'output' });

  return {
    task: {
      id: `gen-cnn-${i}`,
      spec: `Design a convolutional classifier for a 3x${side}x${side} image with ${C} classes. Use ${stages} conv layers with ReLU and a linear head. Keep it under 50M params.`,
      budget: { maxParams: 50_000_000 },
      constraints: {
        forbidBlockers: true, minScore: 50, maxParams: 50_000_000,
        mustContainTypes: ['conv2d', 'linear'], minComponents: comps.length, mustReachOutput: true,
      },
    },
    start: emptyStub(`gen-cnn-${i}`, [1, 3, side, side]),
    reference: [{ type: 'replace_model', components: comps, connections: conns }],
  };
}

function genTransformerEncoder(i, r) {
  const seq = pick(r, [64, 128]);
  const D = pick(r, [128, 256]); // divisible by every head count below
  const heads = pick(r, [4, 8]);
  const blocks = pick(r, [1, 2]);
  const C = pick(r, [2, 5]);
  const vocab = pick(r, [20_000, 30_000]);

  const comps = [
    { componentType: 'input', name: 'input', params: { shape: [1, seq] } },
    { componentType: 'embedding', name: 'embed', params: { numEmbeddings: vocab, embeddingDim: D } },
  ];
  const conns = [{ from: 'input', to: 'embed' }];
  let prev = 'embed';
  for (let k = 0; k < blocks; k++) {
    comps.push({ componentType: 'multiHeadAttention', name: `attn${k + 1}`, params: { embedDim: D, numHeads: heads } });
    conns.push({ from: prev, to: `attn${k + 1}` });
    prev = `attn${k + 1}`;
  }
  comps.push({ componentType: 'linear', name: 'head', params: { inFeatures: D, outFeatures: C } });
  conns.push({ from: prev, to: 'head' });
  comps.push({ componentType: 'output', name: 'output', params: {} });
  conns.push({ from: 'head', to: 'output' });

  return {
    task: {
      id: `gen-txf-${i}`,
      spec: `Design a transformer encoder for ${seq}-token sequences with ${C} output classes. Token embedding, ${blocks} multi-head attention block(s) with a valid head configuration, then a linear head. Keep it under 60M params.`,
      budget: { maxParams: 60_000_000 },
      constraints: {
        forbidBlockers: true, minScore: 50, maxParams: 60_000_000,
        mustContainTypes: ['embedding', 'multiHeadAttention', 'linear'], minComponents: comps.length, mustReachOutput: true,
      },
    },
    start: emptyStub(`gen-txf-${i}`, [1, seq]),
    reference: [{ type: 'replace_model', components: comps, connections: conns }],
  };
}

function genGQAEncoder(i, r) {
  const seq = pick(r, [64, 128]);
  const D = pick(r, [128, 256]); // divisible by 8
  const kv = pick(r, [2, 4]);    // divides 8
  const blocks = pick(r, [1, 2]);
  const C = pick(r, [2, 5]);
  const vocab = pick(r, [20_000, 30_000]);

  const comps = [
    { componentType: 'input', name: 'input', params: { shape: [1, seq] } },
    { componentType: 'embedding', name: 'embed', params: { numEmbeddings: vocab, embeddingDim: D } },
  ];
  const conns = [{ from: 'input', to: 'embed' }];
  let prev = 'embed';
  for (let k = 0; k < blocks; k++) {
    comps.push({ componentType: 'groupedQueryAttention', name: `attn${k + 1}`, params: { embedDim: D, numHeads: 8, numKVHeads: kv } });
    conns.push({ from: prev, to: `attn${k + 1}` });
    comps.push({ componentType: 'layerNorm', name: `norm${k + 1}`, params: { normalizedShape: [D] } });
    conns.push({ from: `attn${k + 1}`, to: `norm${k + 1}` });
    prev = `norm${k + 1}`;
  }
  comps.push({ componentType: 'linear', name: 'head', params: { inFeatures: D, outFeatures: C } });
  conns.push({ from: prev, to: 'head' });
  comps.push({ componentType: 'output', name: 'output', params: {} });
  conns.push({ from: 'head', to: 'output' });

  return {
    task: {
      id: `gen-gqa-${i}`,
      spec: `Design a grouped-query-attention encoder for ${seq}-token sequences with ${C} output classes: token embedding, ${blocks} GQA block(s) each followed by layer normalization, then a linear head. numHeads must be divisible by numKVHeads. Keep it under 60M params.`,
      budget: { maxParams: 60_000_000 },
      constraints: {
        forbidBlockers: true, minScore: 50, maxParams: 60_000_000,
        mustContainTypes: ['embedding', 'groupedQueryAttention', 'layerNorm', 'linear'],
        minComponents: comps.length, mustReachOutput: true,
      },
    },
    start: emptyStub(`gen-gqa-${i}`, [1, seq]),
    reference: [{ type: 'replace_model', components: comps, connections: conns }],
  };
}

// ─── Edit-in-place families ──────────────────────────────────────────────────

function genRepairAttention(i, r) {
  const seq = pick(r, [64, 128]);
  const D = pick(r, [96, 128, 192, 256]);
  const C = pick(r, [2, 5]);
  const vocab = pick(r, [20_000, 30_000]);
  const gqa = r() < 0.5;

  // Broken by construction: none of these head counts divide any D above /
  // none of these KV counts divide 8.
  const badHeads = pick(r, [5, 7, 11]);
  const badKV = pick(r, [3, 5, 7]);
  const fixHeads = pick(r, [4, 8]); // divides every D above
  const fixKV = pick(r, [2, 4]);    // divides 8

  const attn = gqa
    ? { componentType: 'groupedQueryAttention', name: 'attn', params: { embedDim: D, numHeads: 8, numKVHeads: badKV } }
    : { componentType: 'multiHeadAttention', name: 'attn', params: { embedDim: D, numHeads: badHeads } };
  const nodes = [
    { componentType: 'input', name: 'input', params: { shape: [1, seq] } },
    { componentType: 'embedding', name: 'embed', params: { numEmbeddings: vocab, embeddingDim: D } },
    attn,
    { componentType: 'linear', name: 'head', params: { inFeatures: D, outFeatures: C } },
    { componentType: 'output', name: 'output', params: {} },
  ];
  const edges = [['input', 'embed'], ['embed', 'attn'], ['attn', 'head'], ['head', 'output']];
  const defect = gqa
    ? `numHeads (8) is not divisible by numKVHeads (${badKV})`
    : `embedDim (${D}) is not divisible by numHeads (${badHeads})`;
  return {
    task: {
      id: `gen-fix-${i}`,
      spec: `This encoder fails validation: ${defect}. Repair the attention configuration in place with at most 2 actions. Do not rebuild the model from scratch.`,
      constraints: {
        forbidBlockers: true, minScore: 50, mustReachOutput: true,
        minComponents: nodes.length, maxActions: 2, forbidActionTypes: EDIT_FORBIDDEN,
      },
    },
    start: buildGraph(`gen-fix-${i}`, nodes, edges),
    reference: [
      gqa
        ? { type: 'update_params', name: 'attn', params: { numKVHeads: fixKV } }
        : { type: 'update_params', name: 'attn', params: { numHeads: fixHeads } },
    ],
  };
}

function genBudgetTrim(i, r) {
  const D = pick(r, [64, 128]);
  const C = pick(r, [2, 10]);
  const depth = pick(r, [2, 3]);
  const bigW = pick(r, [2048, 4096]);
  const smallW = pick(r, [128, 256]);
  const budget = 2_000_000;

  const { nodes, edges } = mlpNodes(D, Array.from({ length: depth }, () => bigW), C);

  const fixes = [];
  let inDim = D;
  for (let k = 1; k <= depth; k++) {
    fixes.push({ type: 'update_params', name: `fc${k}`, params: { inFeatures: inDim, outFeatures: smallW } });
    inDim = smallW;
  }
  fixes.push({ type: 'update_params', name: 'head', params: { inFeatures: inDim, outFeatures: C } });

  return {
    task: {
      id: `gen-trim-${i}`,
      spec: `This ${depth}-hidden-layer MLP uses ${bigW}-wide layers and blows a 2M parameter budget. Shrink the widths in place so total params fit the budget, keeping all ${depth} hidden layers and consistent in/out features. Do not rebuild the model from scratch.`,
      budget: { maxParams: budget },
      constraints: {
        forbidBlockers: true, minScore: 50, maxParams: budget,
        mustContainTypes: ['linear', 'relu'], minComponents: nodes.length, mustReachOutput: true,
        maxActions: depth + 2, forbidActionTypes: EDIT_FORBIDDEN,
      },
    },
    start: buildGraph(`gen-trim-${i}`, nodes, edges),
    reference: fixes,
  };
}

function genInsertNorm(i, r) {
  const D = pick(r, [32, 64, 128]);
  const C = pick(r, [2, 3, 10]);
  const depth = pick(r, [2, 3]);
  const widths = Array.from({ length: depth }, () => pick(r, [64, 128, 256]));

  const { nodes, edges } = mlpNodes(D, widths, C);
  const inserts = widths.map((w, k) => ({
    type: 'add_component', componentType: 'batchNorm1d', name: `bn${k + 1}`,
    afterName: `fc${k + 1}`, params: { numFeatures: w },
  }));

  return {
    task: {
      id: `gen-norm-${i}`,
      spec: `This MLP trains unstably. Insert a batchNorm1d after each of the ${depth} hidden linear layers (between the linear and its activation), with numFeatures matching each layer's output width. Do not rebuild the model from scratch.`,
      constraints: {
        forbidBlockers: true, minScore: 50, mustContainTypes: ['batchNorm1d'],
        minComponents: nodes.length + depth, mustReachOutput: true,
        maxActions: depth + 1, forbidActionTypes: EDIT_FORBIDDEN,
      },
    },
    start: buildGraph(`gen-norm-${i}`, nodes, edges),
    reference: inserts,
  };
}

/** Two-tower retrieval: two inputs -> two equal-width MLP towers ->
 *  concatenate -> scoring head. Exercises multi-input graphs. */
function genTwoTower(i, r) {
  const Du = pick(r, [32, 64, 128]);
  const Di = pick(r, [32, 64, 128]);
  const W = pick(r, [64, 128, 256]);

  const comps = [
    { componentType: 'input', name: 'user_input', params: { shape: [1, Du] } },
    { componentType: 'input', name: 'item_input', params: { shape: [1, Di] } },
    { componentType: 'linear', name: 'user_fc1', params: { inFeatures: Du, outFeatures: W } },
    { componentType: 'relu', name: 'user_act', params: {} },
    { componentType: 'linear', name: 'user_fc2', params: { inFeatures: W, outFeatures: W } },
    { componentType: 'linear', name: 'item_fc1', params: { inFeatures: Di, outFeatures: W } },
    { componentType: 'relu', name: 'item_act', params: {} },
    { componentType: 'linear', name: 'item_fc2', params: { inFeatures: W, outFeatures: W } },
    { componentType: 'concatenate', name: 'merge', params: { dim: -1 } },
    { componentType: 'linear', name: 'head', params: { inFeatures: 2 * W, outFeatures: 1 } },
    { componentType: 'output', name: 'output', params: {} },
  ];
  const conns = [
    { from: 'user_input', to: 'user_fc1' }, { from: 'user_fc1', to: 'user_act' }, { from: 'user_act', to: 'user_fc2' },
    { from: 'item_input', to: 'item_fc1' }, { from: 'item_fc1', to: 'item_act' }, { from: 'item_act', to: 'item_fc2' },
    { from: 'user_fc2', to: 'merge' }, { from: 'item_fc2', to: 'merge' },
    { from: 'merge', to: 'head' }, { from: 'head', to: 'output' },
  ];

  // Two-input start stub.
  const start = {
    name: `gen-tower-${i}`,
    components: [
      { id: 'iu', type: 'input', name: 'user_input', params: { shape: [1, Du] }, inputs: [], outputs: ['o'] },
      { id: 'ii', type: 'input', name: 'item_input', params: { shape: [1, Di] }, inputs: [], outputs: ['o'] },
      { id: 'o', type: 'output', name: 'output', params: {}, inputs: ['iu', 'ii'], outputs: [] },
    ],
    connections: [
      { id: 'c0', from: 'iu', to: 'o' },
      { id: 'c1', from: 'ii', to: 'o' },
    ],
  };

  return {
    task: {
      id: `gen-tower-${i}`,
      spec: `Design a two-tower retrieval scorer: a ${Du}-feature user input and a ${Di}-feature item input, each through its own 2-layer MLP tower ending at width ${W}, concatenated and scored by a linear head. Both tower outputs must have identical shape for the merge. Keep it under 10M params.`,
      budget: { maxParams: 10_000_000 },
      constraints: {
        forbidBlockers: true, minScore: 50, maxParams: 10_000_000,
        mustContainTypes: ['linear', 'relu', 'concatenate'],
        minComponents: comps.length, mustReachOutput: true,
      },
    },
    start,
    reference: [{ type: 'replace_model', components: comps, connections: conns }],
  };
}

/** Grow-to-band: the start MLP is far too small; widen it in place so total
 *  params land inside [min, max]. The inverse of genBudgetTrim, and the first
 *  family with a two-sided budget. */
function genParamGrow(i, r) {
  const D = pick(r, [32, 64]);
  const C = pick(r, [2, 10]);
  const tinyW = pick(r, [8, 16]);
  const bigW = pick(r, [768, 1024]);
  const minP = 400_000;
  const maxP = 4_000_000;

  const { nodes, edges } = mlpNodes(D, [tinyW, tinyW], C);

  const fixes = [
    { type: 'update_params', name: 'fc1', params: { inFeatures: D, outFeatures: bigW } },
    { type: 'update_params', name: 'fc2', params: { inFeatures: bigW, outFeatures: bigW } },
    { type: 'update_params', name: 'head', params: { inFeatures: bigW, outFeatures: C } },
  ];

  return {
    task: {
      id: `gen-grow-${i}`,
      spec: `This 2-hidden-layer MLP is far too small for its workload. Widen the hidden layers in place so total params land between ${minP / 1000}k and ${maxP / 1e6}M, keeping consistent in/out features. Do not rebuild the model from scratch.`,
      budget: { maxParams: maxP },
      constraints: {
        forbidBlockers: true, minScore: 50, minParams: minP, maxParams: maxP,
        mustContainTypes: ['linear', 'relu'], minComponents: nodes.length, mustReachOutput: true,
        maxActions: 4, forbidActionTypes: EDIT_FORBIDDEN,
      },
    },
    start: buildGraph(`gen-grow-${i}`, nodes, edges),
    reference: fixes,
  };
}

const FAMILIES = [
  genClassifier, genAutoencoder, genConvClassifier, genTransformerEncoder,
  genGQAEncoder, genRepairAttention, genBudgetTrim, genInsertNorm,
  genTwoTower, genParamGrow,
];

/**
 * Deterministically generate `count` cases from `seed`. Same (count, seed)
 * always yields the identical split — this is what makes held-out splits and
 * reproducible RL training data possible.
 *
 * Returns [{ task, start, reference }]: the task (spec + constraints), a fully
 * hydrated start graph, and a known-good reference action sequence.
 */
export function generateCases(count, seed = 1) {
  const r = rng(seed);
  const cases = [];
  for (let i = 0; i < count; i++) cases.push(FAMILIES[i % FAMILIES.length](i, r));
  return cases;
}
