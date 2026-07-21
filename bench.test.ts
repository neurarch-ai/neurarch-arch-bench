import { describe, it, expect } from 'vitest';
// @ts-expect-error — plain ESM module, no type declarations needed for the test.
import { loadBenchmark, loadSolutions, buildFixture, applyActions, gradeTask, scoreModel, findBlockers, inputReachesOutput } from './bench.mjs';

const bench = loadBenchmark();
const solutions = loadSolutions();
const task = (id: string) => {
  const t = bench.tasks.find((x: any) => x.id === id);
  if (!t) throw new Error(`no task ${id}`);
  return t;
};

describe('benchmark integrity', () => {
  it('every task points at a known fixture', () => {
    for (const t of bench.tasks) expect(() => buildFixture(bench, t.start)).not.toThrow();
  });

  it('every task has a reference solution', () => {
    for (const t of bench.tasks) expect(solutions[t.id], `no reference solution for ${t.id}`).toBeTruthy();
  });

  // Well-formedness guard: if a constraint gets over-tightened to where the task
  // is unsolvable, this fails loudly instead of silently making the benchmark
  // impossible. Every task must be passable by its known-good reference.
  it.each(bench.tasks.map((t: any) => t.id))('reference solution passes task: %s', (id: string) => {
    const t = task(id);
    const start = buildFixture(bench, t.start);
    const { model } = applyActions(start, solutions[id]);
    const g = gradeTask(t, model, solutions[id].length);
    expect(g.failures, `${id} reference failed: ${g.failures.join('; ')}`).toEqual([]);
    expect(g.pass).toBe(true);
  });
});

describe('verifier', () => {
  it('flags non-divisible attention heads as a blocker', () => {
    const m = buildFixture(bench, 'broken-attention');
    const blockers = findBlockers(m);
    expect(blockers.join(' ')).toMatch(/not divisible/);
    expect(scoreModel(m).score).toBeLessThan(40);
  });

  it('treats a disconnected graph as not reaching output', () => {
    const m = { name: 'x', components: [
      { id: 'i', type: 'input', name: 'input', params: {}, inputs: [], outputs: [] },
      { id: 'o', type: 'output', name: 'output', params: {}, inputs: [], outputs: [] },
    ], connections: [] };
    expect(inputReachesOutput(m)).toBe(false);
  });

  it('scores a clean deep CNN well above a broken graph', () => {
    const clean = buildFixture(bench, 'tiny-cnn');
    const broken = buildFixture(bench, 'broken-attention');
    expect(scoreModel(clean).score).toBeGreaterThan(scoreModel(broken).score);
  });
});

describe('apply + grade round trip', () => {
  it('fixing the broken attention head clears the blocker and passes', () => {
    const start = buildFixture(bench, 'broken-attention');
    // 100 -> 98 is divisible by 7? no. Use numHeads 5 (100 % 5 == 0).
    const { model } = applyActions(start, [
      { type: 'update_params', name: 'attn', params: { numHeads: 5 } },
    ]);
    const g = gradeTask(task('fix-broken-attention'), model, 1);
    expect(g.blockers.length).toBe(0);
    expect(g.pass).toBe(true);
  });

  it('grades a from-scratch CNN design against the cnn-cifar task', () => {
    const start = buildFixture(bench, 'empty-image');
    const { model } = applyActions(start, [
      { type: 'add_component', componentType: 'conv2d', name: 'conv1', afterName: 'input', params: { inChannels: 3, outChannels: 32, kernelSize: 3 } },
      { type: 'add_component', componentType: 'relu', name: 'act1', afterName: 'conv1' },
      { type: 'add_component', componentType: 'conv2d', name: 'conv2', afterName: 'act1', params: { inChannels: 32, outChannels: 64, kernelSize: 3 } },
      { type: 'add_component', componentType: 'relu', name: 'act2', afterName: 'conv2' },
      { type: 'add_component', componentType: 'linear', name: 'head', afterName: 'act2', params: { inFeatures: 64, outFeatures: 10 } },
    ]);
    const g = gradeTask(task('cnn-cifar'), model, 5);
    expect(g.blockers.length).toBe(0);
    expect(g.failures).toEqual([]);
    expect(g.pass).toBe(true);
  });

  it('non-structural actions are skipped, not applied', () => {
    const start = buildFixture(bench, 'empty-image');
    const r = applyActions(start, [{ type: 'auto_layout' }, { type: 'open_library' }]);
    expect(r.applied).toBe(0);
    expect(r.skipped).toEqual(['auto_layout', 'open_library']);
  });
});

// The lower bound of the verifier: degenerate strategies MUST fail. Reference
// solutions pin the upper bound (a clean solve passes); this suite pins the
// floor, so a rubric change that lets a lazy policy through fails CI instead
// of the benchmark silently becoming a rubber stamp. See ANTI_GAMING.md for
// the full strategy-by-strategy write-up.
describe('degenerate solutions must fail (anti-gaming floor)', () => {
  it('the empty plan fails a design-from-spec task', () => {
    const start = buildFixture(bench, task('cnn-cifar').start);
    const g = gradeTask(task('cnn-cifar'), start, 0);
    expect(g.pass).toBe(false);
    expect(g.failures.join(' ')).toMatch(/components|missing layer type/);
  });

  it('the empty plan fails a repair task (the start graph is broken)', () => {
    const start = buildFixture(bench, task('fix-broken-attention').start);
    const g = gradeTask(task('fix-broken-attention'), start, 0);
    expect(g.pass).toBe(false);
    expect(g.failures.join(' ')).toMatch(/blocker/);
  });

  it('a disconnected showpiece (required types present, nothing wired to output) fails reachability', () => {
    const start = buildFixture(bench, 'empty-image');
    // Add the required conv/linear as a chain but never reconnect to output.
    const { model } = applyActions(start, [
      { type: 'delete_connection', fromName: 'input', toName: 'output' },
      { type: 'add_component', componentType: 'conv2d', name: 'conv1', afterName: 'input', params: { inChannels: 3, outChannels: 32, kernelSize: 3 } },
      { type: 'add_component', componentType: 'conv2d', name: 'conv2', afterName: 'conv1', params: { inChannels: 32, outChannels: 64, kernelSize: 3 } },
      { type: 'add_component', componentType: 'relu', name: 'act1', afterName: 'conv2' },
      { type: 'add_component', componentType: 'linear', name: 'head', afterName: 'act1', params: { inFeatures: 64, outFeatures: 10 } },
      { type: 'delete_connection', fromName: 'head', toName: 'output' },
    ]);
    const g = gradeTask(task('cnn-cifar'), model, 6);
    if (g.pass) {
      // If the applier auto-rewired output, the case is moot — assert the guard
      // itself instead so the test can't silently pass for the wrong reason.
      expect(inputReachesOutput(model)).toBe(true);
    } else {
      expect(g.failures.join(' ')).toMatch(/reach output|components/);
    }
  });

  it('blowing past the param budget fails even with a perfect structure', () => {
    const start = buildFixture(bench, task('mlp-tabular').start);
    const { model } = applyActions(start, [
      { type: 'add_component', componentType: 'linear', name: 'fc1', afterName: 'input', params: { inFeatures: 32, outFeatures: 4096 } },
      { type: 'add_component', componentType: 'relu', name: 'act1', afterName: 'fc1' },
      { type: 'add_component', componentType: 'linear', name: 'fc2', afterName: 'act1', params: { inFeatures: 4096, outFeatures: 4096 } },
      { type: 'add_component', componentType: 'relu', name: 'act2', afterName: 'fc2' },
      { type: 'add_component', componentType: 'linear', name: 'head', afterName: 'act2', params: { inFeatures: 4096, outFeatures: 2 } },
    ]);
    const g = gradeTask(task('mlp-tabular'), model, 5);
    expect(g.pass).toBe(false);
    expect(g.failures.join(' ')).toMatch(/> budget/);
  });

  it('wholesale rebuild via replace_model fails a repair task regardless of graph quality', () => {
    const t = task('fix-broken-attention');
    const start = buildFixture(bench, t.start);
    const { model } = applyActions(start, [
      { type: 'update_params', name: 'attn', params: { numHeads: 5 } },
    ]);
    // The graph itself is now clean — but the action log says replace_model.
    const g = gradeTask(t, model, 1, ['replace_model']);
    if (t.constraints.forbidActionTypes?.includes('replace_model')) {
      expect(g.pass).toBe(false);
      expect(g.failures.join(' ')).toMatch(/forbidden action/);
    } else {
      // Curated repair task doesn't forbid it (generated ones do) — then the
      // action cap must be the binding guard instead.
      expect(typeof t.constraints.maxActions).toBe('number');
    }
  });
});

// Algorithm 1 line 8 (shapeMismatch) and the orphan check. Both classes are
// forward-pass failures in PyTorch that the earlier rubric scored as clean;
// these tests are pinned so the checks cannot be silently weakened again.
describe('width propagation and orphan detection', () => {
  const chain = (nodes: any[]) => {
    const components = nodes.map((n, i) => ({
      id: `n${i}`, type: n.t, name: n.name, params: n.params ?? {}, inputs: [], outputs: [],
    }));
    const connections = components.slice(0, -1).map((c, i) => ({ id: `c${i}`, from: c.id, to: components[i + 1].id }));
    for (const cn of connections) {
      components.find(c => c.id === cn.from)!.outputs.push(cn.to);
      components.find(c => c.id === cn.to)!.inputs.push(cn.from);
    }
    return { name: 'probe', components, connections };
  };

  it('flags a mid-chain linear whose inFeatures disagree with the upstream width', () => {
    const m = chain([
      { t: 'input', name: 'input', params: { shape: [1, 128] } },
      { t: 'embedding', name: 'embed', params: { numEmbeddings: 1000, embeddingDim: 256 } },
      { t: 'linear', name: 'fc1', params: { inFeatures: 512, outFeatures: 64 } },
      { t: 'output', name: 'output' },
    ]);
    expect(findBlockers(m).join(' ')).toMatch(/fc1: linear declares input width 512 but upstream emits 256/);
  });

  it('accepts the same chain once the widths agree', () => {
    const m = chain([
      { t: 'input', name: 'input', params: { shape: [1, 128] } },
      { t: 'embedding', name: 'embed', params: { numEmbeddings: 1000, embeddingDim: 256 } },
      { t: 'linear', name: 'fc1', params: { inFeatures: 256, outFeatures: 64 } },
      { t: 'output', name: 'output' },
    ]);
    expect(findBlockers(m)).toEqual([]);
  });

  it('propagates through normalization and activation without losing the width', () => {
    const m = chain([
      { t: 'input', name: 'input', params: { shape: [1, 128] } },
      { t: 'embedding', name: 'embed', params: { numEmbeddings: 1000, embeddingDim: 256 } },
      { t: 'layerNorm', name: 'ln' },
      { t: 'relu', name: 'act' },
      { t: 'linear', name: 'fc1', params: { inFeatures: 128, outFeatures: 64 } },
      { t: 'output', name: 'output' },
    ]);
    expect(findBlockers(m).join(' ')).toMatch(/fc1: .* upstream emits 256/);
  });

  it('flags attention fed a width other than its embedDim', () => {
    const m = chain([
      { t: 'input', name: 'input', params: { shape: [1, 128] } },
      { t: 'embedding', name: 'embed', params: { numEmbeddings: 1000, embeddingDim: 256 } },
      { t: 'multiHeadAttention', name: 'attn', params: { embedDim: 512, numHeads: 8 } },
      { t: 'output', name: 'output' },
    ]);
    expect(findBlockers(m).join(' ')).toMatch(/attn: multiHeadAttention declares input width 512/);
  });

  it('stays silent when the upstream width is not statically known', () => {
    // conv2d straight off a raw input tensor: the rubric cannot tell channels
    // from spatial dims, so it must not guess. No false positive.
    const m = chain([
      { t: 'input', name: 'input', params: { shape: [3, 32, 32] } },
      { t: 'conv2d', name: 'c1', params: { inChannels: 3, outChannels: 32, kernelSize: 3 } },
      { t: 'output', name: 'output' },
    ]);
    expect(findBlockers(m)).toEqual([]);
  });

  it('flags an orphaned middle layer even while another path still reaches the output', () => {
    // Two towers; the edge into the user tower's activation is severed. Some
    // input still reaches some output, so reachability alone cannot see this.
    const components = [
      { id: 'i', type: 'input', name: 'input', params: { shape: [1, 64] }, inputs: [], outputs: [] },
      { id: 'u', type: 'linear', name: 'user_fc', params: { inFeatures: 64, outFeatures: 32 }, inputs: [], outputs: [] },
      { id: 'ua', type: 'relu', name: 'user_act', params: {}, inputs: [], outputs: [] },
      { id: 'v', type: 'linear', name: 'item_fc', params: { inFeatures: 64, outFeatures: 32 }, inputs: [], outputs: [] },
      { id: 'o', type: 'output', name: 'output', params: {}, inputs: [], outputs: [] },
    ];
    const connections = [
      { id: 'c0', from: 'i', to: 'u' },
      { id: 'c1', from: 'i', to: 'v' },
      { id: 'c2', from: 'v', to: 'o' },
      // c3 (u -> ua) deliberately absent: user_act is orphaned.
    ];
    for (const cn of connections) {
      components.find(c => c.id === cn.from)!.outputs.push(cn.to);
      components.find(c => c.id === cn.to)!.inputs.push(cn.from);
    }
    const m = { name: 'two-tower-severed', components, connections };
    expect(inputReachesOutput(m)).toBe(true);
    expect(findBlockers(m).join(' ')).toMatch(/user_act: no incoming connection \(orphaned\)/);
  });
});
