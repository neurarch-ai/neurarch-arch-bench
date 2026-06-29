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
