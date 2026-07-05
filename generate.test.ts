import { describe, it, expect } from 'vitest';
// @ts-expect-error plain ESM module without types
import { applyActions, gradeTask, categorizeFailure } from './bench.mjs';
// @ts-expect-error plain ESM module without types
import { generateCases } from './generate.mjs';

describe('procedural task generation (oss)', () => {
  it('is deterministic for a given seed', () => {
    expect(JSON.stringify(generateCases(16, 42))).toBe(JSON.stringify(generateCases(16, 42)));
  });

  it('different seeds produce different splits', () => {
    expect(JSON.stringify(generateCases(16, 1))).not.toBe(JSON.stringify(generateCases(16, 2)));
  });

  it('cycles through all ten task families', () => {
    const ids = generateCases(20, 7).map((c: any) => c.task.id);
    for (const prefix of ['gen-mlp-', 'gen-ae-', 'gen-cnn-', 'gen-txf-', 'gen-gqa-', 'gen-fix-', 'gen-trim-', 'gen-norm-', 'gen-tower-', 'gen-grow-']) {
      expect(ids.some((id: string) => id.startsWith(prefix)), `no ${prefix} task generated`).toBe(true);
    }
  });

  // No unsatisfiable tasks: every generated reference passes its own task.
  it('every generated reference solution passes its task (500 cases)', () => {
    const cases = generateCases(500, 123);
    expect(new Set(cases.map((c: any) => c.task.id)).size).toBe(500);
    for (const { task, start, reference } of cases) {
      const applied = applyActions(start, reference);
      expect(applied.errors, `${task.id} apply errors: ${applied.errors.join('; ')}`).toEqual([]);
      const result = gradeTask(task, applied.model, reference.length, reference.map((a: any) => a.type));
      expect(result.pass, `${task.id} failed: ${result.failures.join('; ')}`).toBe(true);
    }
  });

  // No vacuous tasks: every edit-in-place start graph fails its task untouched.
  it('edit-in-place start graphs fail their own task before the fix', () => {
    const cases = generateCases(64, 9).filter((c: any) =>
      ['gen-fix-', 'gen-trim-', 'gen-norm-', 'gen-grow-'].some(p => c.task.id.startsWith(p)),
    );
    expect(cases.length).toBeGreaterThan(0);
    for (const { task, start } of cases) {
      expect(gradeTask(task, start, 0, []).pass, `${task.id} passed with no edit at all`).toBe(false);
    }
  });

  // Anti-gaming: replace_model on a repair task fails even if the graph is valid.
  it('repair tasks reject replace_model solutions', () => {
    const fix = generateCases(16, 11).find((c: any) => c.task.id.startsWith('gen-fix-'))!;
    const rebuilt = applyActions(fix.start, [{
      type: 'replace_model',
      components: [
        { componentType: 'input', name: 'input', params: { shape: [1, 64] } },
        { componentType: 'embedding', name: 'embed', params: { numEmbeddings: 20000, embeddingDim: 128 } },
        { componentType: 'multiHeadAttention', name: 'attn', params: { embedDim: 128, numHeads: 8 } },
        { componentType: 'linear', name: 'head', params: { inFeatures: 128, outFeatures: 2 } },
        { componentType: 'output', name: 'output', params: {} },
      ],
      connections: [
        { from: 'input', to: 'embed' },
        { from: 'embed', to: 'attn' },
        { from: 'attn', to: 'head' },
        { from: 'head', to: 'output' },
      ],
    }]);
    const result = gradeTask(fix.task, rebuilt.model, 1, ['replace_model']);
    expect(result.pass).toBe(false);
    expect(result.failures.some((f: string) => f.includes('forbidden action'))).toBe(true);
  });

  it('failure taxonomy maps grader messages to stable categories', () => {
    expect(categorizeFailure('structural blocker: attn: embedDim 100 not divisible by numHeads 7')).toBe('attention-divisibility');
    expect(categorizeFailure('input does not reach output')).toBe('connectivity');
    expect(categorizeFailure('params 51000000 > budget 50000000')).toBe('over-budget');
    expect(categorizeFailure('params 3000 < required minimum 400000')).toBe('under-band');
    expect(categorizeFailure('missing layer type "batchNorm1d"')).toBe('missing-layer-type');
    expect(categorizeFailure('used forbidden action "replace_model"')).toBe('forbidden-action');
    expect(categorizeFailure('5 actions > max 2')).toBe('action-limit');
    expect(categorizeFailure('Error: no JSON object in reply')).toBe('parse-error');
    expect(categorizeFailure('score 42 < min 50')).toBe('low-score');
  });
});

describe('kvBytesPerToken (oss canonical formula)', () => {
  it('MHA caches 2 x dim, GQA caches kv-heads share, MLA caches the latent', async () => {
    const { kvBytesPerToken } = await import('./bench.mjs') as any;
    const g = (type: string, params: Record<string, unknown>) => ({
      components: [{ id: 'a', type, name: 'a', params, inputs: [], outputs: [] }],
      connections: [],
    });
    expect(kvBytesPerToken(g('multiHeadAttention', { embedDim: 4096, numHeads: 32 }))).toBe(2 * 4096 * 2);
    expect(kvBytesPerToken(g('groupedQueryAttention', { embedDim: 4096, numHeads: 32, numKVHeads: 8 }))).toBe(2 * 8 * 128 * 2);
    expect(kvBytesPerToken(g('mla', { embedDim: 4096, numHeads: 32, kvLatentDim: 512, ropeHeadDim: 64 }))).toBe((512 + 64) * 2);
    expect(kvBytesPerToken(g('linear', { inFeatures: 8, outFeatures: 8 }))).toBe(0);
  });
});
