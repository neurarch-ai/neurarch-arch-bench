import { describe, it, expect } from 'vitest';
// @ts-expect-error plain ESM module without types
import { applyActions, gradeTask, categorizeFailure, kvBytesPerToken } from './bench.mjs';
// @ts-expect-error plain ESM module without types
import { generateFrontierCases } from './generate-frontier.mjs';

describe('frontier tier (MoE / MLA-KV / GQA retrofit)', () => {
  it('is deterministic for a given seed', () => {
    expect(JSON.stringify(generateFrontierCases(12, 42))).toBe(JSON.stringify(generateFrontierCases(12, 42)));
  });

  it('cycles through all three frontier families', () => {
    const ids = generateFrontierCases(6, 7).map((c: any) => c.task.id);
    for (const prefix of ['gen-fmoe-', 'gen-fmla-', 'gen-fkv-']) {
      expect(ids.some((id: string) => id.startsWith(prefix)), `no ${prefix} task generated`).toBe(true);
    }
  });

  // No unsatisfiable tasks: every reference passes its own task.
  it('every frontier reference solution passes its task (201 cases)', () => {
    const cases = generateFrontierCases(201, 123);
    for (const { task, start, reference } of cases) {
      const applied = applyActions(start, reference);
      expect(applied.errors, `${task.id} apply errors: ${applied.errors.join('; ')}`).toEqual([]);
      const result = gradeTask(task, applied.model, reference.length, reference.map((a: any) => a.type));
      expect(result.pass, `${task.id} failed: ${result.failures.join('; ')}`).toBe(true);
    }
  });

  // No vacuous tasks: every start graph fails its task untouched.
  it('every frontier start graph fails its own task before any edit', () => {
    for (const { task, start } of generateFrontierCases(60, 9)) {
      expect(gradeTask(task, start, 0, []).pass, `${task.id} passed with no edit at all`).toBe(false);
    }
  });

  it('a noop submission fails every frontier task', () => {
    for (const { task, start } of generateFrontierCases(30, 5)) {
      const applied = applyActions(start, []);
      expect(gradeTask(task, applied.model, 0, []).pass).toBe(false);
    }
  });

  // MoE routing validity: topK > numExperts is a hard blocker.
  it('flags topK > numExperts as a blocker on the MoE family', () => {
    const moe = generateFrontierCases(3, 11).find((c: any) => c.task.id.startsWith('gen-fmoe-'))!;
    const bad = JSON.parse(JSON.stringify(moe.reference));
    const moeComp = bad[0].components.find((c: any) => c.componentType === 'mixtureOfExperts');
    moeComp.params.topK = moeComp.params.numExperts + 3;
    const applied = applyActions(moe.start, bad);
    const result = gradeTask(moe.task, applied.model, bad.length, bad.map((a: any) => a.type));
    expect(result.pass).toBe(false);
    expect(result.failures.join('; ')).toContain('topK');
    expect(result.failures.some((f: string) => categorizeFailure(f) === 'moe-routing')).toBe(true);
  });

  // KV budget really bites: the retrofit start is over budget, the reference
  // lands under it, and the failure categorizes as kv-over-budget.
  it('grades the KV budget from the canonical kvBytesPerToken formula', () => {
    const kv = generateFrontierCases(6, 13).find((c: any) => c.task.id.startsWith('gen-fkv-'))!;
    const startKv = kvBytesPerToken(kv.start);
    expect(startKv).toBeGreaterThan(kv.task.constraints.maxKvBytesPerToken);
    const startResult = gradeTask(kv.task, kv.start, 0, []);
    expect(startResult.failures.some((f: string) => categorizeFailure(f) === 'kv-over-budget')).toBe(true);
    const applied = applyActions(kv.start, kv.reference);
    expect(kvBytesPerToken(applied.model)).toBeLessThanOrEqual(kv.task.constraints.maxKvBytesPerToken);
  });

  // Anti-gaming: replace_model on the retrofit family fails even if valid.
  it('retrofit tasks reject replace_model solutions', () => {
    const kv = generateFrontierCases(6, 17).find((c: any) => c.task.id.startsWith('gen-fkv-'))!;
    const D = kv.start.components.find((c: any) => c.name === 'embed').params.embeddingDim;
    const rebuilt = applyActions(kv.start, [{
      type: 'replace_model',
      components: [
        { componentType: 'input', name: 'input', params: { shape: [1, 128] } },
        { componentType: 'embedding', name: 'embed', params: { numEmbeddings: 8000, embeddingDim: D } },
        { componentType: 'groupedQueryAttention', name: 'gqa', params: { embedDim: D, numHeads: 8, numKVHeads: 2 } },
        { componentType: 'layerNorm', name: 'ln', params: { normalizedShape: D } },
        { componentType: 'linear', name: 'head', params: { inFeatures: D, outFeatures: 10 } },
        { componentType: 'output', name: 'output', params: {} },
      ],
      connections: [
        { from: 'input', to: 'embed' }, { from: 'embed', to: 'gqa' },
        { from: 'gqa', to: 'ln' }, { from: 'ln', to: 'head' }, { from: 'head', to: 'output' },
      ],
    }]);
    const result = gradeTask(kv.task, rebuilt.model, 1, ['replace_model']);
    expect(result.pass).toBe(false);
    expect(result.failures.join('; ')).toContain('forbidden action');
  });

  // Deleting the attention outright cannot pass (mustContainTypesAny + depth).
  it('amputating attention fails the retrofit task', () => {
    const kv = generateFrontierCases(6, 19).find((c: any) => c.task.id.startsWith('gen-fkv-'))!;
    const applied = applyActions(kv.start, [{ type: 'delete_component', name: 'attn' }]);
    const result = gradeTask(kv.task, applied.model, 1, ['delete_component']);
    expect(result.pass).toBe(false);
  });

  it('spec distinctness holds at scale', () => {
    const specs = generateFrontierCases(3000, 1).map((c: any) => c.task.spec);
    expect(new Set(specs).size).toBeGreaterThan(2600);
  });
});
