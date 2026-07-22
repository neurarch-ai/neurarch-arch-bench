import { describe, it, expect } from 'vitest';
// @ts-expect-error plain ESM module without types
import { applyActions, gradeTask, categorizeFailure, kvBytesPerToken } from './bench.mjs';
// @ts-expect-error plain ESM module without types
import { generateEdgeCases } from './generate-edge.mjs';

describe('edge tier (joint param + KV budgets)', () => {
  it('is deterministic for a given seed', () => {
    expect(JSON.stringify(generateEdgeCases(8, 42))).toBe(JSON.stringify(generateEdgeCases(8, 42)));
  });

  it('cycles through both edge families', () => {
    const ids = generateEdgeCases(4, 7).map((c: any) => c.task.id);
    for (const prefix of ['gen-eedge-', 'gen-eshrink-']) {
      expect(ids.some((id: string) => id.startsWith(prefix)), `no ${prefix} task generated`).toBe(true);
    }
  });

  // No unsatisfiable tasks: every reference passes its own task.
  it('every edge reference solution passes its task (200 cases)', () => {
    for (const { task, start, reference } of generateEdgeCases(200, 123)) {
      const applied = applyActions(start, reference);
      expect(applied.errors, `${task.id} apply errors: ${applied.errors.join('; ')}`).toEqual([]);
      const result = gradeTask(task, applied.model, reference.length, reference.map((a: any) => a.type));
      expect(result.pass, `${task.id} failed: ${result.failures.join('; ')}`).toBe(true);
    }
  });

  // No vacuous tasks: every start graph fails untouched.
  it('every edge start graph fails its own task before any edit', () => {
    for (const { task, start } of generateEdgeCases(40, 9)) {
      expect(gradeTask(task, start, 0, []).pass, `${task.id} passed with no edit at all`).toBe(false);
    }
  });

  // The shrink start fails BOTH budgets, and the reference lands under both.
  it('joint budgets both bite on the shrink family', () => {
    const sk = generateEdgeCases(4, 13).find((c: any) => c.task.id.startsWith('gen-eshrink-'))!;
    const startResult = gradeTask(sk.task, sk.start, 0, []);
    const cats = startResult.failures.map((f: string) => categorizeFailure(f));
    expect(cats).toContain('over-budget');
    expect(cats).toContain('kv-over-budget');
    const applied = applyActions(sk.start, sk.reference);
    expect(kvBytesPerToken(applied.model)).toBeLessThanOrEqual(sk.task.constraints.maxKvBytesPerToken);
  });

  // Anti-gaming: replace_model is forbidden on the shrink family.
  it('shrink tasks reject replace_model solutions', () => {
    const sk = generateEdgeCases(4, 17).find((c: any) => c.task.id.startsWith('gen-eshrink-'))!;
    const result = gradeTask(sk.task, sk.start, 1, ['replace_model']);
    expect(result.pass).toBe(false);
    expect(result.failures.join('; ')).toContain('forbidden action');
  });

  it('spec distinctness holds at scale', () => {
    const specs = generateEdgeCases(2000, 1).map((c: any) => c.task.spec);
    expect(new Set(specs).size).toBeGreaterThan(1800);
  });
});
