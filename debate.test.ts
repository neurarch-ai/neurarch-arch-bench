import { describe, expect, it } from 'vitest';
import { arbitrate } from './debate.mjs';

const p = (pass: boolean, score: number, tokens: number) => ({ pass, score, tokens });

describe('debate arbitration', () => {
  it('pass beats fail regardless of score', () => {
    expect(arbitrate([p(false, 90, 10), p(true, 40, 999)])).toBe(1);
    expect(arbitrate([p(true, 40, 999), p(false, 90, 10)])).toBe(0);
  });

  it('among passing proposals, higher health score wins', () => {
    expect(arbitrate([p(true, 60, 10), p(true, 75, 10), p(true, 70, 10)])).toBe(1);
  });

  it('score tie goes to fewer tokens', () => {
    expect(arbitrate([p(true, 70, 500), p(true, 70, 200)])).toBe(1);
  });

  it('full tie keeps the lower seat index (stable)', () => {
    expect(arbitrate([p(true, 70, 200), p(true, 70, 200), p(true, 70, 200)])).toBe(0);
    expect(arbitrate([p(false, 0, 0), p(false, 0, 0)])).toBe(0);
  });

  it('is deterministic: same proposals, same winner', () => {
    const panel = [p(false, 20, 50), p(true, 55, 300), p(true, 55, 100), p(false, 80, 10)];
    const first = arbitrate(panel);
    for (let i = 0; i < 5; i++) expect(arbitrate(panel)).toBe(first);
    expect(first).toBe(2);
  });
});
