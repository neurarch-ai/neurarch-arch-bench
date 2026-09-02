import { describe, it, expect } from 'vitest';
import { readFileSync, writeFileSync, unlinkSync } from 'node:fs';
import { SYSTEM_PROMPT, loadPolicy } from './providers.mjs';

/**
 * The policy seam that makes program.md's overnight loop possible: one file
 * the agent may edit, loaded in place of the built-in prompt.
 */
describe('loadPolicy', () => {
  it('returns the built-in prompt when no policy file is configured', () => {
    expect(loadPolicy(undefined)).toBe(SYSTEM_PROMPT);
  });

  /** A search must start from the published baseline, or its first "win" is
   *  just the gap between two different starting prompts. */
  it('ships policy.md byte-identical to the built-in prompt', () => {
    expect(loadPolicy('policy.md')).toBe(SYSTEM_PROMPT);
  });

  it('strips HTML comments so notes to the human never reach the model', () => {
    const f = 'policy.test.tmp.md';
    writeFileSync(f, '<!-- do not send this -->\nDesign a network.\n');
    try {
      expect(loadPolicy(f)).toBe('Design a network.');
    } finally { unlinkSync(f); }
  });

  it('refuses a policy file that is nothing but comments', () => {
    const f = 'policy.empty.tmp.md';
    writeFileSync(f, '<!-- only a note -->\n');
    try {
      expect(() => loadPolicy(f)).toThrow(/no prompt text/);
    } finally { unlinkSync(f); }
  });

  /** program.md tells the agent that policy.md is the ONLY editable file, so
   *  the commands it prints must be the commands that exist. */
  it('program.md points at the flag the loader actually reads', () => {
    const program = readFileSync('program.md', 'utf8');
    expect(program).toContain('NEURARCH_POLICY_FILE=policy.md');
    expect(program).toContain('--generate=40');
  });
});
