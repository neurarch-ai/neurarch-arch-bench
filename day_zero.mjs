/**
 * day_zero — the full evaluation battery as one command, for model-launch day.
 *
 * When a new model ships (a Grok release, a new Claude, an open-weights drop),
 * run this once and every measurement this project reports exists as a fresh,
 * artifact-backed set within a couple of hours:
 *
 *   1. leaderboard, curated split (12 production tasks)
 *   2. leaderboard, generated split (120 tasks, seed 7)
 *   3. verifier-in-the-loop amplification (120 tasks, 3 turns)
 *   4. verifier-as-tool, standard + frontier tiers (30 tasks each)
 *   5. reward-model audit, blatant + near-miss tiers (60 examples each)
 *
 * Everything lands in day0-<label>/ with one JSON/txt artifact per step, ready
 * for the thread template in the repo (see blog draft) and for VERIFICATION.md.
 *
 * Usage:
 *   node day_zero.mjs --dry-run                     # keyless: self-check every harness
 *   XAI_API_KEY=... node day_zero.mjs --provider=grok --label=grok-5 [--delay=500]
 *   XAI_API_KEY=... XAI_MODEL=grok-5 node day_zero.mjs --provider=grok --label=grok-5
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const m = a.match(/^--([^=]+)(?:=(.*))?$/); return m ? [m[1], m[2] ?? 'true'] : [a, 'true'];
}));

function run(title, cmd, cmdArgs, env = {}) {
  console.log(`\n=== ${title} ===`);
  const r = spawnSync(cmd, cmdArgs, { stdio: 'inherit', env: { ...process.env, ...env } });
  if (r.status !== 0) console.error(`[day-zero] step "${title}" exited ${r.status} — continuing`);
  return r.status === 0;
}

if (args['dry-run']) {
  console.log('day-zero dry run: keyless self-checks for every harness in the battery');
  let ok = true;
  ok = run('leaderboard self-test (reference oracle)', 'node', ['leaderboard.mjs', '--providers=reference']) && ok;
  ok = run('calibrate self-test (reference)', 'node', ['calibrate.mjs', '--policy=reference']) && ok;
  ok = run('calibrate self-test (noop)', 'node', ['calibrate.mjs', '--policy=noop']) && ok;
  ok = run('amplify self-check', 'node', ['amplify.mjs', '--self-check']) && ok;
  ok = run('tool_use self-check', 'node', ['tool_use.mjs', '--self-check']) && ok;
  ok = run('reward_anchor keyless set', 'node', ['reward_anchor.mjs', '--count=5']) && ok;
  console.log(ok ? '\nDry run OK: the battery is ready to fire.' : '\nDRY RUN FOUND FAILURES — fix before launch day.');
  process.exit(ok ? 0 : 1);
}

const provider = args.provider ?? 'grok';
const label = args.label ?? provider;
const delay = args.delay ?? '500';
const dir = `day0-${label}`;
fs.mkdirSync(dir, { recursive: true });
console.log(`day-zero battery: provider=${provider}, label=${label} -> ${dir}/`);

run('1/6 leaderboard, curated', 'node', ['leaderboard.mjs', `--providers=${provider}`],
  { LEADERBOARD_OUT: `${dir}/lb-curated.json` });
run('2/6 leaderboard, generated (120, seed 7)', 'node', ['leaderboard.mjs', `--providers=${provider}`, '--generate=120', '--seed=7'],
  { LEADERBOARD_OUT: `${dir}/lb-generated.json` });
run('3/6 amplification (120, seed 7, 3 turns)', 'node', ['amplify.mjs', `--providers=${provider}`, '--generate=120', '--seed=7', '--turns=3'],
  { AMPLIFY_OUT: `${dir}/amplify.json` });
run('4/6 verifier-as-tool, standard tier', 'node', ['tool_use.mjs', `--provider=${provider}`, '--generate=30', '--seed=7', `--delay=${delay}`],
  { TOOLUSE_OUT: `${dir}/tooluse-standard.json` });
run('5/6 verifier-as-tool, frontier tier', 'node', ['tool_use.mjs', `--provider=${provider}`, '--generate=30', '--seed=7', '--tier=frontier', `--delay=${delay}`],
  { TOOLUSE_OUT: `${dir}/tooluse-frontier.json` });
run('6/6 reward audit, blatant + near-miss', 'sh', ['-c',
  `node reward_anchor.mjs --provider=${provider} --delay=${delay} | tee ${dir}/reward-blatant.txt; ` +
  `node reward_anchor.mjs --provider=${provider} --near-miss --delay=${delay} | tee ${dir}/reward-nearmiss.txt`]);

console.log(`\nBattery complete. Artifacts in ${dir}/:`);
for (const f of fs.readdirSync(dir)) console.log(`  ${dir}/${f}`);
console.log('Fill the thread template from these, add the rows to VERIFICATION.md, publish.');
