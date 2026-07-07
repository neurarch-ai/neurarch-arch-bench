#!/usr/bin/env node
/**
 * card — turn a run's JSON output into a shareable, self-contained SVG card.
 *
 * The distribution-friction remover: every leaderboard / amplify / arena /
 * calibrate run already writes JSON; this renders it into one branded graphic
 * you can embed in the README (GitHub renders SVG natively), drop in the
 * paper, or screenshot for X. Zero dependencies, pure string SVG.
 *
 *   LEADERBOARD_OUT=b.json node leaderboard.mjs --providers=grok ... && node card.mjs b.json
 *   AMPLIFY_OUT=a.json     node amplify.mjs ...                       && node card.mjs a.json --out=amp.svg
 *   CALIBRATE_OUT=c.json   node calibrate.mjs --provider=grok ...     && node card.mjs c.json
 *   ARENA_OUT=d.json       node arena.mjs --a=grok --b=claude ...     && node card.mjs d.json
 *
 * Format is auto-detected from the JSON shape. Output is 1200x630 (OG size).
 * To post on X, open the SVG in a browser and screenshot, or convert with any
 * svg->png tool; GitHub READMEs render the SVG directly.
 */
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const inPath = args.find(a => !a.startsWith('-'));
const outArg = (args.find(a => a.startsWith('--out=')) || '').slice('--out='.length);
if (!inPath) { console.error('usage: node card.mjs <run.json> [--out=card.svg]'); process.exit(2); }

const data = JSON.parse(fs.readFileSync(path.resolve(inPath), 'utf8'));

// ── SVG helpers ──────────────────────────────────────────────────────────────
const W = 1200, H = 630;
const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const BG = '#07091a', CARD = '#0d0f1e', BORDER = '#1e2440', TEXT = '#f1f5f9', MUTED = '#94a3b8', DIM = '#64748b';
const G1 = '#818cf8', G2 = '#c084fc', GREEN = '#6ee7b7', AMBER = '#f59e0b';
const FONT = "-apple-system, 'Segoe UI', Roboto, Inter, sans-serif";
const MONO = "'SFMono-Regular', Consolas, 'JetBrains Mono', monospace";

const frame = (title, subtitle, body) => `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="${FONT}">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="${G1}"/><stop offset="1" stop-color="${G2}"/></linearGradient></defs>
  <rect width="${W}" height="${H}" fill="${BG}"/>
  <rect x="1" y="1" width="${W - 2}" height="${H - 2}" rx="18" fill="${CARD}" stroke="${BORDER}"/>
  <text x="64" y="88" font-size="20" font-weight="700" letter-spacing="3" fill="${DIM}">NEURARCH-ARCH-BENCH</text>
  <text x="64" y="140" font-size="40" font-weight="800" fill="${TEXT}">${esc(title)}</text>
  ${subtitle ? `<text x="64" y="176" font-size="20" fill="${MUTED}">${esc(subtitle)}</text>` : ''}
  ${body}
  <text x="64" y="${H - 40}" font-size="18" fill="${DIM}" font-family="${MONO}">github.com/neurarch-ai/neurarch-arch-bench</text>
  <text x="${W - 64}" y="${H - 40}" font-size="17" fill="${DIM}" text-anchor="end">deterministic verifier &#183; no LLM judge &#183; reproducible</text>
</svg>`;

// ── Renderers ────────────────────────────────────────────────────────────────
function amplifyCard(d) {
  const rows = (d.results || []).slice(0, 6);
  const y0 = 240, rh = 52;
  const body = `
  <text x="64" y="${y0 - 20}" font-size="17" fill="${DIM}" letter-spacing="1">MODEL</text>
  <text x="640" y="${y0 - 20}" font-size="17" fill="${DIM}" letter-spacing="1">SINGLE-SHOT</text>
  <text x="860" y="${y0 - 20}" font-size="17" fill="${DIM}" letter-spacing="1">+ VERIFIER</text>
  <text x="1090" y="${y0 - 20}" font-size="17" fill="${DIM}" letter-spacing="1" text-anchor="end">LIFT</text>
  ${rows.map((r, i) => {
    const y = y0 + i * rh;
    const ss = Math.round((r.singleShot ?? 0) * 100), wv = Math.round((r.withVerifier ?? 0) * 100);
    const lift = Math.round(r.liftPoints ?? (wv - ss));
    return `<text x="64" y="${y}" font-size="24" font-weight="600" fill="${TEXT}" font-family="${MONO}">${esc(r.model || r.provider)}</text>
      <text x="640" y="${y}" font-size="24" fill="${MUTED}" font-family="${MONO}">${ss}%</text>
      <text x="860" y="${y}" font-size="24" font-weight="700" fill="url(#g)" font-family="${MONO}">${wv}%</text>
      <text x="1090" y="${y}" font-size="24" font-weight="700" fill="${lift >= 0 ? GREEN : AMBER}" font-family="${MONO}" text-anchor="end">+${lift} pts</text>`;
  }).join('')}`;
  return frame('Verifier feedback makes every model better', 'Same tasks, same model, same prompt; the only variable is the verifier loop', body);
}

function calibrateCard(d) {
  // Accept both the oss schema ({rows:[{family,rate,band}]}) and the private
  // rl:calibrate schema ({families:[{family,passRate,band}]}).
  const src = d.rows || (d.families || []).map(f => ({ family: f.family, rate: f.passRate, band: f.band }));
  const rows = src.slice(0, 12);
  const bandColor = b => {
    const u = (b || '').toUpperCase();
    return u.startsWith('TARGET') ? GREEN : u.startsWith('TOO-HARD') ? '#f87171' : u.startsWith('SATURATED') ? AMBER : MUTED;
  };
  const y0 = 235, rh = 34;
  const body = rows.map((r, i) => {
    const y = y0 + i * rh;
    const pct = Math.round((r.rate ?? 0) * 100);
    return `<text x="64" y="${y}" font-size="20" fill="${TEXT}" font-family="${MONO}">${esc(r.family)}</text>
      <rect x="360" y="${y - 15}" width="${Math.max(2, pct * 4)}" height="16" rx="3" fill="url(#g)"/>
      <text x="800" y="${y}" font-size="20" fill="${MUTED}" font-family="${MONO}">${pct}%</text>
      <text x="900" y="${y}" font-size="18" font-weight="700" fill="${bandColor(r.band)}">${esc((r.band || '').split(' ')[0])}</text>`;
  }).join('');
  return frame('Difficulty calibration', `${esc(d.who || 'model')} - per-family pass rate vs the 2-3% hard-band floor labs cite`, body);
}

function arenaCard(d) {
  const body = `
  <text x="${W / 2}" y="330" font-size="80" font-weight="800" fill="${TEXT}" text-anchor="middle" font-family="${MONO}">${esc(d.a)} <tspan fill="url(#g)">${d.winsA}</tspan> : <tspan fill="url(#g)">${d.winsB}</tspan> ${esc(d.b)}</text>
  <text x="${W / 2}" y="400" font-size="24" fill="${MUTED}" text-anchor="middle">${d.draws} draws - deterministic judge: pass, then health score, then fewer tokens</text>`;
  return frame('Head to head, decided by the verifier', 'No human votes, no LLM judge; reproducible from the seed', body);
}

function leaderboardCard(d) {
  const board = (d.board || []).slice(0, 7);
  const y0 = 240, rh = 46;
  const body = board.map((b, i) => {
    const y = y0 + i * rh;
    const model = (d.models && d.models[b.provider]) || b.provider;
    return `<text x="64" y="${y}" font-size="24" fill="${DIM}" font-family="${MONO}">${i + 1}</text>
      <text x="120" y="${y}" font-size="24" font-weight="600" fill="${b.provider === 'reference' ? G2 : TEXT}" font-family="${MONO}">${esc(model)}</text>
      <text x="760" y="${y}" font-size="24" font-weight="700" fill="url(#g)" font-family="${MONO}">${b.passed}/${b.total}</text>
      <text x="1090" y="${y}" font-size="24" fill="${MUTED}" font-family="${MONO}" text-anchor="end">avg ${b.avgScore}</text>`;
  }).join('');
  return frame('Architecture-design leaderboard', 'Design a valid, budget-respecting network; graded by a deterministic verifier', body);
}

// ── Dispatch on shape ────────────────────────────────────────────────────────
let svg, kind;
if (Array.isArray(data.results) && data.results[0] && 'withVerifier' in data.results[0]) { svg = amplifyCard(data); kind = 'amplify'; }
else if ((Array.isArray(data.rows) && data.rows[0] && 'band' in data.rows[0]) || (Array.isArray(data.families) && data.families[0] && 'band' in data.families[0])) { svg = calibrateCard(data); kind = 'calibrate'; }
else if ('winsA' in data && 'winsB' in data) { svg = arenaCard(data); kind = 'arena'; }
else if (Array.isArray(data.board)) { svg = leaderboardCard(data); kind = 'leaderboard'; }
else { console.error('Unrecognized JSON. Expected a leaderboard / amplify / arena / calibrate output.'); process.exit(2); }

const outPath = outArg || inPath.replace(/\.json$/, '') + `-${kind}-card.svg`;
fs.writeFileSync(path.resolve(outPath), svg);
console.log(`Wrote ${outPath} (${kind} card). Embed in the README, or screenshot for X.`);
