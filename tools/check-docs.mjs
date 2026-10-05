import { readFileSync, existsSync, readdirSync } from 'node:fs';

/**
 * Documentation integrity check: every repository path the README names in
 * backticks must exist, and every tool it mentions must be present. Catches the
 * class of documentation rot where the code moved and the prose did not.
 */
const md = readFileSync('README.md', 'utf8');

const TICK = String.fromCharCode(96);
const pathRe = new RegExp(TICK + '((?:src|tools|perf|dist)/[\\w./-]+)' + TICK, 'g');
const paths = [...new Set([...md.matchAll(pathRe)].map((m) => m[1]))];
const missing = paths.filter((p) => !existsSync(p));

console.log(`README: ${md.split('\n').length} lines`);
console.log(`referenced repo paths: ${paths.length}, missing: ${missing.length}`);
if (missing.length) console.log('  MISSING:', missing.join(', '));

const toolRe = /tools\/([\w.-]+\.mjs)/g;
const mentioned = [...new Set([...md.matchAll(toolRe)].map((m) => m[1]))];
const actual = readdirSync('tools');
console.log(`tools mentioned: ${mentioned.join(', ')}`);
console.log(`tools present  : ${actual.join(', ')}`);
const absent = mentioned.filter((t) => !actual.includes(t));
if (absent.length) console.log('  MENTIONED BUT ABSENT:', absent.join(', '));

// Scripts declared in package.json must resolve too.
const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
console.log('package scripts:', Object.entries(pkg.scripts).map(([k, v]) => `${k}="${v}"`).join('  '));
console.log('declared deps  :', Object.keys({ ...pkg.dependencies, ...pkg.devDependencies }).join(', '));
