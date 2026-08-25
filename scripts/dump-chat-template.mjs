/**
 * Print a GGUF's chat template and any reasoning-control hints in it.
 *
 * Reads metadata only — no weights are loaded and no inference runs.
 *
 *   node scripts/dump-chat-template.mjs <file.gguf> [--full]
 */

import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
execFileSync(process.execPath, [path.join(ROOT, 'scripts/build-tests.mjs')], { stdio: 'ignore' })
const { readGguf } = await import('./built/gguf.js')

const file = process.argv[2]
const full = process.argv.includes('--full')
if (!file) {
  console.error('usage: node scripts/dump-chat-template.mjs <file.gguf> [--full]')
  process.exit(1)
}

const meta = await readGguf(file)
const template = meta.kv?.['tokenizer.chat_template']

console.log(`\n${path.basename(file)}`)
console.log(`  arch: ${meta.kv?.['general.architecture']}`)
console.log(`  name: ${meta.kv?.['general.name'] ?? '—'}`)

if (typeof template !== 'string') {
  console.log('  no chat template in metadata')
  process.exit(0)
}

console.log(`  template: ${template.length} chars`)

// What the template actually reacts to.
const probes = [
  ['reasoning_effort', /reasoning_effort/g],
  ['enable_thinking', /enable_thinking/g],
  ['thinking', /\bthinking\b/g],
  ['/think literal', /\/think\b/g],
  ['/no_think literal', /\/no_?think\b/g],
  ['<think> tag', /<think>/g],
  ['reasoning_budget', /reasoning_budget/g]
]
console.log('\n  signals:')
for (const [label, re] of probes) {
  const n = (template.match(re) ?? []).length
  if (n) console.log(`    ${label.padEnd(20)} ${n}`)
}

// Any string literal compared against a variable is a candidate level name.
const levels = new Set()
for (const m of template.matchAll(/reasoning_effort\s*(?:==|!=|in)\s*([^%}]+)/g)) {
  for (const s of m[1].matchAll(/['"]([a-z_]+)['"]/g)) levels.add(s[1])
}
for (const m of template.matchAll(/['"]([a-z_]+)['"]\s*(?:==|!=)\s*reasoning_effort/g)) levels.add(m[1])
if (levels.size) console.log(`\n  effort levels found: ${[...levels].join(', ')}`)

if (full) {
  console.log('\n--- template ---\n')
  console.log(template)
} else {
  // The lines that mention reasoning, with a little context.
  const lines = template.split('\n')
  const hits = lines
    .map((l, i) => [i, l])
    .filter(([, l]) => /reasoning_effort|enable_thinking|think/i.test(l))
  if (hits.length) {
    console.log('\n  matching lines:')
    for (const [i, l] of hits.slice(0, 24)) console.log(`   ${String(i).padStart(4)}| ${l.trim().slice(0, 150)}`)
    if (hits.length > 24) console.log(`   … ${hits.length - 24} more`)
  }
}
