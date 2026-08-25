/**
 * Run the reasoning detector against real GGUF chat templates.
 *
 * Synthetic fixtures would only prove the regexes match what I wrote them against. These are
 * templates shipped by the model authors, which is where the awkward cases live — aliases,
 * derived variables, validation tuples.
 *
 *   node scripts/reasoning-check.mjs <file.gguf> [more.gguf ...]
 */

import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
execFileSync(process.execPath, [path.join(ROOT, 'scripts/build-tests.mjs')], { stdio: 'ignore' })
const { readGguf } = await import('./built/gguf.js')
const { detectReasoning } = await import('./built/reasoning.js')

const files = process.argv.slice(2)
if (!files.length) {
  console.error('usage: node scripts/reasoning-check.mjs <file.gguf> [...]')
  process.exit(1)
}

for (const file of files) {
  let template
  try {
    template = (await readGguf(file)).kv?.['tokenizer.chat_template']
  } catch (err) {
    console.log(`${path.basename(file)}\n  unreadable: ${err.message}\n`)
    continue
  }

  const r = detectReasoning(template)
  console.log(path.basename(file))
  console.log(`  kind:     ${r.kind}`)
  if (r.kind === 'effort') {
    console.log(`  levels:   ${r.levels.join(' < ')}`)
    console.log(`  default:  ${r.defaultLevel}`)
  }
  console.log(`  canDisable: ${r.canDisable}`)
  console.log()
}
