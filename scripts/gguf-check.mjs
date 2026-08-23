// Parses a real GGUF file and prints what the auto-fit engine would see.
import { readGguf, extractArchInfo, templateSupportsTools } from './built/gguf.js'

const file = process.argv[2]
const meta = await readGguf(file)
const arch = extractArchInfo(meta)

console.log(`\nGGUF v${meta.version}  ${meta.tensorCount} tensors  ${Object.keys(meta.kv).length} metadata keys`)
console.log(`data starts at byte ${meta.dataOffset.toLocaleString()}\n`)

console.log('Architecture facts the fit engine uses:')
for (const [k, v] of Object.entries(arch)) {
  const shown = typeof v === 'number' && v > 100000 ? v.toLocaleString() : v
  console.log(`  ${k.padEnd(18)} ${shown}`)
}

console.log('\nSelected metadata:')
for (const k of Object.keys(meta.kv).sort()) {
  const v = meta.kv[k]
  if (v && typeof v === 'object' && 'elided' in v) {
    console.log(`  ${k.padEnd(42)} <array of ${v.count.toLocaleString()} elided>`)
  } else if (typeof v === 'string' && v.length > 60) {
    console.log(`  ${k.padEnd(42)} "${v.slice(0, 55)}…" (${v.length} chars)`)
  } else if (!Array.isArray(v)) {
    console.log(`  ${k.padEnd(42)} ${v}`)
  } else {
    console.log(`  ${k.padEnd(42)} [${v.length} items]`)
  }
}

console.log(`\nchat template mentions tools: ${templateSupportsTools(meta)}`)
console.log(`\nFirst 5 tensors:`)
for (const t of meta.tensors.slice(0, 5)) {
  console.log(`  ${t.name.padEnd(34)} ${JSON.stringify(t.dims).padEnd(16)} type=${t.ggmlType} ${(t.bytes / 1024 ** 2).toFixed(1)} MB`)
}
