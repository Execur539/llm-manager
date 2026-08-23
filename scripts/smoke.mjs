import { planFit, kvCacheBytes, proportionalSplit, DEFAULT_CONSTRAINTS, fmtBytes } from './built/engine.js'
import { toolCallGrammar } from './built/gbnf.js'

const GB = 1024 ** 3
let pass = 0, fail = 0
const check = (name, cond, extra = '') => {
  if (cond) { pass++; console.log(`  PASS  ${name}`) }
  else { fail++; console.log(`  FAIL  ${name} ${extra}`) }
}

// A Qwen3.8-27B-shaped dense model: 62 layers, GQA 8 KV heads, head_dim 128, 262K trained ctx.
const arch = {
  architecture: 'qwen3', name: 'Qwen3.8-27B', blockCount: 62, embeddingLength: 5120,
  headCount: 40, headCountKv: 8, headDim: 128, contextLength: 262144, vocabSize: 152064,
  quant: 'Q4_K', weightBytes: 16 * GB, perLayerBytes: 15 * GB, nonLayerBytes: 1 * GB, expertCount: 0
}
const hw = (gpus) => ({ gpus, totalRam: 64 * GB, freeRam: 40 * GB, cpuName: 'test', cpuThreads: 16, backend: 'cuda', takenAt: Date.now() })
const gpu = (name, totalGb, freeGb, measured = true) => ({
  index: 0, name, vendor: 'nvidia', totalVram: totalGb * GB,
  freeVram: freeGb < 0 ? -1 : freeGb * GB, utilisation: 5, freeIsMeasured: measured
})

console.log('\nKV cache maths')
const kv128 = kvCacheBytes(arch, 131072, 'q8_0')
// 2 * 8 heads * 128 dim * 62 layers * 131072 tokens * 1.0625 bytes
const expected = 2 * 8 * 128 * 62 * 131072 * (34 / 32)
check('128K q8_0 KV matches closed form', Math.abs(kv128 - expected) < 1, `${kv128} vs ${expected}`)
check('q4_0 is ~half of q8_0', Math.abs(kvCacheBytes(arch, 131072, 'q4_0') / kv128 - 0.529) < 0.01)
console.log(`  128K KV at q8_0 = ${fmtBytes(kv128)}`)

console.log('\nP1: sizes against FREE vram, not total')
// 24 GB card with only 6 GB free: a total-based engine would happily plan a 16 GB model.
const tight = planFit(arch, hw([gpu('RTX 4090', 24, 6)]), DEFAULT_CONSTRAINTS)
check('does not claim a full-GPU fit with 6 GB free', !(tight.chosen && !tight.chosen.spillsToHost))
const roomy = planFit(arch, hw([gpu('RTX 4090', 24, 23)]), DEFAULT_CONSTRAINTS)
check('same card with 23 GB free does fit', !!roomy.chosen)
if (roomy.chosen) console.log(`  chose ${roomy.chosen.contextLength.toLocaleString()} ctx at ${roomy.chosen.kvType}, ${roomy.chosen.gpuLayers}/${roomy.chosen.totalLayers} layers on GPU`)

console.log('\nP2: proportional multi-GPU split')
const mixed = planFit(arch, hw([gpu('RTX 4090', 24, 22), { ...gpu('RTX 3070', 8, 7), index: 1 }]), DEFAULT_CONSTRAINTS)
const split = (mixed.chosen ?? mixed.alternatives[0])?.tensorSplit ?? []
check('split is not 50/50', split.length === 2 && Math.abs(split[0] - 0.5) > 0.1, JSON.stringify(split))
check('big card gets the larger share', split[0] > split[1])
console.log(`  split = ${split.map((s) => (s * 100).toFixed(1) + '%').join(' / ')}`)
check('split sums to 1', Math.abs(split.reduce((a, b) => a + b, 0) - 1) < 1e-9)

console.log('\nNever silently degrade')
const cramped = planFit(arch, hw([gpu('RTX 3060', 12, 11)]), DEFAULT_CONSTRAINTS)
check('flags that the user must choose', cramped.needsUserChoice === true)
check('offers real alternatives', cramped.alternatives.length >= 1)
cramped.alternatives.forEach((a) => console.log(`  option: ${a.label} — ${a.contextLength.toLocaleString()} ctx, ${a.gpuLayers}/${a.totalLayers} layers, speed ${a.speedScore}`))

console.log('\nUnmeasured free VRAM is handled conservatively')
const amd = planFit(arch, hw([{ ...gpu('Radeon RX 7900', 24, -1, false), vendor: 'amd' }]), DEFAULT_CONSTRAINTS)
check('notes that free VRAM was estimated', amd.notes.some((n) => /could not be measured/i.test(n)))

console.log('\nKV floor is respected')
const all = [cramped.chosen, ...cramped.alternatives].filter(Boolean)
check('never quantises below q4_0', all.every((p) => ['f16', 'q8_0', 'q4_0'].includes(p.kvType)))

console.log('\nP4: full-context KV reserved up front')
if (roomy.chosen) {
  const reserved = kvCacheBytes(arch, roomy.chosen.contextLength, roomy.chosen.kvType)
  check('plan.kvBytes equals full-context KV', Math.abs(roomy.chosen.kvBytes - reserved) < 1)
}

console.log('\nGBNF compiler')
const grammar = toolCallGrammar([
  { name: 'read_file', parameters: { type: 'object', properties: { path: { type: 'string' }, limit: { type: 'integer' } }, required: ['path'] } },
  { name: 'run_command', parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] } }
])
check('has a root rule', /^root ::= /m.test(grammar))
check('pins the tool name as a literal', grammar.includes('\\"read_file\\"'))
check('emits both tool branches', (grammar.match(/call-\d+/g) ?? []).length >= 2)
check('defines JSON primitives', grammar.includes('string      ::=') && grammar.includes('number      ::='))

console.log(`\n${pass} passed, ${fail} failed\n`)
process.exit(fail ? 1 : 0)
