// Plans a fit for a real GGUF on this machine's real hardware.
import { readGguf, extractArchInfo } from './built/gguf.js'
import { detectHardware } from './built/gpu.js'
import { planFit, DEFAULT_CONSTRAINTS, fmtBytes, kvCacheBytes } from './built/engine.js'

const meta = await readGguf(process.argv[2])
const arch = extractArchInfo(meta)
const hw = await detectHardware()

console.log(`\n${arch.name}  (${arch.architecture}, ${arch.quant})`)
console.log(`  ${arch.blockCount} blocks: ${arch.attentionLayers} attention + ${arch.ssmLayers} SSM`)
console.log(`  GQA ${arch.headCount}/${arch.headCountKv} heads, head_dim ${arch.headDim}`)
console.log(`  weights ${fmtBytes(arch.weightBytes)}`)

const naive = 2 * arch.headCountKv * arch.headDim * arch.blockCount * 131072 * (34/32)
console.log(`\nKV at 128K (q8_0):`)
console.log(`  counting all ${arch.blockCount} layers:  ${fmtBytes(naive)}   <- what a naive estimate gives`)
console.log(`  counting ${arch.attentionLayers} attention layers: ${fmtBytes(kvCacheBytes(arch, 131072, 'q8_0'))}   <- actual`)

console.log(`\nGPUs: ${hw.gpus.filter(g=>g.vendor==='nvidia').map(g => `${g.name} (${fmtBytes(g.freeVram)} free)`).join(', ')}`)

const r = planFit(arch, hw, DEFAULT_CONSTRAINTS)
for (const n of r.notes) console.log(`\nnote: ${n}`)

const show = (p) => {
  console.log(`\n  ${p.label}: ${p.contextLength.toLocaleString()} ctx @ ${p.kvType}, ${p.gpuLayers}/${p.totalLayers} layers on GPU`)
  console.log(`     KV ${fmtBytes(p.kvBytes)}  per-GPU ${p.predictedVramPerGpu.map(fmtBytes).join(' + ')}  split ${p.tensorSplit.map(s=>(s*100).toFixed(1)+'%').join('/')}`)
  p.rationale.forEach(x => console.log(`     - ${x}`))
}
if (r.chosen) show(r.chosen)
if (r.needsUserChoice) { console.log('\ntradeoffs:'); r.alternatives.forEach(show) }
console.log('')
