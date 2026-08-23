// Runs detection and the fit engine against this machine's real hardware.
import { detectHardware, refreshFreeVram } from './built/gpu.js'
import { planFit, DEFAULT_CONSTRAINTS, fmtBytes, kvCacheBytes } from './built/engine.js'

const GB = 1024 ** 3
const hw = await detectHardware()

console.log(`\nBackend: ${hw.backend.toUpperCase()}   CPU: ${hw.cpuName} (${hw.cpuThreads}t)`)
console.log(`RAM: ${fmtBytes(hw.freeRam)} free of ${fmtBytes(hw.totalRam)}\n`)
console.log('GPUs detected:')
for (const g of hw.gpus) {
  console.log(
    `  [${g.index}] ${g.name.padEnd(32)} ${fmtBytes(g.totalVram).padStart(9)} total  ` +
    `${(g.freeVram >= 0 ? fmtBytes(g.freeVram) : 'unmeasured').padStart(9)} free  ` +
    `measured=${g.freeIsMeasured}  util=${g.utilisation}%`
  )
}

// Real models, sized from their actual published geometry.
const models = [
  { label: 'Qwen3.8-27B Q4_K_M', blockCount: 62, embeddingLength: 5120, headCount: 40, headCountKv: 8,
    headDim: 128, contextLength: 262144, vocabSize: 152064, weightBytes: 16.5 * GB },
  { label: 'Llama-3.1-8B Q5_K_M', blockCount: 32, embeddingLength: 4096, headCount: 32, headCountKv: 8,
    headDim: 128, contextLength: 131072, vocabSize: 128256, weightBytes: 5.7 * GB },
  { label: 'Llama-3.3-70B Q4_K_M', blockCount: 80, embeddingLength: 8192, headCount: 64, headCountKv: 8,
    headDim: 128, contextLength: 131072, vocabSize: 128256, weightBytes: 42.5 * GB }
]

for (const m of models) {
  const arch = { ...m, architecture: 'test', name: m.label, quant: 'Q4_K', expertCount: 0,
    perLayerBytes: m.weightBytes * 0.94, nonLayerBytes: m.weightBytes * 0.06 }

  console.log(`\n${'─'.repeat(66)}\n${m.label}  (${fmtBytes(m.weightBytes)} weights, ${m.blockCount} layers, GQA ${m.headCount}/${m.headCountKv})`)
  const result = planFit(arch, hw, DEFAULT_CONSTRAINTS)

  for (const n of result.notes) console.log(`  note: ${n}`)

  const show = (p, tag) => {
    console.log(`  ${tag} ${p.label}: ${p.contextLength.toLocaleString()} ctx @ ${p.kvType}, ` +
      `${p.gpuLayers}/${p.totalLayers} layers on GPU, speed ${p.speedScore}/100`)
    console.log(`      KV ${fmtBytes(p.kvBytes)}  |  per-GPU ${p.predictedVramPerGpu.map(fmtBytes).join(' + ')}` +
      (p.spillsToHost ? `  |  host ${fmtBytes(p.predictedHostBytes)}` : ''))
    if (p.tensorSplit.length > 1) console.log(`      split ${p.tensorSplit.map(s => (s*100).toFixed(1)+'%').join(' / ')}`)
  }

  if (result.chosen) show(result.chosen, '✓')
  if (result.needsUserChoice) {
    console.log('  → target unreachable; offering tradeoffs instead of degrading silently:')
    result.alternatives.forEach(p => show(p, '  •'))
  }
}
console.log('')
