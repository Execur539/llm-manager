// Predicted context per quant on this machine, using geometry measured from the real model.
import { detectHardware } from './built/gpu.js'
import { planFit, DEFAULT_CONSTRAINTS, fmtBytes } from './built/engine.js'

const GB = 1024 ** 3

// Measured from ggml-org/Qwen3.8-27B-GGUF; matches the 65-layer / 17-attention split the
// unsloth build reports. Hybrid: only attention layers grow a KV cache.
const base = {
  architecture: 'qwen35', name: 'Qwen3.8-27B', blockCount: 65,
  embeddingLength: 5120, headCount: 24, headCountKv: 4, headDim: 256,
  contextLength: 262144, vocabSize: 248320, quant: 'Q4_K', expertCount: 0,
  attentionLayers: 17, ssmLayers: 48, ssmStateBytesPerLayer: 3211264
}

const MMPROJ = 0.59 * GB

const options = [
  ['UD-IQ4_XS ', 13.3], ['UD-Q4_K_S ', 14.3], ['Q4_0      ', 15.0],
  ['UD-Q4_K_M ', 15.3], ['Q4_1      ', 16.3], ['UD-Q4_K_XL', 16.4],
  ['(Q5_K_M)  ', 19.5], ['(Q6_K)    ', 22.5]
]

const hw = await detectHardware()
const nv = hw.gpus.filter((g) => g.vendor === 'nvidia')
console.log(`\nGPUs: ${nv.map((g) => `${g.name} ${fmtBytes(g.freeVram)} free`).join('  +  ')}`)
console.log(`Projector reserved on primary: ${fmtBytes(MMPROJ)}\n`)
console.log('  quant          size      context     KV      layers    note')
console.log('  ' + '-'.repeat(68))

for (const [label, gb] of options) {
  const weightBytes = gb * GB
  const arch = {
    ...base,
    weightBytes,
    perLayerBytes: weightBytes * 0.907,
    nonLayerBytes: weightBytes * 0.093
  }
  const r = planFit(arch, hw, { ...DEFAULT_CONSTRAINTS, companionBytes: MMPROJ })
  const p = r.chosen ?? r.alternatives[0]
  if (!p) { console.log(`  ${label}  ${gb.toFixed(1)} GB    no workable plan`); continue }
  const note = p.gpuLayers < p.totalLayers ? 'partial CPU offload' : p.contextLength >= 131072 ? 'full context' : ''
  console.log(
    `  ${label}   ${gb.toFixed(1)} GB   ${p.contextLength.toLocaleString().padStart(8)}   ${p.kvType}   ` +
    `${String(p.gpuLayers).padStart(2)}/${p.totalLayers}     ${note}`
  )
}
console.log('')
