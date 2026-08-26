/**
 * Smoke tests for the pure logic: auto-fit maths, GBNF compilation, the permission gate,
 * the GGUF parser, and quant recommendation.
 *
 * Run with `npm test`. These do not need Electron, a GPU, or a model file — anything that
 * needs a real runtime is verified by hand and recorded in BUILD_STATUS.md instead.
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// Rebuild the test bundles so the suite always runs against current source.
execFileSync(process.execPath, [path.join(ROOT, 'scripts/build-tests.mjs')], { stdio: 'inherit' })

const { planFit, kvCacheBytes, computeBufferBytes, proportionalSplit, DEFAULT_CONSTRAINTS, fmtBytes, verifyPrediction } =
  await import('./built/engine.js')
const { toolCallGrammar, schemaGrammar } = await import('./built/gbnf.js')
const { checkHardBlock, describeCall, PermissionEngine } = await import('./built/permissions.js')
const { readGguf, extractArchInfo, tensorByteSize, isKnownGgmlType } = await import('./built/gguf.js')
const { recommendQuant, findMmprojFor } = await import('./built/hf.js')
 const { isVirtualAdapter } = await import('./built/gpu.js')
const { exportFilename, uniquePath } = await import('./built/filenames.js')
const { detectReasoning, reasoningRequestFields } = await import('./built/reasoning.js')

const GB = 1024 ** 3
let pass = 0
let fail = 0
const failures = []

function check(name, cond, extra = '') {
  if (cond) {
    pass++
    console.log(`  PASS  ${name}`)
  } else {
    fail++
    failures.push(name)
    console.log(`  FAIL  ${name} ${extra}`)
  }
}

function section(title) {
  console.log(`\n${title}`)
}

// ---------------------------------------------------------------- fixtures

// Qwen3.8-27B-shaped: dense 27B, 62 layers, GQA with 8 KV heads, 262K trained context.
const arch = {
  architecture: 'qwen3',
  name: 'Qwen3.8-27B',
  blockCount: 62,
  embeddingLength: 5120,
  headCount: 40,
  headCountKv: 8,
  headDim: 128,
  contextLength: 262144,
  vocabSize: 152064,
  quant: 'Q4_K',
  weightBytes: 16 * GB,
  perLayerBytes: 15 * GB,
  nonLayerBytes: 1 * GB,
  expertCount: 0
}

const hw = (gpus, backend = 'cuda') => ({
  gpus,
  totalRam: 64 * GB,
  freeRam: 40 * GB,
  cpuName: 'test cpu',
  cpuThreads: 16,
  backend,
  takenAt: Date.now()
})

const gpu = (name, totalGb, freeGb, measured = true, index = 0) => ({
  index,
  name,
  vendor: 'nvidia',
  totalVram: totalGb * GB,
  freeVram: freeGb < 0 ? -1 : freeGb * GB,
  utilisation: 5,
  freeIsMeasured: measured
})

// ---------------------------------------------------------------- auto-fit

section('KV cache maths')
{
  const kv128 = kvCacheBytes(arch, 131072, 'q8_0')
  const expected = 2 * 8 * 128 * 62 * 131072 * (34 / 32)
  check('128K q8_0 KV matches the closed form', Math.abs(kv128 - expected) < 1)
  check('q4_0 is ~53% of q8_0', Math.abs(kvCacheBytes(arch, 131072, 'q4_0') / kv128 - 0.529) < 0.01)
  check('KV scales linearly with context', Math.abs(kvCacheBytes(arch, 2048, 'f16') * 2 - kvCacheBytes(arch, 4096, 'f16')) < 1)
  console.log(`  128K at q8_0 = ${fmtBytes(kv128)}`)
}

section('Hybrid attention/SSM models cache only on attention layers')
{
  // Real geometry from ggml-org/Qwen3.8-27B-GGUF: 64 blocks, full_attention_interval 4,
  // so 16 attention layers and 48 state-space layers. Counting all 64 overestimates the KV
  // cache ~3.9x and costs most of the achievable context.
  const hybrid = {
    ...arch,
    architecture: 'qwen35',
    blockCount: 64,
    headCount: 24,
    headCountKv: 4,
    headDim: 256,
    attentionLayers: 16,
    ssmLayers: 48,
    ssmStateBytesPerLayer: 3211264,
    weightBytes: 17.66 * GB,
    perLayerBytes: 16 * GB,
    nonLayerBytes: 1.66 * GB
  }

  const naive = 2 * hybrid.headCountKv * hybrid.headDim * hybrid.blockCount * 131072 * (34 / 32)
  const actual = kvCacheBytes(hybrid, 131072, 'q8_0')
  check('hybrid KV is far below the all-layers estimate', actual < naive / 3, `${fmtBytes(actual)} vs ${fmtBytes(naive)}`)

  // The SSM state is real memory, but constant — doubling context must not double it.
  // Asking for zero context isolates the state, so this holds whatever safety factor is applied.
  const stateOnly = kvCacheBytes(hybrid, 0, 'q8_0')
  const at64k = kvCacheBytes(hybrid, 65536, 'q8_0')
  const at128k = kvCacheBytes(hybrid, 131072, 'q8_0')
  check('SSM state is included', stateOnly > 0 && at64k > stateOnly)
  check('SSM state does not scale with context', Math.abs((at128k - at64k) - (at64k - stateOnly)) < 1)

  // A dense model of the same block count must be unaffected by the hybrid path.
  const dense = { ...hybrid, attentionLayers: 64, ssmLayers: 0, ssmStateBytesPerLayer: 0 }
  check('dense models still count every layer', Math.abs(kvCacheBytes(dense, 131072, 'q8_0') - naive) < 1)

  // Metadata written before hybrid support leaves these unset; fall back to block count.
  const legacy = { ...dense, attentionLayers: undefined, ssmLayers: undefined, ssmStateBytesPerLayer: undefined }
  check('missing hybrid fields fall back to block count', Math.abs(kvCacheBytes(legacy, 131072, 'q8_0') - naive) < 1)

  // On this rig the fix is the difference between the ideal context and a degraded one.
  const rig = hw([gpu('RTX 5080', 16, 13.4), gpu('RTX 4070 Ti', 12, 11.7, true, 1)])
  const result = planFit(hybrid, rig, DEFAULT_CONSTRAINTS)
  check('hybrid 27B reaches the ideal context on a 5080+4070Ti', result.chosen?.contextLength === 131072, `${result.chosen?.contextLength}`)
  check('at the preferred KV quality, not the floor', result.chosen?.kvType === 'q8_0')
  check('explains the hybrid layout', result.notes.some((n) => /hybrid architecture/i.test(n)))
  console.log(`  128K KV: ${fmtBytes(naive)} naive -> ${fmtBytes(actual)} actual`)
}

section('Compute buffer: flash attention is a precondition at long context')
{
  const withFa = computeBufferBytes(arch, 131072, 512, true)
  const withoutFa = computeBufferBytes(arch, 131072, 512, false)
  check('no-FA attention buffer is far larger at 128K', withoutFa > withFa * 10)
  check('FA buffer does not scale with context', computeBufferBytes(arch, 4096, 512, true) === withFa)
  console.log(`  128K compute buffer: FA on ${fmtBytes(withFa)}, FA off ${fmtBytes(withoutFa)}`)
}

section('P1 — sizes against FREE VRAM, not total')
{
  // A 24 GB card with only 6 GB free. An engine sizing off `total` would plan a 16 GB model on it.
  const tight = planFit(arch, hw([gpu('RTX 4090', 24, 6)]), DEFAULT_CONSTRAINTS)
  check('does not claim a full-GPU fit with 6 GB free', !(tight.chosen && !tight.chosen.spillsToHost))

  const roomy = planFit(arch, hw([gpu('RTX 4090', 24, 23)]), DEFAULT_CONSTRAINTS)
  check('the same card with 23 GB free does fit', !!roomy.chosen)
  check('fitted plan keeps every layer on GPU', roomy.chosen?.gpuLayers === arch.blockCount)
  if (roomy.chosen) {
    console.log(`  chose ${roomy.chosen.contextLength.toLocaleString()} ctx at ${roomy.chosen.kvType}`)
  }
}

section('P2 — proportional multi-GPU split, not an even one')
{
  const mixed = planFit(arch, hw([gpu('RTX 4090', 24, 22), gpu('RTX 3070', 8, 7, true, 1)]), DEFAULT_CONSTRAINTS)
  const split = (mixed.chosen ?? mixed.alternatives[0])?.tensorSplit ?? []
  check('split is not 50/50', split.length === 2 && Math.abs(split[0] - 0.5) > 0.1, JSON.stringify(split))
  check('the larger card gets the larger share', split[0] > split[1])
  check('split sums to 1', Math.abs(split.reduce((a, b) => a + b, 0) - 1) < 1e-9)
  console.log(`  split = ${split.map((s) => `${(s * 100).toFixed(1)}%`).join(' / ')}`)

  // Three mismatched cards, to be sure it is not just a two-element special case.
  const three = proportionalSplit([20 * GB, 10 * GB, 5 * GB])
  check('three-way split is proportional', Math.abs(three[0] - 4 / 7) < 0.01 && Math.abs(three[2] - 1 / 7) < 0.01)
}

section('P4 — full-context KV reserved up front')
{
  const roomy = planFit(arch, hw([gpu('RTX 4090', 24, 23)]), DEFAULT_CONSTRAINTS)
  if (roomy.chosen) {
    const reserved = kvCacheBytes(arch, roomy.chosen.contextLength, roomy.chosen.kvType)
    check('plan.kvBytes equals the whole context, not the current fill', Math.abs(roomy.chosen.kvBytes - reserved) < 1)
  } else {
    check('plan.kvBytes equals the whole context', false, '(no plan produced)')
  }
}

section('Never degrades silently')
{
  const cramped = planFit(arch, hw([gpu('RTX 3060', 12, 11)]), DEFAULT_CONSTRAINTS)
  check('flags that the user must choose', cramped.needsUserChoice === true)
  check('offers at least one real alternative', cramped.alternatives.length >= 1)
  check('every alternative carries a rationale', cramped.alternatives.every((a) => a.rationale.length > 0))
  check('no alternative quantises below the floor', cramped.alternatives.every((a) => ['f16', 'q8_0', 'q4_0'].includes(a.kvType)))
  cramped.alternatives.forEach((a) =>
    console.log(`  option: ${a.label} — ${a.contextLength.toLocaleString()} ctx, ${a.gpuLayers}/${a.totalLayers} layers, speed ${a.speedScore}`)
  )
}

section('Mixed rigs — only addressable devices take part')
{
  // Regression: this was found by running against a real machine that had an RTX 5080, an
  // RTX 4070 Ti, an AMD iGPU, and two virtual display adapters from screen-sharing tools.
  // The phantom devices stole budget share and cut a 27B model from 89K to 20K context.
  const mixed = hw([
    gpu('NVIDIA GeForce RTX 5080', 16, 13.2),
    gpu('NVIDIA GeForce RTX 4070 Ti', 12, 11.7, true, 1),
    { ...gpu('AMD Radeon(TM) Graphics', 2, -1, false, 2), vendor: 'amd' }
  ])

  const result = planFit(arch, mixed, DEFAULT_CONSTRAINTS)
  const plan = result.chosen ?? result.alternatives[0]

  check('CUDA backend excludes the AMD iGPU', plan?.tensorSplit.length === 2, JSON.stringify(plan?.tensorSplit))
  check('says why it excluded it', result.notes.some((n) => /excluded from the split.*AMD/i.test(n)))
  check('split still favours the bigger card', (plan?.tensorSplit[0] ?? 0) > (plan?.tensorSplit[1] ?? 1))
  check('the 27B fits fully on GPU on this rig', plan?.gpuLayers === arch.blockCount)
  check('reaches a useful context', (plan?.contextLength ?? 0) >= 65536, `${plan?.contextLength}`)
  console.log(`  ${plan?.contextLength.toLocaleString()} ctx across ${plan?.tensorSplit.length} GPUs`)

  // Under Vulkan an iGPU is usable, but not while discrete cards are present.
  const vulkanMixed = planFit(arch, { ...mixed, backend: 'vulkan' }, DEFAULT_CONSTRAINTS)
  const vplan = vulkanMixed.chosen ?? vulkanMixed.alternatives[0]
  check('Vulkan also skips the iGPU when discrete cards exist', vplan?.tensorSplit.length === 2)

  const igpuOnly = planFit(
    { ...arch, weightBytes: 1 * GB, perLayerBytes: 0.9 * GB, nonLayerBytes: 0.1 * GB, blockCount: 8 },
    hw([{ ...gpu('AMD Radeon(TM) Graphics', 8, -1, false), vendor: 'amd' }], 'vulkan'),
    DEFAULT_CONSTRAINTS
  )
  const iplan = igpuOnly.chosen ?? igpuOnly.alternatives[0]
  check('an iGPU is used when it is the only device', (iplan?.tensorSplit.length ?? 0) === 1)
}

section('Virtual display adapters are not GPUs')
{
  // Screen-sharing tools install display adapters that WMI reports like GPUs.
  const virtual = [
    'Virtual Desktop Monitor',
    'Parsec Virtual Display Adapter',
    'IDD HDR',
    'Sunshine Gamestream Mirror',
    'Microsoft Basic Display Adapter'
  ]
  for (const name of virtual) {
    check(`rejects phantom adapter: ${name}`, isVirtualAdapter(name))
  }

  const real = ['NVIDIA GeForce RTX 5080', 'AMD Radeon RX 7900 XTX', 'Intel Arc A770', 'AMD Radeon(TM) Graphics']
  for (const name of real) {
    check(`keeps real adapter: ${name}`, !isVirtualAdapter(name))
  }
}

section('Honest about unmeasurable hardware')
{
  const amd = planFit(arch, hw([{ ...gpu('Radeon RX 7900 XTX', 24, -1, false), vendor: 'amd' }], 'vulkan'), DEFAULT_CONSTRAINTS)
  check('notes that free VRAM was estimated', amd.notes.some((n) => /could not be measured/i.test(n)))

  const cpuOnly = planFit(arch, hw([], 'cpu'), DEFAULT_CONSTRAINTS)
  check('falls back to a CPU plan with no GPU', cpuOnly.chosen?.gpuLayers === 0)
}

section('User overrides are respected, not overwritten')
{
  const forced = planFit(arch, hw([gpu('RTX 4090', 24, 23)]), {
    ...DEFAULT_CONSTRAINTS,
    overrides: { contextLength: 8192, kvType: 'f16', batchSize: 256, flashAttention: false }
  })
  const plan = forced.chosen ?? forced.alternatives[0]
  check('honours the KV type override', plan?.kvType === 'f16')
  check('honours the batch size override', plan?.batchSize === 256)
  check('honours the flash-attention override', plan?.flashAttention === false)
  check('does not exceed the context override', (plan?.contextLength ?? 0) <= 8192)
  check('warns that flash attention is off', forced.notes.some((n) => /flash attention is off/i.test(n)))
}

section('Trained-context ceiling')
{
  const shortCtx = planFit({ ...arch, contextLength: 8192 }, hw([gpu('RTX 4090', 24, 23)]), DEFAULT_CONSTRAINTS)
  const plan = shortCtx.chosen ?? shortCtx.alternatives[0]
  check('never plans beyond the trained context', (plan?.contextLength ?? 0) <= 8192)
  check('says why it capped', shortCtx.notes.some((n) => /trained for/i.test(n)))
}

section('Prediction verification feeds back')
{
  const plan = { predictedVramPerGpu: [10 * GB] }
  check('flags a large under-prediction', !!verifyPrediction(plan, [13 * GB]).suggestion)
  check('flags a large over-prediction', !!verifyPrediction(plan, [7 * GB]).suggestion)
  check('stays quiet when close', verifyPrediction(plan, [10.2 * GB]).suggestion === null)
}

// ---------------------------------------------------------------- GBNF

section('GBNF compiler')
{
  const grammar = toolCallGrammar([
    {
      name: 'read_file',
      parameters: { type: 'object', properties: { path: { type: 'string' }, limit: { type: 'integer' } }, required: ['path'] }
    },
    { name: 'run_command', parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] } }
  ])
  check('has a root rule', /^root ::= /m.test(grammar))
  check('pins each tool name as a literal', grammar.includes('\\"read_file\\"') && grammar.includes('\\"run_command\\"'))
  check('emits one branch per tool', (grammar.match(/call-\d+ ::=/g) ?? []).length === 2)
  check('defines JSON primitives', grammar.includes('string      ::=') && grammar.includes('number      ::='))
  // Strip every optional group; a required key must survive, an optional one must not.
  const withoutOptionals = grammar.replace(/\([^()]*\)\?/g, '')
  check('required fields stay mandatory', withoutOptionals.includes('\\"path\\"'))
  check('optional fields are wrapped in an optional group', !withoutOptionals.includes('\\"limit\\"'))

  const enumGrammar = schemaGrammar({ type: 'object', properties: { mode: { type: 'string', enum: ['fast', 'slow'] } } })
  check('compiles enums to alternations', enumGrammar.includes('fast') && enumGrammar.includes('slow'))

  const arrayGrammar = schemaGrammar({ type: 'object', properties: { items: { type: 'array', items: { type: 'string' } } } })
  check('compiles arrays', /arr-\d+ ::= "\["/.test(arrayGrammar))
}

// ---------------------------------------------------------------- permissions

section('Hard-block list')
{
  const blocked = [
    'format C:',
    'diskpart /s script.txt',
    'rm -rf /',
    'Set-MpPreference -DisableRealtimeMonitoring $true',
    'bcdedit /set nointegritychecks on',
    'vssadmin delete shadows /all',
    'shutdown /r /t 0'
  ]
  for (const cmd of blocked) {
    check(`blocks: ${cmd.slice(0, 40)}`, checkHardBlock(cmd).blocked)
  }

  const allowed = [
    'npm install',
    'git status',
    'rm -rf ./node_modules',
    'Remove-Item -Recurse .\\dist',
    'python train.py --format csv'
  ]
  for (const cmd of allowed) {
    check(`allows: ${cmd.slice(0, 40)}`, !checkHardBlock(cmd).blocked)
  }
}

section('Permission tiers')
{
  const asked = []
  const engine = new PermissionEngine({
    hardBlocksDisabled: () => false,
    ask: async (req) => {
      asked.push(req.tool)
      return 'allow-tool'
    }
  })
  const cwd = process.cwd()

  const read = await engine.authorise('read_file', 'read', { path: 'x.txt' }, cwd)
  check('read-tier runs without prompting', read.allowed && asked.length === 0)

  const write = await engine.authorise('write_file', 'write', { path: 'x.txt', content: 'hi' }, cwd)
  check('write-tier prompts', write.allowed && asked.length === 1)

  const again = await engine.authorise('write_file', 'write', { path: 'y.txt', content: 'hi' }, cwd)
  check('allow-tool is remembered for later calls', again.allowed && asked.length === 1)

  const hard = await engine.authorise('run_command', 'execute', { command: 'format C:' }, cwd)
  check('hard blocks refuse without prompting', !hard.allowed && asked.length === 1)
  check('hard block explains itself', /hard-block/i.test(hard.reason ?? ''))

  const denier = new PermissionEngine({ hardBlocksDisabled: () => false, ask: async () => 'deny' })
  const denied = await denier.authorise('delete_file', 'write', { path: 'x' }, cwd)
  check('denial is reported back to the model', !denied.allowed && /denied/i.test(denied.reason ?? ''))

  const override = new PermissionEngine({ hardBlocksDisabled: () => true, ask: async () => 'allow-once' })
  const overridden = await override.authorise('run_command', 'execute', { command: 'format C:' }, cwd)
  check('hard blocks can be overridden when explicitly disabled', overridden.allowed)
}

section('Approval prompts show resolved targets')
{
  const described = describeCall('write_file', { path: 'sub/../out.txt', content: 'x' }, 'C:\\work')
  check('relative paths are resolved before display', described.includes('out.txt') && !described.includes('..'))
  check('byte count is shown for writes', /\d+ bytes/.test(described))

  const cmd = describeCall('run_command', { command: 'npm test' }, 'C:\\work')
  check('commands are shown verbatim', cmd.includes('npm test') && cmd.includes('C:\\work'))
}

// ---------------------------------------------------------------- GGUF

section('GGUF parser')
{
  check('Q4_K tensor sizing matches the block layout', tensorByteSize(12, [256, 10]) === 10 * 144)
  check('F16 tensor sizing', tensorByteSize(1, [100, 100]) === 20000)
  check('unknown types degrade instead of throwing', tensorByteSize(999, [10]) === 10)

  // Build a minimal but valid GGUF file and read it back.
  const file = path.join(os.tmpdir(), `llmm-test-${Date.now()}.gguf`)
  fs.writeFileSync(file, buildTestGguf())
  try {
    const meta = await readGguf(file)
    check('reads the header', meta.version === 3 && meta.tensorCount === 2)
    check('reads string metadata', meta.kv['general.architecture'] === 'llama')
    check('reads numeric metadata', meta.kv['llama.block_count'] === 2)
    check('elides oversized arrays', meta.kv['tokenizer.ggml.tokens']?.elided === true)
    check('reads the tensor directory', meta.tensors.length === 2 && meta.tensors[0].name === 'blk.0.attn_q.weight')

    const info = extractArchInfo(meta)
    check('extracts architecture', info.architecture === 'llama')
    check('derives head_dim from embedding/heads', info.headDim === 64)
    check('separates per-layer from non-layer weights', info.perLayerBytes > 0 && info.nonLayerBytes > 0)
    check('takes vocab size from the elided token array', info.vocabSize === 32000)
  } finally {
    fs.rmSync(file, { force: true })
  }
}

// ---------------------------------------------------------------- recommendation

section('Unknown quant types reconcile against file size')
{
  // llama.cpp keeps adding tensor types (TQ1_0, TQ2_0, MXFP4, Q1_0...). A type this build has no
  // block layout for would otherwise be counted at 1 byte/element — an under-estimate, which is
  // the dangerous direction: the engine would plan more context than fits and the load OOMs.
  check('known types are recognised', isKnownGgmlType(12) && isKnownGgmlType(14))
  check('a future type is flagged unknown', !isKnownGgmlType(99))
  check('unknown types fall back to 1 byte/element', tensorByteSize(99, [1000]) === 1000)

  const file = path.join(os.tmpdir(), `llmm-unknown-${Date.now()}.gguf`)
  // Same fixture, but the weight tensors claim a type this build does not know.
  fs.writeFileSync(file, buildTestGguf(99))
  try {
    const meta = await readGguf(file)
    const naive = extractArchInfo(meta)
    // Pretend the real file is far larger than the naive 1-byte-per-element arithmetic suggests.
    const realSize = meta.dataOffset + 8 * 1024 ** 3
    const reconciled = extractArchInfo(meta, realSize)

    check('flags the unrecognised type', reconciled.unknownTensorTypes.includes(99))
    check('naive total under-counts', naive.weightBytes < 8 * 1024 ** 3)
    check(
      'reconciled total matches the file',
      Math.abs(reconciled.weightBytes - 8 * 1024 ** 3) < 1024,
      `${reconciled.weightBytes}`
    )
    check('per-layer and non-layer both scale', reconciled.perLayerBytes > naive.perLayerBytes && reconciled.nonLayerBytes > naive.nonLayerBytes)
    check('the split ratio is preserved', Math.abs(
      reconciled.perLayerBytes / reconciled.weightBytes - naive.perLayerBytes / naive.weightBytes
    ) < 1e-6)
  } finally {
    fs.rmSync(file, { force: true })
  }

  // A known-type file must be left alone: small deltas are padding, not error.
  const known = path.join(os.tmpdir(), `llmm-known-${Date.now()}.gguf`)
  fs.writeFileSync(known, buildTestGguf(12))
  try {
    const meta = await readGguf(known)
    const plain = extractArchInfo(meta)
    const withSize = extractArchInfo(meta, meta.dataOffset + plain.weightBytes)
    check('no unknown types on a normal file', withSize.unknownTensorTypes.length === 0)
    check('an accurate total is not rewritten', Math.abs(withSize.weightBytes - plain.weightBytes) < 1)
  } finally {
    fs.rmSync(known, { force: true })
  }
}

section('Quant recommendation')
{
  const files = [
    { filename: 'm-Q2_K.gguf', bytes: 3 * GB, quant: 'Q2_K', url: '', isMmproj: false, shard: null },
    { filename: 'm-Q4_K_M.gguf', bytes: 6 * GB, quant: 'Q4_K_M', url: '', isMmproj: false, shard: null },
    { filename: 'm-Q8_0.gguf', bytes: 12 * GB, quant: 'Q8_0', url: '', isMmproj: false, shard: null },
    { filename: 'mmproj-f16.gguf', bytes: 0.6 * GB, quant: null, url: '', isMmproj: true, shard: null }
  ]

  const big = recommendQuant(files, hw([gpu('RTX 4090', 24, 23)]), 65536)
  check('picks the highest quality that fits on a big card', big?.filename === 'm-Q8_0.gguf', big?.filename)
  check('says it fits', big?.fitsFullyOnGpu === true)

  const small = recommendQuant(files, hw([gpu('RTX 3060', 12, 11)]), 65536)
  check('steps down on a smaller card', small?.filename !== 'm-Q8_0.gguf', small?.filename)

  const tiny = recommendQuant(files, hw([gpu('GTX 1050', 4, 3.5)]), 65536)
  check('admits when nothing fits', tiny?.fitsFullyOnGpu === false)
  check('explains the recommendation', !!tiny?.reason && tiny.reason.length > 20)

  check('never recommends an mmproj as the model', ![big, small, tiny].some((r) => r?.filename.includes('mmproj')))
  check('finds the mmproj companion', findMmprojFor(files)?.filename === 'mmproj-f16.gguf')
}

section('Byte formatting')
{
  // A missing kilobyte step printed "438009 B" where "428 KB" was meant.
  check('bytes below a kilobyte', fmtBytes(512) === '512 B', fmtBytes(512))
  check('kilobytes', fmtBytes(438009) === '428 KB', fmtBytes(438009))
  check('megabytes', fmtBytes(17_000_000) === '16 MB', fmtBytes(17_000_000))
  check('gigabytes keep two decimals below ten', fmtBytes(4 * GB) === '4.00 GB', fmtBytes(4 * GB))
  check('gigabytes drop a decimal above ten', fmtBytes(18_973_870_432) === '17.7 GB', fmtBytes(18_973_870_432))
  check('zero is not a raw number with no unit', fmtBytes(0) === '0 B', fmtBytes(0))
  check('non-finite input is handled', fmtBytes(Number.NaN) === '?', fmtBytes(Number.NaN))
}


section('Reasoning detection')
{
  // Shaped after the templates actually shipped by these models. The local-file check in
  // scripts/reasoning-check.mjs covers the real thing; these pin the awkward cases.

  // Qwen3.8: validates against a tuple *after* remapping 'high' onto 'xhigh'.
  const qwen38 = `
    {%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort == 'high' %}
    {%- set resolved_reasoning_effort = 'xhigh' %}
    {%- endif %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
    {{- raise_exception('Unexpected reasoning effort') }}
    {%- endif %}
  `
  const a = detectReasoning(qwen38)
  check('effort model is detected', a.kind === 'effort', a.kind)
  check('levels come from the validation tuple', a.levels.join(',') === 'low,medium,xhigh', a.levels.join(','))
  // 'high' is an alias for 'xhigh' here; offering it would be a stop that changes nothing.
  check('an aliased level is not offered', !a.levels.includes('high'), a.levels.join(','))
  check('the template default is used', a.defaultLevel === 'xhigh', String(a.defaultLevel))
  check('enable_thinking implies it can be switched off', a.canDisable === true)

  // gpt-oss: no validation tuple, only equality branches, and different level names.
  const gptoss = `
    {%- if reasoning_effort == 'low' %}Reasoning: low
    {%- elif reasoning_effort == 'medium' %}Reasoning: medium
    {%- elif reasoning_effort == 'high' %}Reasoning: high
    {%- endif %}
  `
  const b = detectReasoning(gptoss)
  check('equality branches are used when there is no tuple', b.kind === 'effort', b.kind)
  check('gpt-oss levels are read, not assumed', b.levels.join(',') === 'low,medium,high', b.levels.join(','))
  check('without a template default, the strongest level is assumed', b.defaultLevel === 'high', String(b.defaultLevel))
  check('no enable_thinking means it cannot be switched off', b.canDisable === false)

  // Qwen3.6 / nanbeige: a toggle and nothing more.
  const toggle = `{%- if enable_thinking is defined and enable_thinking is false %}<think></think>{%- endif %}`
  const c = detectReasoning(toggle)
  check('a toggle-only template is detected', c.kind === 'toggle', c.kind)
  check('a toggle exposes no levels', c.levels.length === 0)

  // Hermes-3 and friends.
  check('a plain template reports none', detectReasoning('{{ messages }}').kind === 'none')
  check('an absent template reports none', detectReasoning(undefined).kind === 'none')
  check('an empty template reports none', detectReasoning('').kind === 'none')

  // 'none' is llama.cpp's off switch, not a level.
  const withNone = `{%- if reasoning_effort in ('none', 'low', 'high') %}{%- endif %}`
  const d = detectReasoning(withNone)
  check("'none' is treated as off, not a level", d.levels.join(',') === 'low,high', d.levels.join(','))
  check("'none' implies it can be switched off", d.canDisable === true)

  // A derived variable with an unrelated name: templates normalise reasoning_effort before
  // branching on it, and the name they pick is arbitrary. Matching only names that happen to
  // contain 'reasoning_effort' worked for Qwen3.8 by luck and lost the control entirely for a
  // template that used a short name.
  const aliased = `
    {%- set r = reasoning_effort|default('medium') %}
    {%- if r not in ('low', 'medium', 'high') %}{{- raise_exception('nope') }}{%- endif %}
  `
  const f = detectReasoning(aliased)
  check('a renamed effort variable is still detected', f.kind === 'effort', f.kind)
  check('its levels are read from the tuple', f.levels.join(',') === 'low,medium,high', f.levels.join(','))
  check('its default is read too', f.defaultLevel === 'medium', String(f.defaultLevel))

  // A single level is not a choice worth a slider.
  const one = `{%- if reasoning_effort == 'high' %}{%- endif %}`
  check('one level alone is not an effort control', detectReasoning(one).kind === 'none', detectReasoning(one).kind)

  // Unknown names must still order deterministically rather than throwing them away.
  const exotic = `{%- if reasoning_effort in ('turbo', 'low', 'max') %}{%- endif %}`
  const e = detectReasoning(exotic)
  check('unknown level names are kept', e.levels.includes('turbo'), e.levels.join(','))
  check('known names still order correctly around them', e.levels.indexOf('low') < e.levels.indexOf('max'),
    e.levels.join(','))

  // The off value is only recorded when the template's own accepted set contains one.
  check('an off level is recorded when the template lists one', d.offValue === 'none', String(d.offValue))
  check('no off level is invented for a levels-only template', a.offValue === null, String(a.offValue))
  check('nor for one with only equality branches', b.offValue === null, String(b.offValue))
}

/*
 * Turning thinking off.
 *
 * Most effort templates enumerate levels and provide no way to say "none" — Qwen3.8 validates
 * against ('xhigh','medium','low') and raises on anything else. Off therefore cannot be built out
 * of the effort level alone; it rests on llama.cpp's own reasoning budget, which ends the thought
 * block without asking the template's permission. What must never happen is sending a level name
 * the template will reject, which turns "stop thinking" into a failed request.
 */
section('Reasoning: off')
{
  const levelsOnly = detectReasoning(
    `{%- set r = reasoning_effort|default('xhigh') %}
     {%- if r not in ('xhigh', 'medium', 'low') %}{{- raise_exception('nope') }}{%- endif %}`
  )
  const withOff = detectReasoning(`{%- if reasoning_effort in ('none', 'low', 'high') %}{%- endif %}`)
  const toggleOnly = detectReasoning(
    `{%- if enable_thinking is defined and enable_thinking is false %}<think></think>{%- endif %}`
  )

  check('a levels-only template cannot disable via its own template', levelsOnly.canDisable === false)

  const offLevels = reasoningRequestFields(levelsOnly, 'off')
  check('off still works on a levels-only model', offLevels.reasoningBudget === 0, JSON.stringify(offLevels))
  check(
    'off never sends a level the template would raise on',
    offLevels.reasoningEffort === undefined,
    String(offLevels.reasoningEffort)
  )
  check('off asks the template too, in case it listens', offLevels.chatTemplateKwargs?.enable_thinking === false)

  const offNamed = reasoningRequestFields(withOff, 'off')
  check('off sends the level when the template accepts one', offNamed.reasoningEffort === 'none', String(offNamed.reasoningEffort))
  check('and still sets the budget', offNamed.reasoningBudget === 0)

  const offToggle = reasoningRequestFields(toggleOnly, 'off')
  check('off on a toggle model clears enable_thinking', offToggle.chatTemplateKwargs?.enable_thinking === false)
  check('off on a toggle model sends no level', offToggle.reasoningEffort === undefined, String(offToggle.reasoningEffort))

  // Everything that is not "off" must be untouched by the above.
  const on = reasoningRequestFields(levelsOnly, 'medium')
  check('a chosen level is sent as-is', on.reasoningEffort === 'medium', String(on.reasoningEffort))
  check('a chosen level sets no budget', on.reasoningBudget === undefined, String(on.reasoningBudget))
  check(
    'a level the model never advertised is dropped',
    reasoningRequestFields(levelsOnly, 'ludicrous').reasoningEffort === undefined
  )
  check('null means leave the template alone', Object.keys(reasoningRequestFields(levelsOnly, null)).length === 0)
  check('a non-reasoning model sends nothing at all',
    Object.keys(reasoningRequestFields(detectReasoning('{{ messages }}'), 'off')).length === 0)
}

section('Export filenames')
{
  const F = (t) => exportFilename(t, 'fallback')

  check('ordinary titles pass through', F('State space models') === 'State space models', F('State space models'))
  check('forbidden characters are replaced', F('a/b:c*d?e') === 'a-b-c-d-e', F('a/b:c*d?e'))
  // Windows drops a trailing dot or space, so the file would not match the name we returned.
  check('trailing dot is removed', F('Notes.') === 'Notes', F('Notes.'))
  check('trailing space is removed', F('Notes   ') === 'Notes', F('Notes   '))
  // A title with nothing usable in it must not produce a bare ".md".
  check('punctuation-only title falls back', F('???') === 'fallback', F('???'))
  check('empty title falls back', F('') === 'fallback', F(''))
  check('dots-only title falls back', F('...') === 'fallback', F('...'))
  check('whitespace-only title falls back', F('   ') === 'fallback', F('   '))
  // Reserved device names are rejected by the filesystem no matter the extension.
  check('reserved name is escaped', F('CON') === 'CON-file', F('CON'))
  check('reserved name is escaped case-insensitively', F('nul') === 'nul-file', F('nul'))
  check('a name merely containing a reserved word is left alone', F('console') === 'console', F('console'))
  // Unicode is legal in Windows filenames; mangling it would be wrong.
  check('emoji survive', F('rocket ' + String.fromCodePoint(0x1f680)) === 'rocket ' + String.fromCodePoint(0x1f680), F('rocket'))
  check('CJK survives', F('\u4e2d\u6587') === '\u4e2d\u6587', F('\u4e2d\u6587'))
  // Control characters would be written verbatim into a path otherwise.
  check('control characters are stripped', F('a' + String.fromCharCode(7) + 'b') === 'a-b', F('a' + String.fromCharCode(7) + 'b'))
  check('long titles are truncated', F('x'.repeat(200)).length <= 60, String(F('x'.repeat(200)).length))
  check('truncation cannot leave a trailing dot', !F('y'.repeat(59) + '.' + 'z'.repeat(20)).endsWith('.'), F('y'.repeat(59) + '.' + 'z'.repeat(20)))
  check('backslash is treated as forbidden', F('a' + String.fromCharCode(92) + 'b') === 'a-b', F('a' + String.fromCharCode(92) + 'b'))

  // Re-exporting the same conversation must not destroy the earlier export.
  const taken = new Set([path.join('d', 'chat.md'), path.join('d', 'chat (2).md')])
  check('first free name is used', uniquePath('d', 'fresh', 'md', (x) => taken.has(x)) === path.join('d', 'fresh.md'))
  check('collisions get a suffix', uniquePath('d', 'chat', 'md', (x) => taken.has(x)) === path.join('d', 'chat (3).md'),
    uniquePath('d', 'chat', 'md', (x) => taken.has(x)))
}

// ---------------------------------------------------------------- schema migrations

/*
 * Migrations run destructive DDL against a database full of the user's chat history, and they
 * are forward-only — a mistake here is not recoverable on the next launch. Migration 4 rebuilds
 * `tasks` to add the foreign key it was originally declared without, so it is exercised against
 * a database shaped like the one a real upgrade would meet: live rows, and orphans left behind
 * by the missing cascade.
 */
section('Schema migrations')

// storage/paths resolves APPDATA_DIR at import time. Point it at a throwaway directory before
// db.js loads, so the suite can never read or write the real chat history — the stubbed
// Electron would refuse anyway, but only after the fact.
process.env.LLMM_APPDATA_DIR ??= fs.mkdtempSync(path.join(os.tmpdir(), 'llmm-test-'))

{
  const { DatabaseSync } = await import('node:sqlite')
  const { MIGRATIONS } = await import('./built/db.js')

  // Mirrors getDb: each migration is applied once, in id order, and only if it has not been.
  const applyRange = (db, afterId, throughId) => {
    for (const m of MIGRATIONS) {
      if (m.id > afterId && m.id <= throughId) db.exec(m.sql)
    }
  }

  const db = new DatabaseSync(':memory:')
  db.exec('PRAGMA foreign_keys = ON;')
  applyRange(db, 0, 3)

  db.exec("INSERT INTO chats (id, title, kind, created_at, updated_at) VALUES ('live', 'Live', 'agent', 1, 1);")
  db.exec("INSERT INTO tasks (id, chat_id, text, done, ord) VALUES ('t1', 'live', 'live task', 0, 0);")
  db.exec("INSERT INTO tasks (id, chat_id, text, done, ord) VALUES ('t2', 'live', 'finished task', 1, 3);")
  // Rows the missing cascade stranded when their chats were deleted.
  db.exec("INSERT INTO tasks (id, chat_id, text, done, ord) VALUES ('t3', 'gone', 'orphan', 0, 0);")

  check('pre-migration schema keeps orphaned tasks', db.prepare('SELECT COUNT(*) n FROM tasks').get().n === 3)

  applyRange(db, 3, 4)

  const rows = db.prepare('SELECT id, chat_id, text, done, ord FROM tasks ORDER BY id').all()
  check('migration drops orphaned tasks', rows.length === 2 && rows.every((r) => r.chat_id === 'live'),
    JSON.stringify(rows.map((r) => r.id)))
  check('migration preserves done and ord', rows.find((r) => r.id === 't2')?.done === 1 && rows.find((r) => r.id === 't2')?.ord === 3)

  db.exec("DELETE FROM chats WHERE id = 'live';")
  check('deleting a chat now cascades to its tasks', db.prepare('SELECT COUNT(*) n FROM tasks').get().n === 0)

  let rejected = false
  try {
    db.exec("INSERT INTO tasks (id, chat_id, text, done, ord) VALUES ('bad', 'nonexistent', 'x', 0, 0);")
  } catch {
    rejected = true
  }
  check('a task cannot be written against a missing chat', rejected)

  // Migration 5: the turn's plan is stored beside its message rather than pasted into it.
  applyRange(db, 4, 5)
  const cols = db.prepare('PRAGMA table_info(messages)').all().map((c) => c.name)
  check('messages can hold a plan', cols.includes('plan'), cols.join(','))
  check('and still hold reasoning', cols.includes('reasoning'), cols.join(','))

  db.close()
}

{
  // Every migration must be safe to re-run: getDb applies only unapplied ids, but a partially
  // applied upgrade should not wedge the app on the next launch either.
  const { DatabaseSync } = await import('node:sqlite')
  const { MIGRATIONS } = await import('./built/db.js')
  check('migration ids are unique and ordered',
    MIGRATIONS.every((m, i) => i === 0 || m.id > MIGRATIONS[i - 1].id))

  const fresh = new DatabaseSync(':memory:')
  fresh.exec('PRAGMA foreign_keys = ON;')
  let ok = true
  try {
    for (const m of MIGRATIONS) fresh.exec(m.sql)
  } catch {
    ok = false
  }
  check('a fresh database applies every migration cleanly', ok)
  fresh.close()
}

// ---------------------------------------------------------------- ultra

/*
 * Ultra is mostly orchestration, but two pure pieces decide whether it is worth its cost: the
 * spread of temperatures that makes the samples differ, and the synthesis prompt that has to
 * turn several answers into one without averaging a right answer with a wrong one.
 */
section('Ultra')
{
  const { sampleTemperatures, synthesisMessages, looksDegenerate } = await import('./built/ultra.js')

  const three = sampleTemperatures(3)
  check('a sample per attempt', three.length === 3, three.join(','))
  check('temperatures rise across the run', three[0] < three[1] && three[1] < three[2], three.join(','))
  check('the first attempt is the least adventurous', three[0] <= 0.5, String(three[0]))
  check('none of them are incoherent', three.every((t) => t <= 1), three.join(','))
  check('a single sample still gets a usable temperature', sampleTemperatures(1).length === 1)
  // Identical samples would make best-of-N an expensive way to ask the same thing N times.
  check('two samples still differ from each other', new Set(sampleTemperatures(2)).size === 2)
  check('eight samples are all distinct', new Set(sampleTemperatures(8)).size === 8)

  const original = [{ role: 'user', content: 'what is 2+2?' }]
  const samples = [
    { index: 0, answer: 'four', reasoning: 'x'.repeat(9000), continuations: 2, temperature: 0.45, ms: 10 },
    { index: 1, answer: 'five', reasoning: 'y'.repeat(9000), continuations: 1, temperature: 0.95, ms: 12 }
  ]
  const msgs = synthesisMessages(original, samples)

  check('the original turn is preserved', msgs[0].content === 'what is 2+2?')
  check('one instruction is appended', msgs.length === original.length + 1)

  const prompt = msgs.at(-1).content
  check('every candidate is offered', prompt.includes('four') && prompt.includes('five'))
  /*
   * The reasoning must not come along. A forced sample thinks for thousands of tokens, and
   * passing that back for each candidate would exhaust the window before the pass that has to
   * read them could start.
   */
  check('reasoning is left out of the prompt', !prompt.includes('x'.repeat(50)) && !prompt.includes('y'.repeat(50)),
    `${prompt.length} chars`)
  check('it is told to choose rather than blend', /decide which is correct|blend/i.test(prompt))
  check('it is told not to narrate the process', /do not mention/i.test(prompt))

  /*
   * Degeneration detection, against the real thing.
   *
   * Forced to keep thinking about "Idk", the model produced this: sentence after sentence
   * announcing a check, on a question containing no algebra at all. Every line differs, so
   * comparing lines finds nothing — what gives it away is that they all open the same way.
   */
  const vamping = [
    'Let me verify the coset',
    'Let me reconsider the homomorphism',
    'Let me check the kernel',
    'Let me verify the image',
    'Let me reconsider the exact sequence',
    'Let me check the extension',
    'Let me verify the splitting',
    'Let me reconsider the quotient',
    'Let me check the injection'
  ].join('. ')
  check('a forced loop is recognised', looksDegenerate(vamping))

  // Real reasoning must survive the same test, or forcing is disabled by its own safety net.
  const genuine = [
    'The user is asking about the config key rename',
    'There are three call sites in the loader',
    'The second one reads from an env var, so it needs care',
    'I should grep before editing anything',
    'Renaming without checking the tests would break the suite',
    'Let me look at how the default is applied',
    'The migration also references the old name',
    'That means two changes, not one',
    'I will start with the grep'
  ].join('. ')
  check('methodical reasoning is not mistaken for a loop', !looksDegenerate(genuine))

  check('a short passage is never judged', !looksDegenerate('Let me check. Let me verify. Let me see.'))
  check('empty text is not degenerate', !looksDegenerate(''))

  const empty = synthesisMessages(original, [{ index: 0, answer: '   ', reasoning: '', continuations: 0, temperature: 0.6, ms: 1 }])
  check('an empty answer is labelled, not dropped silently',
    empty.at(-1).content.includes('no answer produced'))
}

/*
 * The agent's variant answers a different question: these candidates are courses of action, and
 * the output has to be one coherent sequence rather than a merge. A plan assembled from parts of
 * several is one nobody checked end to end.
 */
section('Ultra: agent plans')
{
  const { planSynthesisMessages, planPreamble } = await import('./built/ultra.js')

  const msgs = planSynthesisMessages('rename the config key', ['1. grep for it, 2. edit', '1. edit blindly'])
  check('the plan prompt is a single turn', msgs.length === 1 && msgs[0].role === 'user')

  const p = msgs[0].content
  check('the task is restated', p.includes('rename the config key'))
  check('every plan is offered', p.includes('grep for it') && p.includes('edit blindly'))
  check('it is told to pick a base rather than interleave', /interleaving|soundest/i.test(p))
  check('it is told to answer with the plan alone', /no preamble/i.test(p))

  const blank = planSynthesisMessages('task', ['', '   '])
  check('an empty plan is labelled', blank[0].content.includes('no plan produced'))

  /*
   * The preamble has to say the plan is provisional. It was written before any of the work was
   * done, so an agent that follows it past a contradicting tool result is following a guess.
   */
  const pre = planPreamble('  1. do the thing  ')
  check('the plan survives into the preamble', pre.includes('1. do the thing'))
  check('the preamble is trimmed', !pre.endsWith(' '))
  check('the run is told the tool result wins', /trust the tool result/i.test(pre))
}

// ---------------------------------------------------------------- markdown

// Lives in its own module: the grammar has enough cases to be worth grouping.
const { runMarkdownTests } = await import('./markdown-tests.mjs')
await runMarkdownTests(check, section)

// ---------------------------------------------------------------- summary

console.log(`\n${'='.repeat(60)}`)
console.log(`${pass} passed, ${fail} failed`)
if (failures.length) {
  console.log('\nFailures:')
  failures.forEach((f) => console.log(`  - ${f}`))
}
console.log('')
process.exit(fail ? 1 : 0)

// ---------------------------------------------------------------- helpers

/** Construct a minimal valid GGUF v3 file in memory. */
function buildTestGguf(weightType = 12) {
  const chunks = []
  const u32 = (n) => {
    const b = Buffer.alloc(4)
    b.writeUInt32LE(n)
    return b
  }
  const u64 = (n) => {
    const b = Buffer.alloc(8)
    b.writeBigUInt64LE(BigInt(n))
    return b
  }
  const str = (s) => {
    const body = Buffer.from(s, 'utf8')
    return Buffer.concat([u64(body.length), body])
  }

  chunks.push(u32(0x46554747)) // "GGUF"
  chunks.push(u32(3)) // version
  chunks.push(u64(2)) // tensor count
  chunks.push(u64(5)) // kv count

  // general.architecture = "llama"  (type 8 = string)
  chunks.push(str('general.architecture'), u32(8), str('llama'))
  // llama.block_count = 2  (type 4 = uint32)
  chunks.push(str('llama.block_count'), u32(4), u32(2))
  // llama.embedding_length = 512
  chunks.push(str('llama.embedding_length'), u32(4), u32(512))
  // llama.attention.head_count = 8  -> head_dim should derive to 64
  chunks.push(str('llama.attention.head_count'), u32(4), u32(8))
  // tokenizer.ggml.tokens: an array of 32000 strings, which must be elided
  chunks.push(str('tokenizer.ggml.tokens'), u32(9), u32(8), u64(32000))
  for (let i = 0; i < 32000; i++) chunks.push(str(`t${i}`))

  // Tensor directory: one block tensor and one output tensor.
  chunks.push(str('blk.0.attn_q.weight'), u32(2), u64(512), u64(512), u32(weightType), u64(0))
  chunks.push(str('output.weight'), u32(2), u64(512), u64(32000), u32(weightType), u64(1024))

  return Buffer.concat(chunks)
}
