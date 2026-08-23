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
const { readGguf, extractArchInfo, tensorByteSize } = await import('./built/gguf.js')
const { recommendQuant, findMmprojFor } = await import('./built/hf.js')
 const { isVirtualAdapter } = await import('./built/gpu.js')

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
function buildTestGguf() {
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
  chunks.push(str('blk.0.attn_q.weight'), u32(2), u64(512), u64(512), u32(12), u64(0))
  chunks.push(str('output.weight'), u32(2), u64(512), u64(32000), u32(12), u64(1024))

  return Buffer.concat(chunks)
}
