/**
 * Shared types crossing the main <-> renderer boundary.
 * Kept dependency-free so both sides can import it.
 */

// ---------------------------------------------------------------- GGUF / models

export type Modality = 'text' | 'vision' | 'audio' | 'video'

export interface GgufTensorInfo {
  name: string
  dims: number[]
  ggmlType: number
  offset: number
  /** computed byte size of the tensor data */
  bytes: number
}

export interface GgufMetadata {
  version: number
  tensorCount: number
  /** Raw metadata key/value pairs, large arrays elided. */
  kv: Record<string, GgufValue>
  tensors: GgufTensorInfo[]
  /** byte offset where tensor data begins */
  dataOffset: number
}

export type GgufValue = string | number | boolean | bigint | GgufValue[] | { elided: true; type: number; count: number }

/** The distilled architecture facts the auto-fit engine actually needs. */
export interface ModelArchInfo {
  architecture: string
  name: string | null
  /** number of transformer blocks */
  blockCount: number
  embeddingLength: number
  headCount: number
  /** GQA: number of KV heads. Falls back to headCount when absent (MHA). */
  headCountKv: number
  /** per-head dimension; derived when not stated explicitly */
  headDim: number
  /** trained context length */
  contextLength: number
  vocabSize: number
  /** dominant quantisation of the weight tensors, e.g. "Q4_K" */
  quant: string
  /** total bytes of tensor data (weights) */
  weightBytes: number
  /** bytes of tensors belonging to repeating blocks, i.e. offloadable per-layer */
  perLayerBytes: number
  /** bytes that always live wherever the output/embedding lives */
  nonLayerBytes: number
  /** Mixture-of-experts expert count, 0 when dense */
  expertCount: number
}

export interface ModelCapabilities {
  vision: boolean
  audio: boolean
  /** model was actually trained on video, as opposed to merely accepting frames */
  nativeVideo: boolean
  /** video is reachable at all (llama.cpp expands video -> frames for any vision model) */
  videoPossible: boolean
  tools: boolean
  mmprojPath: string | null
}

export interface ModelRecord {
  id: string
  repo: string | null
  filename: string
  path: string
  bytes: number
  arch: ModelArchInfo | null
  caps: ModelCapabilities
  addedAt: number
  lastUsedAt: number | null
  favourite: boolean
  tags: string[]
  /** set when the file could not be parsed */
  error?: string
}

// ---------------------------------------------------------------- hardware

export interface GpuDevice {
  index: number
  name: string
  vendor: 'nvidia' | 'amd' | 'intel' | 'unknown'
  /** bytes */
  totalVram: number
  /** bytes; -1 when the platform gives us no way to measure it */
  freeVram: number
  /** percent 0-100, -1 when unknown */
  utilisation: number
  /** true when free VRAM is a real measurement rather than an estimate */
  freeIsMeasured: boolean
}

export interface HardwareSnapshot {
  gpus: GpuDevice[]
  totalRam: number
  freeRam: number
  cpuName: string
  cpuThreads: number
  backend: Backend
  takenAt: number
}

export type Backend = 'cuda' | 'vulkan' | 'cpu'

// ---------------------------------------------------------------- auto-fit

export type KvType = 'f16' | 'q8_0' | 'q4_0'

export interface FitConstraints {
  /** hard ceiling the user asked for; undefined means "as much as possible" */
  requestedContext?: number
  /** never quantise KV below this */
  minKvType: KvType
  preferredKvType: KvType
  targetContext: number
  idealContext: number
  /** bytes of VRAM deliberately left unused per GPU */
  headroomBytes: number
  /** user overrides that must be honoured, not silently changed */
  overrides: Partial<{
    contextLength: number
    gpuLayers: number
    kvType: KvType
    batchSize: number
    flashAttention: boolean
    tensorSplit: number[]
  }>
}

export interface FitPlan {
  /** human-facing label, e.g. "Max context" */
  label: string
  contextLength: number
  kvType: KvType
  /** how many of the model's blocks are offloaded to GPU */
  gpuLayers: number
  totalLayers: number
  /** proportional split across devices, summing to 1 */
  tensorSplit: number[]
  batchSize: number
  flashAttention: boolean
  /** predicted bytes per GPU */
  predictedVramPerGpu: number[]
  predictedHostBytes: number
  kvBytes: number
  weightsOnGpuBytes: number
  overheadBytes: number
  /** true when some layers run on CPU */
  spillsToHost: boolean
  /** rough relative speed score, higher is faster */
  speedScore: number
  /** why this plan looks the way it does, shown to the user */
  rationale: string[]
}

export interface FitResult {
  /** the plan the engine would load by default */
  chosen: FitPlan | null
  /** alternatives shown when the target could not be met */
  alternatives: FitPlan[]
  /** true when we could not meet targetContext and the user must decide */
  needsUserChoice: boolean
  hardware: HardwareSnapshot
  notes: string[]
}

// ---------------------------------------------------------------- agent

export type ToolTier = 'read' | 'write' | 'execute'

export interface ToolDefinition {
  name: string
  description: string
  tier: ToolTier
  /** JSON Schema for the arguments; compiled to GBNF for constrained sampling */
  parameters: Record<string, unknown>
  /** when true, this tool is never offered to remote sessions unless remote tools are enabled */
  sensitive?: boolean
}

export interface ToolCall {
  id: string
  name: string
  args: Record<string, unknown>
}

export interface ToolResult {
  callId: string
  ok: boolean
  /** what goes back into the model's context, already truncated */
  content: string
  /** path to the full output when it was truncated */
  fullOutputPath?: string
  truncated: boolean
  error?: string
  durationMs: number
}

export type PermissionDecision =
  | 'allow-once'
  | 'allow-tool'
  | 'allow-exact'
  | 'deny'

export interface PermissionRequest {
  id: string
  tool: string
  tier: ToolTier
  args: Record<string, unknown>
  /** fully resolved, canonicalised description of what will happen */
  resolved: string
  /** set when the request hit the hard-block list */
  hardBlocked: boolean
  blockReason?: string
}

export interface AgentMessage {
  id: string
  role: 'user' | 'assistant' | 'tool' | 'system'
  content: string
  toolCalls?: ToolCall[]
  toolResult?: ToolResult
  createdAt: number
}

export interface AgentSessionState {
  id: string
  title: string
  cwd: string
  planMode: boolean
  messages: AgentMessage[]
  taskList: { id: string; text: string; done: boolean }[]
  createdAt: number
  updatedAt: number
}

// ---------------------------------------------------------------- settings

export type CompactionStrategy = 'auto-compact' | 'sliding-window'

export interface AppSettings {
  modelsDir: string | null
  hfToken: string | null
  autoFit: {
    minKvType: KvType
    preferredKvType: KvType
    targetContext: number
    idealContext: number
    headroomMb: number
  }
  agent: {
    enabled: boolean
    planMode: boolean
    compaction: CompactionStrategy
    maxToolCallsPerTurn: number
    commandTimeoutMs: number
    hardBlocksDisabled: boolean
    remoteToolsEnabled: boolean
  }
  server: {
    enabled: boolean
    port: number
    apiKey: string | null
    jitLoad: boolean
  }
  remote: {
    enabled: boolean
    mode: 'tunnel' | 'own-domain'
    domain: string | null
  }
  ui: {
    closeAction: 'ask' | 'tray' | 'quit'
  }
}
