// Bundles the pure modules under test so they can run outside Electron.
import { build } from 'esbuild'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

await build({
  entryPoints: {
    engine: path.join(ROOT, 'src/main/autofit/engine.ts'),
    gbnf: path.join(ROOT, 'src/main/agent/gbnf.ts'),
    permissions: path.join(ROOT, 'src/main/agent/permissions.ts'),
    hf: path.join(ROOT, 'src/main/downloads/hf.ts'),
    gguf: path.join(ROOT, 'src/main/models/gguf.ts'),
    reasoning: path.join(ROOT, 'src/main/models/reasoning.ts'),
    gpu: path.join(ROOT, 'src/main/hardware/gpu.ts'),
    filenames: path.join(ROOT, 'src/main/storage/filenames.ts'),
    db: path.join(ROOT, 'src/main/storage/db.ts'),
    settings: path.join(ROOT, 'src/main/storage/settings.ts'),
    repo: path.join(ROOT, 'src/main/chat/repo.ts'),
    workflow: path.join(ROOT, 'src/main/agent/tools/workflow.ts'),
    queue: path.join(ROOT, 'src/main/downloads/queue.ts'),
    ultra: path.join(ROOT, 'src/main/ultra/index.ts'),
    markdown: path.join(ROOT, 'src/renderer/src/lib/markdown.ts')
  },
  outdir: path.join(ROOT, 'scripts/built'),
  bundle: true,
  platform: 'node',
  format: 'esm',
  external: ['node:sqlite'],
  logLevel: 'error',
  alias: {
    '@shared': path.join(ROOT, 'src/shared'),
    // db.ts pulls in Electron only by way of storage/paths.ts, which resolves everything from
    // LLMM_APPDATA_DIR under test. Stubbing the import is what lets the schema and its
    // migrations be exercised against a throwaway database in plain Node.
    electron: path.join(ROOT, 'scripts/electron-stub.mjs')
  }
})

// RAG chunking is worth testing but the module imports Electron-bound code; extract the
// pure pieces by bundling with the electron import stubbed out.
console.log('test bundles built')
