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
    gpu: path.join(ROOT, 'src/main/hardware/gpu.ts'),
    markdown: path.join(ROOT, 'src/renderer/src/lib/markdown.ts')
  },
  outdir: path.join(ROOT, 'scripts/built'),
  bundle: true,
  platform: 'node',
  format: 'esm',
  external: ['electron'],
  logLevel: 'error',
  alias: { '@shared': path.join(ROOT, 'src/shared') }
})

// RAG chunking is worth testing but the module imports Electron-bound code; extract the
// pure pieces by bundling with the electron import stubbed out.
console.log('test bundles built')
