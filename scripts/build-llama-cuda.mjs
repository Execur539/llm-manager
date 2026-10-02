/**
 * Rebuild the CUDA llama.cpp that LLM Manager ships, with Qwen3.8-Flash-Next's QSA work applied.
 *
 * fetch-vendor installs the official b10900 release. Two upstream pull requests make QSA cheaper at
 * long context and are not merged yet, so this builds that same release with them applied and swaps
 * in the only two files they change: llama.dll (pooled indexer keys, #28699) and ggml-cuda.dll
 * (sparse flash attention, #28770). Everything else, including the clang-built CPU backends and
 * llama-server itself, stays the official release.
 *
 * The build also compiles flash-attention kernels for the mixed key/value pairs the planner can
 * choose. The official build has none, and converts such a cache to f16 on every token instead.
 *
 * Needs git, CMake, Visual Studio 2022 with C++, and a CUDA 13 toolkit. It downloads the llama.cpp
 * source at the tag, the two pull requests, and CCCL 3.2 during configure.
 *
 *   node scripts/build-llama-cuda.mjs [--src D:\CODE\llama.cpp-qsa]
 *
 * Re-run it after fetch-vendor, which puts the official files back.
 */

import { execFileSync, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const VENDOR_CUDA = path.join(ROOT, 'vendor', 'llama.cpp', 'cuda')

const BASE = { tag: 'b10900', build: 10900, commit: '50182a53f' }
const PULLS = [
  { number: 28699, commit: '141f3f564', title: 'qwen4exp: incremental pooled-key cache for the QSA indexer' },
  { number: 28770, commit: '41a4ad00d', title: 'CUDA: enable sparse fa for qwen4' }
]
const LOCAL_PATCHES = ['qsa-pooled-reserve.patch']
// The matched pairs every CUDA build compiles, plus the mixed pairs the planner can choose.
const FA_QUANTS = ['q4_0-q4_0', 'q8_0-q8_0', 'f16-f16', 'bf16-bf16', 'q8_0-q4_0', 'f16-q8_0', 'f16-q4_0']
const OUTPUTS = ['llama.dll', 'ggml-cuda.dll']

const argIndex = process.argv.indexOf('--src')
const SRC = path.resolve(argIndex > 0 ? process.argv[argIndex + 1] : path.join(ROOT, '..', 'llama.cpp-qsa'))

const git = (...args) => execFileSync('git', args, { cwd: SRC, stdio: 'inherit' })
const gitOut = (...args) => execFileSync('git', args, { cwd: SRC, encoding: 'utf8' }).trim()

function cudaToolkit() {
  if (process.env.LLMM_CUDA_PATH) return process.env.LLMM_CUDA_PATH
  const root = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA'
  const found = fs.existsSync(root)
    ? fs.readdirSync(root).filter((d) => /^v13\.\d+$/.test(d)).sort((a, b) => Number(b.slice(4)) - Number(a.slice(4)))
    : []
  if (!found.length) throw new Error('No CUDA 13 toolkit found; set LLMM_CUDA_PATH')
  return path.join(root, found[0])
}

function visualStudio() {
  const vswhere = path.join(process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)', 'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
  const install = execFileSync(vswhere, ['-latest', '-products', '*', '-property', 'installationPath'], { encoding: 'utf8' }).trim()
  if (!install) throw new Error('Visual Studio with C++ was not found')
  return install
}

// 1. Two DLLs from one build dropped into another would mix ABIs, so the vendor tree must be the base.
const version = spawnSync(path.join(VENDOR_CUDA, 'llama-server.exe'), ['--version'], { encoding: 'utf8' })
const banner = `${version.stdout ?? ''}${version.stderr ?? ''}`
// llama-server prints "version: 0.4.0-dev (build 10900, commit 50182a53f)"
const match = /\(build (\d+), commit ([0-9a-f]+)\)/.exec(banner)
if (!match || Number(match[1]) !== BASE.build || !BASE.commit.startsWith(match[2].slice(0, 7))) {
  throw new Error(`vendor/llama.cpp/cuda is not the official ${BASE.tag} build (${match ? `${match[1]} ${match[2]}` : 'no version'}); run LLAMA_BUILD=${BASE.tag} node scripts/fetch-vendor.mjs llama first`)
}

// 2. Source: the tag, then each pull request as reviewed, then local fixes.
fs.mkdirSync(SRC, { recursive: true })
if (!fs.existsSync(path.join(SRC, '.git'))) {
  git('init', '-q')
  git('remote', 'add', 'origin', 'https://github.com/ggml-org/llama.cpp.git')
}
git('config', 'core.autocrlf', 'false')
git('fetch', '--depth', '1', '--no-tags', 'origin', `refs/tags/${BASE.tag}:refs/tags/${BASE.tag}`)
for (const pr of PULLS) git('fetch', '--depth', '2', '--no-tags', '--force', 'origin', `pull/${pr.number}/head:pr-${pr.number}`)
git('checkout', '-q', '-f', '-B', 'llmm-cuda', BASE.tag)
for (const pr of PULLS) {
  const head = gitOut('rev-parse', `pr-${pr.number}`)
  if (!head.startsWith(pr.commit)) {
    throw new Error(`#${pr.number} is now at ${head.slice(0, 9)}, not the reviewed ${pr.commit}; review the change before updating PULLS`)
  }
  git('cherry-pick', '-n', `pr-${pr.number}`)
}
for (const patch of LOCAL_PATCHES) git('apply', path.join(ROOT, 'scripts', 'llama-patches', patch))

// 3. Configure and build like the official windows-cuda job: MSVC host compiler, backend DLLs.
const cuda = cudaToolkit()
const vs = visualStudio()
const ninja = path.join(vs, 'Common7', 'IDE', 'CommonExtensions', 'Microsoft', 'CMake', 'Ninja')
const script = path.join(os.tmpdir(), `llmm-build-llama-cuda-${process.pid}.cmd`)
fs.writeFileSync(
  script,
  [
    '@echo off',
    `call "${path.join(vs, 'VC', 'Auxiliary', 'Build', 'vcvarsall.bat')}" x64 >nul || exit /b 1`,
    `set "CUDA_PATH=${cuda}"`,
    `set "PATH=${ninja};%CUDA_PATH%\\bin;%PATH%"`,
    `cd /d "${SRC}"`,
    [
      'cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release',
      '-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_CUDA_CUB_3DOT2=ON',
      `"-DGGML_CUDA_FA_QUANTS=${FA_QUANTS.join(';')}"`,
      '"-DCMAKE_CUDA_COMPILER=%CUDA_PATH%\\bin\\nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_PATH%"',
      '-DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF',
      '|| exit /b 1'
    ].join(' '),
    `cmake --build build --target ${OUTPUTS.map((f) => path.basename(f, '.dll')).join(' ')} || exit /b 1`
  ].join('\r\n')
)
try {
  execFileSync('cmd.exe', ['/c', script], { stdio: 'inherit' })
} finally {
  fs.rmSync(script, { force: true })
}

// 4. Swap the two files in, and record what they are beside them.
for (const file of OUTPUTS) fs.copyFileSync(path.join(SRC, 'build', 'bin', file), path.join(VENDOR_CUDA, file))
fs.writeFileSync(
  path.join(VENDOR_CUDA, 'PATCHES.txt'),
  [
    `llama.dll and ggml-cuda.dll in this folder are ${BASE.tag} (${BASE.commit}) rebuilt with:`,
    ...PULLS.map((pr) => `  ggml-org/llama.cpp#${pr.number} ${pr.commit} ${pr.title}`),
    ...LOCAL_PATCHES.map((p) => `  scripts/llama-patches/${p}`),
    `  GGML_CUDA_FA_QUANTS=${FA_QUANTS.join(';')}`,
    `  CUDA ${path.basename(cuda)}, CCCL 3.2 (GGML_CUDA_CUB_3DOT2)`,
    'Every other file is the official release. Rebuild with: node scripts/build-llama-cuda.mjs',
    ''
  ].join('\n')
)
console.log(`\nswapped ${OUTPUTS.join(' and ')} into vendor/llama.cpp/cuda`)
