#!/usr/bin/env node
/**
 * Fetches the binaries that get bundled into the exe.
 *
 * Nothing here is committed to the repo — these are large third-party artifacts pulled from
 * their upstream release pages. Run once after cloning, and again when you want newer builds:
 *
 *   node scripts/fetch-vendor.mjs            # everything
 *   node scripts/fetch-vendor.mjs llama ffmpeg
 *   node scripts/fetch-vendor.mjs --list
 *
 * Downloads resume: a partial file is completed with a range request rather than restarted,
 * which matters when llama.cpp's CUDA build is most of a gigabyte.
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { pipeline } from 'node:stream/promises'
import { Readable } from 'node:stream'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const VENDOR = path.join(ROOT, 'vendor')
const CACHE = path.join(VENDOR, '.cache')

const GB = 1024 ** 3
const MB = 1024 ** 2

/**
 * Each component resolves its own download URL, because upstreams differ: llama.cpp and
 * cloudflared publish per-release assets, FFmpeg has a stable "latest" link, and Chromium is
 * pinned to whatever build playwright-core expects.
 */
const COMPONENTS = {
  llama: {
    label: 'llama.cpp (CPU + Vulkan + CUDA)',
    approxBytes: 1.2 * GB,
    async run() {
      // llama.cpp publishes every binary build as a GitHub *prerelease*, so /releases/latest
      // skips them all and returns an unrelated stale tag. Walk the release list instead and
      // take the newest build-numbered tag that actually carries Windows assets.
      const release = await latestLlamaBuild()
      console.log(`  llama.cpp build ${release.tag_name}`)

      // The CUDA *runtime* asset is named `cudart-llama-bin-win-cuda-<ver>-x64.zip`, which ends
      // identically to the build asset — so it must be excluded by prefix, or it gets picked up
      // as the build and the real binaries are never downloaded.
      const winAssets = release.assets.filter(
        (a) => /-bin-win-.*-x64\.zip$/i.test(a.name) && !/^cudart-/i.test(a.name)
      )

      const cpu = winAssets.find((a) => /bin-win-cpu-x64\.zip$/i.test(a.name))
      const vulkan = winAssets.find((a) => /bin-win-vulkan-x64\.zip$/i.test(a.name))

      // Several CUDA builds ship per release (12.4, 13.3, ...). Take the highest: newer GPUs
      // need newer toolkits — Blackwell (RTX 50-series, sm_120) is not supported by CUDA 12.4
      // at all, so picking the first match would silently produce a build that cannot run.
      const cudaBuilds = winAssets
        .map((a) => ({ asset: a, version: a.name.match(/bin-win-cuda-([\d.]+)-x64\.zip$/i)?.[1] }))
        .filter((x) => x.version)
        .sort((a, b) => compareVersions(b.version, a.version))
      const cuda = cudaBuilds[0]

      const targets = [
        { asset: cpu, dest: 'llama.cpp/cpu', name: 'CPU' },
        { asset: vulkan, dest: 'llama.cpp/vulkan', name: 'Vulkan' },
        { asset: cuda?.asset, dest: 'llama.cpp/cuda', name: `CUDA ${cuda?.version ?? ''}` }
      ]

      for (const t of targets) {
        if (!t.asset) {
          console.log(`  ! no ${t.name} asset in ${release.tag_name}; skipping`)
          continue
        }
        const zip = await download(t.asset.browser_download_url, path.join(CACHE, t.asset.name), t.asset.size)
        await unzipFlat(zip, path.join(VENDOR, t.dest))
        console.log(`  ${t.name} -> vendor/${t.dest}`)
      }

      // The CUDA build needs matching runtime DLLs, shipped as a separate asset. Version must
      // agree with the build we just took, or llama-server fails to start with a DLL error.
      if (cuda) {
        const runtime = release.assets.find((a) =>
          new RegExp(`cudart-llama-bin-win-cuda-${cuda.version.replace('.', '\\.')}-x64\\.zip$`, 'i').test(a.name)
        )
        if (runtime) {
          const zip = await download(runtime.browser_download_url, path.join(CACHE, runtime.name), runtime.size)
          await unzipFlat(zip, path.join(VENDOR, 'llama.cpp/cuda'))
          console.log(`  CUDA ${cuda.version} runtime DLLs -> vendor/llama.cpp/cuda`)
        } else {
          console.log(`  ! no cudart asset matching CUDA ${cuda.version}; the CUDA build may not start`)
        }
      }
    }
  },

  ffmpeg: {
    label: 'FFmpeg (LGPL build, required for video input)',
    approxBytes: 90 * MB,
    async run() {
      // gyan.dev publishes LGPL "essentials" builds; BtbN is the fallback mirror.
      const candidates = [
        'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
        'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip'
      ]
      let zip = null
      for (const url of candidates) {
        try {
          zip = await download(url, path.join(CACHE, 'ffmpeg.zip'))
          break
        } catch (err) {
          console.log(`  ! ${url} failed (${err.message}); trying next mirror`)
        }
      }
      if (!zip) throw new Error('every FFmpeg mirror failed')

      await unzipFlat(zip, path.join(VENDOR, 'ffmpeg'), (name) => /(^|\/)bin\/[^/]+\.(exe|dll)$/i.test(name))
      console.log('  FFmpeg -> vendor/ffmpeg')
    }
  },

  cloudflared: {
    label: 'cloudflared (tunnel for remote access)',
    approxBytes: 40 * MB,
    async run() {
      const url = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe'
      const dest = path.join(VENDOR, 'cloudflared', 'cloudflared.exe')
      await fsp.mkdir(path.dirname(dest), { recursive: true })
      const tmp = await download(url, path.join(CACHE, 'cloudflared.exe'))
      await fsp.copyFile(tmp, dest)
      console.log('  cloudflared -> vendor/cloudflared')
    }
  },

  python: {
    label: 'Python (embeddable, for the agent code-execution tool)',
    approxBytes: 25 * MB,
    async run() {
      const version = '3.12.7'
      const url = `https://www.python.org/ftp/python/${version}/python-${version}-embed-amd64.zip`
      const zip = await download(url, path.join(CACHE, `python-${version}.zip`))
      await unzipFlat(zip, path.join(VENDOR, 'python'))

      // The embeddable build disables site-packages by default; enabling import of the local
      // directory makes the interpreter behave the way scripts expect.
      const pth = (await fsp.readdir(path.join(VENDOR, 'python'))).find((f) => /^python\d+\._pth$/.test(f))
      if (pth) {
        const file = path.join(VENDOR, 'python', pth)
        const content = await fsp.readFile(file, 'utf8')
        if (!content.includes('import site')) {
          await fsp.writeFile(file, `${content.replace(/^#\s*import site$/m, 'import site')}\n`)
        }
      }
      console.log('  Python -> vendor/python')
    }
  },

  ripgrep: {
    label: 'ripgrep (fast search for the agent)',
    approxBytes: 5 * MB,
    async run() {
      const release = await githubRelease('BurntSushi/ripgrep')
      const asset = release.assets.find((a) => /x86_64-pc-windows-msvc\.zip$/i.test(a.name))
      if (!asset) throw new Error('no Windows ripgrep asset found')
      const zip = await download(asset.browser_download_url, path.join(CACHE, asset.name), asset.size)
      await unzipFlat(zip, path.join(VENDOR, 'rg'), (name) => /rg\.exe$/i.test(name))
      console.log('  ripgrep -> vendor/rg')
    }
  },

  chromium: {
    label: 'Chromium (Playwright build, for browser automation)',
    approxBytes: 150 * MB,
    async run() {
      // Playwright's own build CDN rejects direct requests, but its pinned Chromium is a
      // "Chrome for Testing" build, which Google publishes at a stable public URL. Read the
      // exact version playwright expects so the browser and the driver never drift apart.
      let version = null
      try {
        const registry = JSON.parse(
          await fsp.readFile(path.join(ROOT, 'node_modules/playwright-core/browsers.json'), 'utf8')
        )
        version = registry.browsers.find((b) => b.name === 'chromium')?.browserVersion ?? null
      } catch {
        console.log('  ! could not read playwright browsers.json')
      }

      if (!version) {
        const res = await fetch('https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions.json')
        version = (await res.json()).channels.Stable.version
        console.log(`  falling back to Chrome for Testing stable ${version}`)
      }

      const url = `https://storage.googleapis.com/chrome-for-testing-public/${version}/win64/chrome-win64.zip`
      const zip = await download(url, path.join(CACHE, `chromium-${version}.zip`))
      await unzipFlat(zip, path.join(VENDOR, 'chromium'))
      console.log(`  Chromium ${version} -> vendor/chromium`)
    }
  },

  embedding: {
    label: 'Embedding model (bundled default for RAG)',
    approxBytes: 120 * MB,
    async run() {
      // A small, permissively-licensed general embedding model in GGUF form.
      const url =
        'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q5_K_M.gguf'
      const dest = path.join(VENDOR, 'models', 'embedding.gguf')
      await fsp.mkdir(path.dirname(dest), { recursive: true })
      const tmp = await download(url, path.join(CACHE, 'embedding.gguf'))
      await fsp.copyFile(tmp, dest)
      console.log('  embedding model -> vendor/models/embedding.gguf')
    }
  }
}

// ---------------------------------------------------------------- helpers

/** Numeric version compare, so "13.3" sorts above "12.4" rather than lexically below it. */
function compareVersions(a, b) {
  const pa = String(a).split('.').map(Number)
  const pb = String(b).split('.').map(Number)
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0)
    if (d !== 0) return d
  }
  return 0
}

/**
 * The newest llama.cpp build release carrying Windows x64 binaries.
 * Their binary releases are all flagged as prereleases, which /releases/latest ignores.
 */
async function latestLlamaBuild() {
  const res = await fetch('https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20', {
    headers: {
      Accept: 'application/vnd.github+json',
      'User-Agent': 'llm-manager-fetch-vendor',
      ...(process.env.GITHUB_TOKEN ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {})
    }
  })
  if (res.status === 403) throw new Error('GitHub rate-limited this request. Set GITHUB_TOKEN and retry.')
  if (!res.ok) throw new Error(`GitHub API ${res.status} listing llama.cpp releases`)

  const releases = await res.json()
  const build = releases
    .filter((r) => /^b\d+$/.test(r.tag_name))
    .filter((r) => r.assets.some((a) => /-bin-win-.*-x64\.zip$/i.test(a.name)))
    .sort((a, b) => Number(b.tag_name.slice(1)) - Number(a.tag_name.slice(1)))[0]

  if (!build) throw new Error('No llama.cpp release with Windows x64 binaries found in the last 20 releases')
  return build
}

async function githubRelease(repo) {
  const res = await fetch(`https://api.github.com/repos/${repo}/releases/latest`, {
    headers: {
      Accept: 'application/vnd.github+json',
      'User-Agent': 'llm-manager-fetch-vendor',
      ...(process.env.GITHUB_TOKEN ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {})
    }
  })
  if (res.status === 403) {
    throw new Error('GitHub rate-limited this request. Set GITHUB_TOKEN and retry.')
  }
  if (!res.ok) throw new Error(`GitHub API ${res.status} for ${repo}`)
  return res.json()
}

function human(n) {
  if (!Number.isFinite(n) || n <= 0) return '?'
  return n >= GB ? `${(n / GB).toFixed(2)} GB` : `${(n / MB).toFixed(0)} MB`
}

/** Resumable download with a progress line. */
async function download(url, dest, expectedBytes) {
  await fsp.mkdir(path.dirname(dest), { recursive: true })

  let have = 0
  try {
    have = (await fsp.stat(dest)).size
  } catch {
    have = 0
  }

  if (expectedBytes && have === expectedBytes) {
    console.log(`  cached ${path.basename(dest)} (${human(have)})`)
    return dest
  }

  const headers = { 'User-Agent': 'llm-manager-fetch-vendor' }
  if (have > 0) headers.Range = `bytes=${have}-`

  const res = await fetch(url, { headers, redirect: 'follow' })
  if (res.status === 416) return dest
  if (!res.ok && res.status !== 206) throw new Error(`HTTP ${res.status} for ${url}`)
  if (have > 0 && res.status !== 206) {
    have = 0
    await fsp.rm(dest, { force: true })
  }

  const total = have + Number(res.headers.get('content-length') ?? 0)
  let done = have
  let lastPrint = 0

  const out = fs.createWriteStream(dest, { flags: have > 0 ? 'a' : 'w' })
  const body = Readable.fromWeb(res.body)
  body.on('data', (chunk) => {
    done += chunk.length
    const now = Date.now()
    if (now - lastPrint > 400) {
      lastPrint = now
      const pct = total ? ((done / total) * 100).toFixed(1) : '?'
      process.stdout.write(`\r  ${path.basename(dest)}  ${human(done)} / ${human(total)}  ${pct}%   `)
    }
  })
  await pipeline(body, out)
  process.stdout.write(`\r  ${path.basename(dest)}  ${human(done)}  done              \n`)
  return dest
}

/**
 * Extract a zip, flattening any single top-level wrapper directory so callers get a
 * predictable layout. Uses PowerShell's Expand-Archive, which is always present on Windows.
 */
async function unzipFlat(zipPath, destDir, filter) {
  const staging = path.join(CACHE, `x-${path.basename(zipPath, '.zip')}`)
  await fsp.rm(staging, { recursive: true, force: true })
  await fsp.mkdir(staging, { recursive: true })

  await runCommand('powershell.exe', [
    '-NoProfile',
    '-NonInteractive',
    '-Command',
    `Expand-Archive -LiteralPath '${zipPath.replace(/'/g, "''")}' -DestinationPath '${staging.replace(/'/g, "''")}' -Force`
  ])

  // Collapse a lone wrapper directory (e.g. ffmpeg-7.1-essentials_build/).
  let source = staging
  const entries = await fsp.readdir(source, { withFileTypes: true })
  if (entries.length === 1 && entries[0].isDirectory()) {
    source = path.join(source, entries[0].name)
  }

  await fsp.mkdir(destDir, { recursive: true })
  await copyTree(source, destDir, source, filter)
  await fsp.rm(staging, { recursive: true, force: true })
}

async function copyTree(dir, destDir, root, filter) {
  for (const entry of await fsp.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    const rel = path.relative(root, full).replace(/\\/g, '/')
    if (entry.isDirectory()) {
      await copyTree(full, destDir, root, filter)
    } else if (!filter || filter(rel)) {
      // Filtered copies flatten into destDir; unfiltered ones keep their structure.
      const target = filter ? path.join(destDir, entry.name) : path.join(destDir, rel)
      await fsp.mkdir(path.dirname(target), { recursive: true })
      await fsp.copyFile(full, target)
    }
  }
}

function runCommand(file, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(file, args, { stdio: ['ignore', 'ignore', 'pipe'] })
    let stderr = ''
    child.stderr?.on('data', (d) => {
      stderr += d
    })
    child.on('error', reject)
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(stderr.slice(-400) || `exit ${code}`))))
  })
}

// ---------------------------------------------------------------- main

async function main() {
  const args = process.argv.slice(2)

  if (args.includes('--list')) {
    console.log('\nComponents:\n')
    for (const [key, c] of Object.entries(COMPONENTS)) {
      console.log(`  ${key.padEnd(12)} ${human(c.approxBytes).padStart(8)}  ${c.label}`)
    }
    const total = Object.values(COMPONENTS).reduce((a, c) => a + c.approxBytes, 0)
    console.log(`\n  ${'TOTAL'.padEnd(12)} ${human(total).padStart(8)}\n`)
    return
  }

  const selected = args.filter((a) => !a.startsWith('--'))
  const keys = selected.length ? selected : Object.keys(COMPONENTS)

  const unknown = keys.filter((k) => !COMPONENTS[k])
  if (unknown.length) {
    console.error(`Unknown component(s): ${unknown.join(', ')}`)
    console.error(`Known: ${Object.keys(COMPONENTS).join(', ')}`)
    process.exit(1)
  }

  await fsp.mkdir(CACHE, { recursive: true })
  console.log(`\nFetching into ${VENDOR}\n`)

  const failures = []
  for (const key of keys) {
    const c = COMPONENTS[key]
    console.log(`> ${key}: ${c.label}`)
    try {
      await c.run()
    } catch (err) {
      console.error(`  FAILED: ${err.message}`)
      failures.push({ key, error: err.message })
    }
    console.log('')
  }

  if (failures.length) {
    console.log('Some components failed:')
    for (const f of failures) console.log(`  ${f.key}: ${f.error}`)
    console.log('\nRe-run just those, e.g. `node scripts/fetch-vendor.mjs ' + failures.map((f) => f.key).join(' ') + '`')
    process.exitCode = 1
  } else {
    console.log('All components fetched.')
    console.log(`Cache kept in ${CACHE} — delete it to reclaim space once everything works.`)
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
