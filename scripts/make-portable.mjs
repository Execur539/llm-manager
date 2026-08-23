#!/usr/bin/env node
/**
 * Builds the single-file portable exe with extract-once semantics.
 *
 * The problem this solves: electron-builder's `portable` target re-extracts the entire payload
 * to %TEMP% on *every* launch. With llama.cpp's CUDA build, FFmpeg, Chromium and Python inside,
 * that payload is well over a gigabyte — a multi-second-to-minute wait, every single time.
 *
 * The fix is a 7-Zip SFX archive configured with a fixed InstallPath and OverwriteMode=2
 * ("skip existing files"). The first launch unpacks; later launches find the files already
 * there and skip straight to running. One file to ship, fast startup after the first run.
 *
 * Requires 7-Zip (for `7z.exe` and the `7zSD.sfx` module). Point SEVENZIP_HOME at it if it is
 * not installed in the usual place.
 *
 *   npm run pack:portable
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const RELEASE = path.join(ROOT, 'release')
const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'))

const PRODUCT = pkg.build?.productName ?? 'LLM Manager'
const VERSION = pkg.version

function findSevenZip() {
  const candidates = [
    process.env.SEVENZIP_HOME && path.join(process.env.SEVENZIP_HOME, '7z.exe'),
    'C:\\Program Files\\7-Zip\\7z.exe',
    'C:\\Program Files (x86)\\7-Zip\\7z.exe'
  ].filter(Boolean)
  return candidates.find((c) => fs.existsSync(c)) ?? null
}

function findSfxModule(sevenZipExe) {
  const dir = path.dirname(sevenZipExe)
  // 7zSD.sfx (from the LZMA SDK) is the one that supports InstallPath + OverwriteMode.
  const candidates = [
    process.env.SFX_MODULE,
    path.join(dir, '7zSD.sfx'),
    path.join(dir, '7z.sfx'),
    path.join(ROOT, 'build', '7zSD.sfx')
  ].filter(Boolean)
  return candidates.find((c) => fs.existsSync(c)) ?? null
}

function run(file, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(file, args, { stdio: 'inherit', ...opts })
    child.on('error', reject)
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`${path.basename(file)} exited ${code}`))))
  })
}

async function main() {
  const unpacked = path.join(RELEASE, 'win-unpacked')
  if (!fs.existsSync(unpacked)) {
    console.error(`No build found at ${unpacked}.\nRun \`npm run pack:dir\` first.`)
    process.exit(1)
  }

  const sevenZip = findSevenZip()
  if (!sevenZip) {
    console.error(
      'Could not find 7z.exe.\n' +
        'Install 7-Zip from https://7-zip.org, or set SEVENZIP_HOME to its folder.\n\n' +
        'Without it, `npm run pack:installer` still produces a working NSIS installer — the\n' +
        'portable single-file build is the only thing that needs 7-Zip.'
    )
    process.exit(1)
  }

  const sfx = findSfxModule(sevenZip)
  if (!sfx) {
    console.error(
      'Could not find an SFX module (7zSD.sfx).\n' +
        'It ships with the LZMA SDK, not the 7-Zip installer: download the SDK from\n' +
        'https://7-zip.org/sdk.html and put 7zSD.sfx in build/, or set SFX_MODULE.\n\n' +
        'This module is what supports extract-once behaviour; the plain 7z.sfx re-extracts every run.'
    )
    process.exit(1)
  }

  // Sanity check: the payload should contain the vendor binaries, or the portable exe will
  // ship without a runtime and every feature will report "not bundled".
  const vendor = path.join(unpacked, 'resources', 'vendor')
  if (!fs.existsSync(vendor)) {
    console.warn('\n! WARNING: resources/vendor is missing from the build.')
    console.warn('  The portable exe will contain no llama.cpp, ffmpeg, Chromium or Python.')
    console.warn('  Run `npm run fetch-vendor` before packaging.\n')
  } else {
    const size = await dirSize(vendor)
    console.log(`  vendor payload: ${(size / 1024 ** 3).toFixed(2)} GB`)
  }

  const archive = path.join(RELEASE, `${PRODUCT.replace(/\s+/g, '-')}-${VERSION}.7z`)
  await fsp.rm(archive, { force: true })

  console.log('\nCompressing payload (this takes a while at maximum compression)…')
  await run(sevenZip, ['a', '-t7z', '-mx=7', '-mmt=on', archive, `${unpacked}${path.sep}*`])

  // The SFX config. InstallPath makes extraction land in a stable per-user cache directory
  // rather than a fresh temp folder; OverwriteMode=2 skips files that are already there, which
  // is what turns "extract every launch" into "extract once".
  const installPath = `%LOCALAPPDATA%\\\\${PRODUCT.replace(/\s+/g, '')}\\\\runtime-${VERSION}`
  const config = [
    ';!@Install@!UTF-8!',
    `Title="${PRODUCT} ${VERSION}"`,
    `BeginPrompt=""`,
    `InstallPath="${installPath}"`,
    'OverwriteMode="2"',
    `RunProgram="%%T\\\\${PRODUCT}.exe"`,
    'GUIMode="2"',
    'Progress="yes"',
    ';!@InstallEnd@!',
    ''
  ].join('\r\n')

  const configFile = path.join(os.tmpdir(), `llmm-sfx-${Date.now()}.txt`)
  await fsp.writeFile(configFile, config, 'utf8')

  const output = path.join(RELEASE, `${PRODUCT.replace(/\s+/g, '-')}-${VERSION}-portable.exe`)
  console.log('Building self-extracting exe…')

  // A 7-Zip SFX is simply: module + config + archive, concatenated.
  await fsp.writeFile(
    output,
    Buffer.concat([
      await fsp.readFile(sfx),
      Buffer.from(config, 'utf8'),
      await fsp.readFile(archive)
    ])
  )

  await fsp.rm(configFile, { force: true })
  await fsp.rm(archive, { force: true })

  const finalSize = (await fsp.stat(output)).size
  console.log(`\nDone: ${output}`)
  console.log(`  size: ${(finalSize / 1024 ** 3).toFixed(2)} GB`)
  console.log(`  first run unpacks to ${installPath.replace(/\\\\/g, '\\')}`)
  console.log('  later runs skip extraction and start immediately')
  console.log('\nNote: the exe is unsigned, so SmartScreen will warn on first run.')
}

async function dirSize(dir) {
  let total = 0
  for (const entry of await fsp.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) total += await dirSize(full)
    else total += (await fsp.stat(full)).size
  }
  return total
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
