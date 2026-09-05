/**
 * Checks the arithmetic and the ffmpeg contracts behind video uploads.
 *
 * None of this was covered, and it is the part of the codebase that has quietly broken twice.
 * Once when ffmpeg 9 removed `-vsync` and the sparse extraction path silently produced no frames
 * at all; once when the condensed clip was muxed at half the rate llama.cpp decodes at, so every
 * frame was handed to the vision encoder twice and charged twice — for months, invisibly, because
 * the cost constant had been fitted to the duplication and the estimate therefore looked right.
 *
 * Both were failures of a contract with an external binary rather than of logic, so the cases
 * that matter run the real ffmpeg and measure what comes out. The planner is bundled from source
 * rather than mirrored, because a mirror that drifts passes while testing nothing.
 *
 *   node scripts/video-plan-check.mjs
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import esbuild from 'esbuild'

const run = promisify(execFile)
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const FFMPEG = path.join(ROOT, 'vendor', 'ffmpeg', 'ffmpeg.exe')
const FFPROBE = path.join(ROOT, 'vendor', 'ffmpeg', 'ffprobe.exe')

let pass = 0
let fail = 0
function check(name, ok, detail = '') {
  if (ok) {
    pass++
    console.log(`  ok   ${name}`)
  } else {
    fail++
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

const work = await fsp.mkdtemp(path.join(os.tmpdir(), 'video-plan-'))
const ff = (args) => run(FFMPEG, args, { maxBuffer: 64 * 1024 * 1024 }).catch((e) => e)

// ---------------------------------------------------------------- the planner

const bundle = await esbuild.build({
  entryPoints: [path.join(ROOT, 'src', 'main', 'chat', 'video-plan.ts')],
  bundle: true,
  format: 'esm',
  write: false,
  platform: 'node'
})
const tmpMod = path.join(work, 'video-plan.mjs')
await fsp.writeFile(tmpMod, bundle.outputFiles[0].text, 'utf8')
const { planVideo, chooseTimestamps } = await import(`file://${tmpMod.replace(/\\/g, '/')}`)

const base = {
  contextLength: 96_000,
  share: 0.45,
  detail: 'balanced',
  durationSeconds: 120,
  aspect: 16 / 9,
  temporalPairing: true
}

console.log('\nplanner')

{
  const p = planVideo(base)
  check('stays inside its budget', p.estimatedTokens <= base.contextLength * base.share,
    `${p.estimatedTokens} > ${base.contextLength * base.share}`)
  /*
   * The 3D convolution consumes frames in pairs, so an odd one at the end is padded: charged
   * for, contributing nothing.
   */
  check('plans an even number of frames', p.count % 2 === 0, `count ${p.count}`)
  check('respects the frame rate ceiling', p.effectiveFps <= 2.001, `${p.effectiveFps.toFixed(3)} fps`)
  check('frame width sits on the 28px patch grid', p.width % 28 === 0, `${p.width}px`)
}

{
  /*
   * Budgeting by width rather than area overcharges anything that is not 16:9 — a vertical
   * phone video at a fixed width is nearly three times the area of the landscape frame that
   * width implies, for no more information.
   */
  const landscape = planVideo({ ...base, aspect: 16 / 9 })
  const portrait = planVideo({ ...base, aspect: 9 / 16 })
  check('a vertical video is not charged more than a horizontal one',
    portrait.estimatedTokens <= landscape.estimatedTokens * 1.1,
    `portrait ${portrait.estimatedTokens} vs landscape ${landscape.estimatedTokens}`)
  check('a vertical video is narrower, not wider', portrait.width < landscape.width,
    `${portrait.width} vs ${landscape.width}`)
}

{
  /*
   * The pairing discount is the whole reason the video path beats sending loose images. If this
   * ever inverts, something has gone wrong with the cost factor.
   */
  const asVideo = planVideo({ ...base, temporalPairing: true })
  const asImages = planVideo({ ...base, temporalPairing: false })
  check('sending as video buys more frames than sending as images',
    asVideo.count > asImages.count, `${asVideo.count} vs ${asImages.count}`)
}

{
  const big = planVideo({ ...base, contextLength: 262_144 })
  const small = planVideo({ ...base, contextLength: 32_768 })
  check('a larger context buys more frames', big.count > small.count, `${big.count} vs ${small.count}`)
  check('a tiny context still plans at least one frame',
    planVideo({ ...base, contextLength: 2048 }).count >= 1)
}

{
  /*
   * A source smaller than the budget must cost less than the budget, because the extraction
   * clamps to the source width and never upscales. Believing otherwise makes the planner
   * under-spend on precisely the clips that were cropped hardest.
   */
  const full = planVideo(base)
  const small = planVideo({ ...base, sourcePixels: 240 * 135 })
  check('a small source frame costs less per frame than a large one',
    small.estimatedTokens / small.count < full.estimatedTokens / full.count,
    `${(small.estimatedTokens / small.count).toFixed(1)} vs ${(full.estimatedTokens / full.count).toFixed(1)}`)
  check('a small source frame is not upscaled past what it has',
    small.width <= 240, `${small.width}px from a 240px source`)
  check('a source larger than the budget is still capped by the budget',
    planVideo({ ...base, sourcePixels: 3840 * 2160 }).width === full.width)
  check('cheaper frames buy more of them, up to the frame rate ceiling',
    small.count >= full.count && small.effectiveFps <= 2.001,
    `${small.count} vs ${full.count} at ${small.effectiveFps.toFixed(2)} fps`)
}

{
  const picked = chooseTimestamps([12, 40, 78], 120, 10)
  check('scene selection returns the count asked for', picked.length === 10, `got ${picked.length}`)
  check('scene selection starts at the beginning', picked[0] === 0, `starts at ${picked[0]}`)
  check('scene selection is ordered and inside the clip',
    picked.every((t, i) => (i === 0 || t > picked[i - 1]) && t >= 0 && t < 120))
  check('every detected cut is kept', [12, 40, 78].every((t) => picked.includes(t)))
}

// ------------------------------------------------------------ ffmpeg contracts

console.log('\nffmpeg contracts')

if (!fs.existsSync(FFMPEG)) {
  console.log('  skipped — vendored ffmpeg not present')
} else {
  const version = (await ff(['-version'])).stdout?.split('\n')[0] ?? ''
  check('vendored ffmpeg is present', version.includes('ffmpeg version'), version.slice(0, 60))

  /*
   * `-vsync 0` was removed in ffmpeg 9 and an unrecognised option is not a warning: the command
   * fails and yields nothing. This is the case that would have caught that.
   */
  const passthrough = await ff([
    '-hide_banner', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=10:duration=1',
    '-vf', "select='lt(abs(t-0.5)\\,0.06)'", '-fps_mode', 'passthrough',
    '-q:v', '3', path.join(work, 'sel-%03d.jpg')
  ])
  const selected = (await fsp.readdir(work)).filter((f) => f.startsWith('sel-'))
  check('the sparse extraction options are accepted by this ffmpeg', selected.length >= 1,
    `produced ${selected.length} frames; ${(passthrough.stderr ?? '').split('\n').slice(-2).join(' ').slice(0, 140)}`)

  /*
   * The one that matters most. llama.cpp decodes video at a fixed rate this build cannot
   * configure; muxing below it makes the decoder duplicate every frame. Measured directly:
   * re-decode the clip at the server's rate and count what comes back.
   */
  /*
   * Read out of the source rather than restated here. Restating it would test that 4 fps muxes
   * to 4 fps — true regardless of what the app does — and would sit there passing if the encode
   * rate drifted back down, which is precisely the regression this exists to catch.
   */
  const source = await fsp.readFile(path.join(ROOT, 'src', 'main', 'chat', 'multimodal.ts'), 'utf8')
  const declared = source.match(/const SERVER_DECODE_FPS\s*=\s*([0-9.]+)/)
  check('the app declares a server decode rate', !!declared,
    'SERVER_DECODE_FPS not found in multimodal.ts')
  const SERVER_DECODE_FPS = declared ? Number(declared[1]) : 4
  check('that rate matches what this build of llama-server decodes at', SERVER_DECODE_FPS === 4,
    `source says ${SERVER_DECODE_FPS}; measured against Qwen3.8-27B, muxing below 4 doubles the token cost`)

  const N = 12
  const src = path.join(work, 'src')
  await fsp.mkdir(src, { recursive: true })
  await ff(['-y', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
    '-i', `testsrc2=size=64x64:rate=${N}:duration=1`, '-frames:v', String(N),
    path.join(src, 'f_%03d.png')])
  const frames = (await fsp.readdir(src)).filter((f) => f.endsWith('.png')).sort()
    .map((f) => path.join(src, f))

  const dur = (1 / SERVER_DECODE_FPS).toFixed(4)
  const list = path.join(work, 'frames.txt')
  await fsp.writeFile(list,
    frames.map((f) => `file '${f.replace(/\\/g, '/')}'\nduration ${dur}`).join('\n') +
      `\nfile '${frames[frames.length - 1].replace(/\\/g, '/')}'\n`, 'utf8')
  const condensed = path.join(work, 'condensed.mp4')
  await ff(['-hide_banner', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
    '-i', list, '-r', String(SERVER_DECODE_FPS), '-pix_fmt', 'yuv420p', condensed])

  const rate = (await run(FFPROBE, ['-v', 'error', '-select_streams', 'v',
    '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', condensed]).catch(() => ({ stdout: '' }))).stdout.trim()
  check('the condensed clip is muxed at the rate the server decodes at',
    rate === `${SERVER_DECODE_FPS}/1`, `got ${rate || 'nothing'}, expected ${SERVER_DECODE_FPS}/1`)

  const out = path.join(work, 'out')
  await fsp.mkdir(out, { recursive: true })
  await ff(['-y', '-hide_banner', '-loglevel', 'error', '-i', condensed,
    '-vf', `fps=${SERVER_DECODE_FPS}`, path.join(out, 'g_%03d.png')])
  const decoded = (await fsp.readdir(out)).filter((f) => f.endsWith('.png')).length
  /*
   * One frame in, one frame out. At 2 fps this came back as 26 for 13 — every picture twice.
   * A margin of one covers the trailing frame the concat demuxer needs to give the last entry
   * a duration.
   */
  check('decoding it at the server rate does not duplicate frames',
    Math.abs(decoded - (N + 1)) <= 1, `${N + 1} frames in, ${decoded} out`)

  // A crop chain built the way sampleVideo builds it has to survive the same filter graph.
  const cropped = path.join(work, 'crop')
  await fsp.mkdir(cropped, { recursive: true })
  await ff(['-y', '-hide_banner', '-loglevel', 'error',
    '-f', 'lavfi', '-i', 'testsrc2=size=320x240:rate=4:duration=1',
    '-vf', "crop=160:120:80:60,mpdecimate=hi=64*12:lo=64*5:frac=0.33,scale='min(112,iw)':-2",
    '-fps_mode', 'passthrough', '-q:v', '3', path.join(cropped, 'c-%03d.jpg')])
  const cropFrames = (await fsp.readdir(cropped)).filter((f) => f.endsWith('.jpg'))
  check('the crop, decimate and scale chain runs as one filter graph', cropFrames.length >= 1,
    `produced ${cropFrames.length} frames`)
}

await fsp.rm(work, { recursive: true, force: true }).catch(() => undefined)

console.log(`\n${pass} passed, ${fail} failed\n`)
process.exit(fail ? 1 : 0)
