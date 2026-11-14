/**
 * Checks quick compact and the recall tools, in plain Node (no app, no windows).
 *
 *   node scripts/quick-compact-check.mjs
 *
 * A made-up agent session heavy with tool output is planned and compacted at every strength, and
 * searched and read back through the recall tools.
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { build } from 'esbuild'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
let pass = 0
let fail = 0
const check = (name, cond, extra = '') => {
  if (cond) {
    pass++
    console.log(`  PASS  ${name}`)
  } else {
    fail++
    console.log(`  FAIL  ${name} ${extra}`)
  }
}

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'llmm-quick-'))
await build({
  entryPoints: { qc: path.join(ROOT, 'src/main/chat/quick-compact.ts'), recall: path.join(ROOT, 'src/main/chat/recall.ts') },
  outdir: work,
  bundle: true,
  platform: 'node',
  format: 'esm',
  logLevel: 'error',
  alias: { '@shared': path.join(ROOT, 'src/shared') }
})
const qc = await import(pathToFileURL(path.join(work, 'qc.js')).href)
const rc = await import(pathToFileURL(path.join(work, 'recall.js')).href)

try {
  // Twelve turns: a question; reads, an edit, a question to the user, a failing command, a remark; an answer.
  const messages = []
  let t = 1000
  const file = (turn, k) => Array.from({ length: 300 }, (_, i) => `line ${i + 1} of file ${k} in turn ${turn}: const value${i} = compute(${i})`).join('\n')
  const tool = (id, name, args, content, ok = true, batch = 'b') => ({
    id,
    role: 'tool',
    content,
    toolCalls: [{ id: `c_${id}`, name, args, batch }],
    toolResult: { callId: `c_${id}`, content, ok, durationMs: 5 },
    createdAt: t++
  })
  for (let turn = 1; turn <= 12; turn++) {
    messages.push({ id: `u${turn}`, role: 'user', content: `Question ${turn}: please look into module ${turn}.`, createdAt: t++ })
    messages.push(tool(`t${turn}_1`, 'read_file', { path: `src/mod${turn}/file1.ts` }, `\u001b[32mok\u001b[0m\n${file(turn, 1)}`))
    messages.push(tool(`t${turn}_2`, 'read_file', { path: `src/mod${turn}/file1.ts` }, `\u001b[32mok\u001b[0m\n${file(turn, 1)}`))
    messages.push({ id: `n${turn}`, role: 'assistant', content: `The parser in module ${turn} drops quoted commas. I will patch the tokenizer instead.`, createdAt: t++ })
    messages.push(
      tool(`t${turn}_3`, 'edit_file', { path: `src/mod${turn}/tokenizer.ts`, old_text: 'x'.repeat(300), new_text: `const marmalade${turn} = splitQuoted(input)\n`.repeat(12) }, 'Edited.')
    )
    messages.push(tool(`t${turn}_q`, 'ask_user', { question: `Keep the old flag in module ${turn}?` }, `No, remove it, and call the new one strictMode${turn}.`))
    messages.push(tool(`t${turn}_4`, 'run_command', { command: 'npm test' }, `npm ERR! Test failed in turn ${turn}\nTypeError: zebra is not a function`, false))
    messages.push({
      id: `a${turn}`,
      role: 'assistant',
      reasoning: 'thinking '.repeat(200),
      content: `# Findings for ${turn}\n\nModule ${turn} computes values. It has a failing test about a zebra. The cause is the tokenizer.\n\n\`\`\`ts\nconst a = 1\nconst b = 2\n\`\`\`\n\n- first point about module ${turn}\n- second point`,
      createdAt: t++
    })
  }
  const total = (list) => list.reduce((n, m) => n + qc.costOf(m), 0)
  const at = (level, upto = 'a10') => qc.quickView(messages, { v: 2, segments: [{ upto, level }] })
  const text = (list) => list.map((m) => m.content).join('\n')

  console.log('the strengths')
  const sizes = [1, 2, 3, 4, 5].map((l) => total(at(l).messages))
  check('each strength is smaller than the one before', sizes.every((s, i) => i === 0 || s < sizes[i - 1]), sizes.join(' > '))
  const l1 = at(1).messages
  check('1 drops thinking and colour codes, keeps the text', !l1.find((m) => m.id === 'a3').reasoning && !l1.find((m) => m.id === 't3_2').content.includes('\u001b') && l1.find((m) => m.id === 't3_2').content.includes('line 300'))
  const l2 = at(2).messages
  check('2 drops the earlier of two identical reads, not the later', /same call was made again/.test(l2.find((m) => m.id === 't3_1').content) && l2.find((m) => m.id === 't3_2').content.includes('line 300'))
  const l3 = at(3)
  const stubbed = l3.messages.find((m) => m.id === 't3_2')
  check('3 turns a result into one line that says how to get it back', /301 lines, starting "ok"/.test(stubbed.content) && stubbed.content.includes('recall_read("t3_2")') && stubbed.content.length < 300, stubbed.content)
  check('a long argument is left out with the way back to it', /left out\. recall_read\("t3_3"\)/.test(l3.messages.find((m) => m.id === 't3_3').toolCalls[0].args.new_text))
  check("the user's answer to a question is never shortened", l3.messages.find((m) => m.id === 't3_q').content.includes('strictMode3'))
  check('attachments stop being sent for the compacted part only', l3.dropSent.has('u1') && !l3.dropSent.has('u11'))
  const l4 = at(4).messages
  const turn3 = l4.find((m) => m.id === 'u3').content
  check('4 makes each turn a question and an answer', l4.slice(0, 20).every((m, i) => m.role === (i % 2 ? 'assistant' : 'user')))
  check('the account says which files changed, apart from those looked at', /changed: src\/mod3\/tokenizer\.ts/.test(turn3) && /looked at: src\/mod3\/file1\.ts/.test(turn3), turn3)
  check('commands are said to have been run, not looked at', /ran: npm test/.test(turn3) && !/looked at:[^.]*npm test/.test(turn3), turn3)
  const gentle = at(1)
  check('the opening note travels with a recorded copy of the first message too', gentle.preface?.id === 'u1' && qc.withPreface(gentle, 'u1', 'as sent').startsWith('[The earlier part') && qc.withPreface(gentle, 'u2', 'as sent') === 'as sent' && qc.withPreface(gentle, 'u1', [{ type: 'text', text: 'x' }]).length === 2)
  check('and what failed, with its error', /failed: run_command npm test: npm ERR! Test failed in turn 3/.test(turn3))
  check("and the user's answer, in full", turn3.includes('the user answered "No, remove it, and call the new one strictMode3."'))
  check('and what the model remarked on the way', turn3.includes('The parser in module 3 drops quoted commas.'))
  check('and its turn number, for searching', /\[Turn 3, shortened\./.test(turn3) && /from_turn 3 and to_turn 3/.test(turn3))
  check('the answer is whole', l4.find((m) => m.id === 'a3').content.includes('const a = 1'))
  check('the model is told once that the start is shortened and how to look back', l4[0].content.startsWith('[The earlier part of this conversation is shown in a shortened form') && !l4[2].content.startsWith('[The earlier'))
  const l5 = at(5).messages.find((m) => m.id === 'a3').content
  check('5 outlines the answer but keeps a short code block and two sentences', l5.includes('# Findings for 3') && l5.includes('const a = 1') && l5.includes('failing test about a zebra.') && !l5.includes('The cause is the tokenizer'), l5)
  check('what comes after the compacted part is untouched', JSON.stringify(at(5).messages.slice(-16)) === JSON.stringify(messages.slice(-16)))
  check('a model without tools is not told about tools it does not have', !text(qc.quickView(messages, { v: 2, segments: [{ upto: 'a10', level: 4 }] }, { recall: false }).messages).includes('recall_'))

  console.log('stretches')
  const two = { v: 2, segments: [{ upto: 'a6', level: 5 }, { upto: 'a10', level: 3 }] }
  const mixed = qc.quickView(messages, two).messages
  check('a later stretch can be gentler than an earlier one', !mixed.find((m) => m.id === 'a2').content.includes('The cause is the tokenizer') && mixed.some((m) => m.id === 't8_1'))
  const first = JSON.stringify(qc.quickView(messages, { v: 2, segments: [{ upto: 'a6', level: 5 }] }).messages.slice(0, 12))
  check('adding a stretch leaves the earlier one exactly as it was', JSON.stringify(mixed.slice(0, 12)) === first)
  check('a stretch whose end was rewound away stops applying, with those after it', qc.quickView(messages, { v: 2, segments: [{ upto: 'gone', level: 5 }, { upto: 'a10', level: 5 }] }).messages === messages)
  const behind = qc.quickView(messages, two, { skip: messages.findIndex((m) => m.id === 'a6') + 1 }).messages
  check('a stretch a summary now stands in front of is passed over', behind[0].id === 'u7' && behind.some((m) => m.id === 't8_1'))
  const midTurn = qc.quickView(messages, { v: 2, segments: [{ upto: 't12_3', level: 4 }] }).messages
  const open = midTurn.find((m) => m.id === 'u12')
  check('a stretch can end part way through a turn', !!open && /changed: src\/mod12\/tokenizer\.ts/.test(open.content) && midTurn[midTurn.indexOf(open) + 1].id === 't12_q', midTurn.slice(-5).map((m) => m.id).join(','))

  console.log('planning')
  const before = total(messages)
  const plan = qc.planQuick(messages, before * 4, 500)
  check('every strength is offered with the size it would leave', plan.ok && plan.options.length === 5 && plan.options.every((o, i) => i === 0 || o.afterTokens <= plan.options[i - 1].afterTokens), JSON.stringify(plan.options?.map((o) => o.afterTokens)))
  check('the gentlest that reaches the target is suggested', plan.options.find((o) => o.level === plan.recommended).afterTokens <= plan.targetTokens && (plan.recommended === 1 || plan.options[plan.recommended - 2].afterTokens > plan.targetTokens), String(plan.recommended))
  const hard = plan.options[4].spec.segments
  check('at the hardest, the newer half is done one strength gentler', hard.length === 2 && hard[0].level === 5 && hard[1].level === 4 && /^a\d+$/.test(hard[0].upto), JSON.stringify(hard))
  check('the most recent messages are kept', qc.quickView(messages, plan.options[4].spec).messages.at(-1).id === 'a12')
  const oneTurn = [messages[0], ...Array.from({ length: 60 }, (_, i) => tool(`big${i}`, 'read_file', { path: `f${i}.ts` }, file(99, i)))]
  const single = qc.planQuick(oneTurn, total(oneTurn) * 2, 500)
  check('one long turn can be compacted', single.ok && single.messages > 30 && single.options[2].afterTokens < single.beforeTokens / 2, single.message)
  const tiny = qc.planQuick(messages.slice(0, 2), 100000, 500)
  check('with only the most recent messages there is nothing to compact, and it says so', !tiny.ok && /most recent messages/.test(tiny.message), tiny.message)
  const chatter = Array.from({ length: 24 }, (_, i) => ({ id: `s${i}`, role: i % 2 ? 'assistant' : 'user', content: 'plain words with no full stop '.repeat(8), createdAt: i }))
  const cramped = qc.planQuick(chatter, 12000, 11000)
  check('when little can be freed it says why', !cramped.ok && cramped.message.length > 20, cramped.message)
  const again = qc.planQuick(messages, before * 4, 500, { previous: { v: 2, segments: [{ upto: 'a4', level: 5 }] } })
  check('a second compact keeps the first as it was and offers every strength for the rest', again.ok && again.options[0].spec.segments[0].level === 5 && again.options[0].spec.segments[1].level === 1, JSON.stringify(again.options?.[0]?.spec))
  const done = qc.planQuick(messages, before * 4, 500, { previous: plan.options[2].spec })
  check('with nothing new since the last, it says so', !done.ok && /Nothing new/.test(done.message), done.message)

  console.log('recall: finding')
  const opts = { visibleFrom: messages.findIndex((m) => m.id === 'a10') + 1 }
  const found = await rc.recallSearch(messages, { query: 'zebra function' }, opts)
  check('matches are listed with ids, turns and one line each', /\[t\d+_4\] turn \d+ · tool run_command npm test \(failed\) · line \d+: TypeError: zebra is not a function/.test(found), found.slice(0, 300))
  check('never the whole result, nor anything still in front of the model', found.length < 3000 && !found.includes('line 300') && !/\[(t|a|u)1[12]/.test(found))
  const natural = await rc.recallSearch(messages, { query: 'why did the parser dropping the quoted commas' }, opts)
  check('a question in plain words still finds it: filler ignored, endings ignored', /\[n\d+\] turn \d+ · assistant/.test(natural.split('\n').find((l) => l.startsWith('[')) ?? '') && !natural.includes('No single earlier item'), natural.slice(0, 260))
  const close = await rc.recallSearch(messages, { query: 'zebra hippopotamus' }, opts)
  check('when no item has every word, the closest are given and said to be only close', /No single earlier item has all of those words/.test(close) && /zebra/.test(close), close.slice(0, 200))
  const none = await rc.recallSearch(messages, { query: 'hippopotamus' }, opts)
  check('no match says what to try next', /No match/.test(none) && /list the turns/.test(none), none)
  const args = await rc.recallSearch(messages, { query: 'marmalade3' }, opts)
  check('what a call was given is searched, not only what it returned', /\[t3_3\] turn 3 · tool edit_file/.test(args), args.slice(0, 200))
  const said = await rc.recallSearch(messages, { query: 'strictMode5', limit: 3 }, opts)
  check("the user's answer to a question is found", /\[t5_q\]/.test(said), said.slice(0, 200))
  const big = await rc.recallSearch(messages, { query: 'zebra', limit: 3 }, opts)
  check('what was said outranks a tool result with the same word', /assistant/.test(big.split('\n')[1] ?? ''), big.slice(0, 200))
  const many = await rc.recallSearch(messages, { query: 'compute', tool: 'read_file', path: 'mod3', limit: 2, context: 1 }, opts)
  check('filters narrow it; other matching lines are counted; context shows neighbours', /turn 3/.test(many) && !/turn 4/.test(many) && /\(\+\d+ more lines\)/.test(many) && />/.test(many), many.slice(0, 300))
  const errors = await rc.recallSearch(messages, { query: 'test', errors_only: true, from_turn: 2, to_turn: 3 }, opts)
  check('errors only, within turns', /2 matches/.test(errors), errors.slice(0, 120))
  const re = await rc.recallSearch(messages, { query: 'value1[0-9]{2} = compute', mode: 'regex', limit: 3 }, opts)
  check('regular expressions work and page', /more: call again with offset 3/.test(re), re.slice(0, 160))
  check('a broken one is explained', /not a valid regular expression/.test(await rc.recallSearch(messages, { query: '(', mode: 'regex' }, opts)))
  const evil = [{ id: 'e1', role: 'user', content: 'a'.repeat(3000) + '!', createdAt: 1 }, { id: 'e2', role: 'assistant', content: 'x', createdAt: 2 }]
  const started = Date.now()
  const slow = await rc.recallSearch(evil, { query: '^(a+)+$', mode: 'regex' }, { visibleFrom: 2 })
  check('a pattern that never finishes is cut off, not waited for', /takes too long/.test(slow) && Date.now() - started < 5000, `${Date.now() - started} ms: ${slow}`)

  console.log('recall: browsing and meaning')
  const toc = await rc.recallSearch(messages, {}, opts)
  check('with no query it lists the turns', /10 earlier turns/.test(toc) && /turn 3: "Question 3: please look into module 3\." · 5 tool calls/.test(toc), toc.slice(0, 300))
  const turn = await rc.recallSearch(messages, { from_turn: 3, to_turn: 3, limit: 20 }, opts)
  check('with a turn and no query, everything in it in order', /8 items, in order/.test(turn) && turn.indexOf('[u3]') < turn.indexOf('[a3]'), turn.slice(0, 200))
  const fake = async (texts) => texts.map((x) => [x.includes('zebra') || x.includes('striped horse') ? 1 : 0, x.includes('compute') ? 1 : 0, 0.1])
  const vectors = new Map()
  const meaning = await rc.recallSearch(messages, { query: 'the striped horse problem', mode: 'meaning', limit: 3 }, { ...opts, embed: fake, vectors })
  check('search by meaning finds what the words do not', /_4\]|\[a\d+\]/.test(meaning.split('\n').find((l) => l.startsWith('[')) ?? ''), meaning.slice(0, 300))
  check('it covers a bounded amount per call and says how much', /Searched by meaning across \d+ of \d+/.test(meaning) || vectors.size > 0, meaning.slice(0, 120))
  const noModel = await rc.recallSearch(messages, { query: 'zebra', mode: 'meaning' }, opts)
  check('with no embedding model it falls back to words and says so', /not available/.test(noModel) && /zebra/.test(noModel))
  check('with nothing compacted it says the conversation is all still there', /still in front of you/.test(await rc.recallSearch(messages, { query: 'zebra' }, { visibleFrom: 0 })))

  console.log('recall: reading')
  const edit = rc.recallRead(messages, { id: 't3_3' }, {})
  check('a call is read with its arguments, then its result', edit.includes('[call] edit_file') && edit.includes('const marmalade3 = splitQuoted(input)') && edit.includes('[result]') && edit.includes('Edited.'), edit.slice(0, 200))
  const part = rc.recallRead(messages, { id: 't3_2', around: 'value150' }, {})
  check('reading gives the lines around what was asked for, numbered', /lines \d+-\d+ of \d+/.test(part) && part.includes('line 151 of file 1'), part.slice(0, 120))
  const range = rc.recallRead(messages, { id: 't3_2', start_line: 10, end_line: 12 }, {})
  check('or a line range, saying how to read on', range.split('\n').length === 5 && range.includes('start_line 13'), range)
  check('a long one is capped', rc.recallRead(messages, { id: 't3_2' }, {}).length < 6600)
  check("a shortened turn's own id reads its first message", String(rc.recallRead(messages, { id: 'u3:end' }, {})).includes('Question 3'))
  const pictures = rc.recallRead(messages, { id: 'u3', images: true }, { sentParts: () => [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } }, { type: 'text', text: 'x' }] })
  check('pictures come back when asked for', typeof pictures === 'object' && pictures.media.length === 1)
  check('an unknown id is explained', /Nothing in this conversation has the id/.test(rc.recallRead(messages, { id: 'nope' }, {})))
} finally {
  fs.rmSync(work, { recursive: true, force: true })
}

console.log(`\n${pass} passed, ${fail} failed`)
process.exit(fail ? 1 : 0)
