/**
 * The tools a chat's model has: its scratchpad, and watching a video in it. Nothing else.
 *
 * Chat is not the agent: it cannot read the user's files, run anything or browse. These reach only
 * the conversation's own scratchpad, by name, so a model asking for a path outside it has no way
 * to say one. Every tool is read-tier, so none of them ever stops a chat to ask permission.
 *
 * The descriptions never change with settings: tool definitions are part of every prompt, and a
 * description that carried, say, the current idle limit would invalidate the prompt cache whenever
 * the setting moved. Exact figures go in what the tools return instead.
 */

import type { ToolDefinition } from '@shared/types'
import type { ToolOutput } from '../agent/tools/base'
import { schema, str, int, bool } from '../agent/tools/base'
import { lookAtVideo } from '../agent/tools/video'
import { MAX_TEXT_BYTES, ScratchpadError, type Scratchpad, type ScratchItem } from './store'

const num = (description: string): Record<string, unknown> => ({ type: 'number', description })

export const CHAT_TOOLS: ToolDefinition[] = [
  {
    name: 'watch_video',
    tier: 'read',
    description:
      'Look at a video in your scratchpad. With no times it shows frames spread across the whole video, which ' +
      'is how to find out what is in it. Give start and end, in seconds, to look at one part, and more frames ' +
      'or higher detail to see it more closely. Every frame is labelled with its time. Look as many times as ' +
      'you need: a cheap overview first, then closer looks at the moments that matter, is far cheaper than ' +
      'asking for everything at once. You see pictures only, not sound. When you have finished with a video, ' +
      'delete it with scratchpad_delete.',
    parameters: schema(
      {
        name: str('The video\'s name in your scratchpad, as given when it was attached'),
        start: num('Where to start looking, in seconds from the beginning (default: the beginning)'),
        end: num('Where to stop looking, in seconds (default: the end of the video)'),
        frames: int('How many frames to look at across that stretch (default 8)'),
        detail: str("'low' (default; small frames, cheap), 'medium' (faces, objects, large text) or 'high' (small text; few frames at a time)")
      },
      ['name']
    )
  },
  {
    name: 'scratchpad_list',
    tier: 'read',
    description:
      'List what is in your scratchpad: the files you have been given or have written in this conversation, ' +
      'with their size and how long since you last used them. The scratchpad is temporary working space. Files ' +
      'you have not used for a while are deleted automatically.',
    parameters: schema({})
  },
  {
    name: 'scratchpad_read',
    tier: 'read',
    description: 'Read a text file from your scratchpad, such as notes you wrote there earlier.',
    parameters: schema(
      {
        name: str('The file\'s name in your scratchpad'),
        offset: int('Where to start reading, in characters from the beginning (default 0), for a file too long to read at once')
      },
      ['name']
    )
  },
  {
    name: 'scratchpad_write',
    tier: 'read',
    description:
      'Write a text file to your scratchpad, to keep notes while you work, such as what you have seen so far in ' +
      'a long video. Replaces the file unless append is true. Like everything in the scratchpad it is temporary: ' +
      'delete it when you have finished with it.',
    parameters: schema(
      {
        name: str('A file name, such as notes.md'),
        content: str('The text to write'),
        append: bool('Add to the end of the file instead of replacing it (default false)')
      },
      ['name', 'content']
    )
  },
  {
    name: 'scratchpad_delete',
    tier: 'read',
    description:
      'Delete a file from your scratchpad once you have finished with it. Do this as soon as you no longer need ' +
      'a file, such as a video you have finished watching: the scratchpad is temporary working space, not storage.',
    parameters: schema({ name: str('The file\'s name in your scratchpad') }, ['name'])
  }
]

export const CHAT_TOOL_NAMES = new Set(CHAT_TOOLS.map((t) => t.name))

export interface ChatToolContext {
  chatId: string
  pad: Scratchpad
  /** How ffmpeg reaches a file held in memory: a loopback URL. */
  urlFor: (id: string) => Promise<string>
  vision: boolean
  signal: AbortSignal
  /** The idle limit now in force, in minutes, for what the list reports. */
  idleMinutes: number
  now?: () => number
}

function size(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`
  return `${bytes} bytes`
}

function ago(ms: number): string {
  const m = Math.floor(ms / 60_000)
  if (m < 1) return 'just now'
  return m === 1 ? '1 minute ago' : `${m} minutes ago`
}

const READ_CHUNK = 12_000

/** What to say when a name is not in the scratchpad: why it went, or what is there instead. */
function missing(ctx: ChatToolContext, name: string): string {
  const why = ctx.pad.whyGone(ctx.chatId, name)
  if (why) return `${name} is no longer in your scratchpad: ${why}. If you still need it, ask the user to attach it again.`
  const here = ctx.pad.list(ctx.chatId).map((i) => i.name)
  return here.length
    ? `There is nothing called ${name} in your scratchpad. It holds: ${here.join(', ')}.`
    : `There is nothing called ${name} in your scratchpad. It is empty.`
}

export async function runChatTool(name: string, args: Record<string, unknown>, ctx: ChatToolContext): Promise<ToolOutput> {
  const wanted = typeof args.name === 'string' ? args.name.trim() : ''
  const find = (): ScratchItem | undefined => (wanted ? ctx.pad.get(ctx.chatId, wanted) : undefined)
  try {
    switch (name) {
      case 'watch_video': {
        if (!wanted) return 'Say which video to look at, by its name in your scratchpad.'
        const item = find()
        if (!item) return missing(ctx, wanted)
        if (item.kind !== 'video') return `${item.name} is not a video. Read it with scratchpad_read instead.`
        ctx.pad.touch(item)
        const source = ctx.pad.pathOf(item.id) ?? (await ctx.urlFor(item.id))
        const out = await lookAtVideo(source, item.name, args, {
          vision: ctx.vision,
          signal: ctx.signal,
          again:
            `To see a moment more closely, call watch_video again with name "${item.name}", start and end around it, ` +
            'and more frames or higher detail. Delete it with scratchpad_delete once you have finished with it.'
        })
        // Looking took a while; the clock starts again from when it finished.
        ctx.pad.touch(item)
        return out
      }

      case 'scratchpad_list': {
        const items = ctx.pad.list(ctx.chatId)
        const now = (ctx.now ?? Date.now)()
        if (!items.length) return 'Your scratchpad is empty.'
        const lines = items.map((i) => `- ${i.name}: ${i.kind}, ${size(i.bytes)}, last used ${ago(now - i.usedAt)}`)
        return (
          `Your scratchpad holds ${items.length} file${items.length === 1 ? '' : 's'}:\n${lines.join('\n')}\n` +
          `Files not used for ${ctx.idleMinutes} minute${ctx.idleMinutes === 1 ? '' : 's'} are deleted automatically.`
        )
      }

      case 'scratchpad_read': {
        if (!wanted) return 'Say which file to read, by its name in your scratchpad.'
        const item = find()
        if (!item) return missing(ctx, wanted)
        if (item.kind === 'video') return `${item.name} is a video. Look at it with watch_video instead.`
        if (item.kind !== 'text') return `${item.name} is not a text file, so it cannot be read as text.`
        ctx.pad.touch(item)
        const text = await ctx.pad.readText(item)
        const offset = Math.max(0, Math.floor(Number(args.offset ?? 0)) || 0)
        if (offset >= text.length && text.length) return `${item.name} is ${text.length.toLocaleString()} characters long, so there is nothing at ${offset.toLocaleString()}.`
        const part = text.slice(offset, offset + READ_CHUNK)
        const rest = text.length - (offset + part.length)
        return rest > 0
          ? `${part}\n\n[${rest.toLocaleString()} more characters: read again with offset ${(offset + part.length).toLocaleString()}]`
          : part || `${item.name} is empty.`
      }

      case 'scratchpad_write': {
        if (!wanted) return 'Give the file a name, such as notes.md.'
        if (typeof args.content !== 'string') return 'Give the text to write as content.'
        const existing = find()
        if (existing && existing.kind !== 'text') return `${existing.name} is a ${existing.kind}, not a text file, so it cannot be written to. Choose another name.`
        if (Buffer.byteLength(args.content) > MAX_TEXT_BYTES) return `That is more than ${MAX_TEXT_BYTES / 1024 / 1024} MB of text, the most a file here can hold.`
        const item = await ctx.pad.writeText(ctx.chatId, wanted, args.content, args.append === true)
        return `${args.append === true && existing ? 'Added to' : 'Wrote'} ${item.name} (${size(item.bytes)}).`
      }

      case 'scratchpad_delete': {
        if (!wanted) return 'Say which file to delete, by its name in your scratchpad.'
        const item = find()
        if (!item) return missing(ctx, wanted)
        await ctx.pad.remove(ctx.chatId, item.name, 'you deleted it when you had finished with it')
        return `Deleted ${item.name} from your scratchpad.`
      }

      default:
        return `There is no tool called ${name} here. The tools in this chat are ${CHAT_TOOLS.map((t) => t.name).join(', ')}.`
    }
  } catch (err) {
    if (err instanceof ScratchpadError) return err.message
    throw err
  }
}
