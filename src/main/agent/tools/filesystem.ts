/**
 * Filesystem tools.
 *
 * Scope is machine-wide by decision, so there is no workspace jail here. What these do
 * provide is canonicalisation (so the approval prompt shows the real target), offset/limit
 * paging (so a huge file doesn't eat the context window), and exact-match editing (so an
 * edit either applies precisely or fails loudly rather than mangling a file).
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import type { Tool } from './base'
import { schema, str, int, bool } from './base'

const exec = promisify(execFile)

function resolve(cwd: string, p: string): string {
  return path.isAbsolute(p) ? path.normalize(p) : path.resolve(cwd, p)
}

const readFile: Tool = {
  name: 'read_file',
  description:
    'Read a text file. Returns lines with 1-based line numbers. Use offset and limit to page ' +
    'through large files rather than reading them whole.',
  tier: 'read',
  parameters: schema(
    {
      path: str('Path to the file, absolute or relative to the working directory'),
      offset: int('1-based line to start from (default 1)'),
      limit: int('Maximum number of lines to return (default 2000)')
    },
    ['path']
  ),
  async run(args, ctx) {
    const file = resolve(ctx.cwd, String(args.path))
    const offset = Math.max(1, Number(args.offset ?? 1))
    const limit = Math.max(1, Number(args.limit ?? 2000))

    const st = await fsp.stat(file)
    if (st.isDirectory()) throw new Error(`${file} is a directory; use list_dir`)
    if (st.size > 200 * 1024 * 1024) throw new Error(`${file} is ${(st.size / 1e6).toFixed(0)} MB — too large to read as text`)

    const content = await fsp.readFile(file, 'utf8')
    const lines = content.split(/\r?\n/)
    const slice = lines.slice(offset - 1, offset - 1 + limit)
    const numbered = slice.map((l, i) => `${String(offset + i).padStart(6)}\t${l}`).join('\n')

    const more = lines.length > offset - 1 + limit
    return (
      numbered +
      (more ? `\n\n[${lines.length - (offset - 1 + limit)} more lines; call again with offset=${offset + limit}]` : '')
    )
  }
}

const writeFile: Tool = {
  name: 'write_file',
  description: 'Write a file, creating parent directories as needed. Overwrites existing content.',
  tier: 'write',
  parameters: schema({ path: str('Path to write'), content: str('Full file content') }, ['path', 'content']),
  async run(args, ctx) {
    const file = resolve(ctx.cwd, String(args.path))
    await fsp.mkdir(path.dirname(file), { recursive: true })
    await fsp.writeFile(file, String(args.content), 'utf8')
    return `Wrote ${Buffer.byteLength(String(args.content))} bytes to ${file}`
  }
}

const editFile: Tool = {
  name: 'edit_file',
  description:
    'Replace an exact string in a file. The old string must appear exactly once unless ' +
    'replace_all is true. Fails loudly rather than guessing.',
  tier: 'write',
  parameters: schema(
    {
      path: str('File to edit'),
      old_string: str('Exact text to find, including indentation'),
      new_string: str('Replacement text'),
      replace_all: bool('Replace every occurrence instead of requiring exactly one')
    },
    ['path', 'old_string', 'new_string']
  ),
  async run(args, ctx) {
    const file = resolve(ctx.cwd, String(args.path))
    const oldStr = String(args.old_string)
    const newStr = String(args.new_string)
    const replaceAll = Boolean(args.replace_all)

    const content = await fsp.readFile(file, 'utf8')
    const count = content.split(oldStr).length - 1

    if (count === 0) throw new Error(`old_string not found in ${file}. The file may have changed since it was read.`)
    if (count > 1 && !replaceAll) {
      throw new Error(`old_string appears ${count} times in ${file}. Provide more surrounding context, or set replace_all.`)
    }

    const updated = replaceAll ? content.split(oldStr).join(newStr) : content.replace(oldStr, newStr)
    await fsp.writeFile(file, updated, 'utf8')
    return `Replaced ${replaceAll ? count : 1} occurrence(s) in ${file}`
  }
}

const listDir: Tool = {
  name: 'list_dir',
  description: 'List the entries of a directory with type and size.',
  tier: 'read',
  parameters: schema({ path: str('Directory to list (default: working directory)') }),
  async run(args, ctx) {
    const dir = resolve(ctx.cwd, String(args.path ?? '.'))
    const entries = await fsp.readdir(dir, { withFileTypes: true })
    const rows = await Promise.all(
      entries.map(async (e) => {
        const full = path.join(dir, e.name)
        if (e.isDirectory()) return `dir   ${''.padStart(12)}  ${e.name}/`
        try {
          const st = await fsp.stat(full)
          return `file  ${String(st.size).padStart(12)}  ${e.name}`
        } catch {
          return `?     ${''.padStart(12)}  ${e.name}`
        }
      })
    )
    return rows.length ? `${dir}\n${rows.join('\n')}` : `${dir} is empty`
  }
}

const globTool: Tool = {
  name: 'glob',
  description: 'Find files matching a glob pattern, e.g. "src/**/*.ts". Returns paths sorted by modification time.',
  tier: 'read',
  parameters: schema({ pattern: str('Glob pattern'), path: str('Root to search from') }, ['pattern']),
  async run(args, ctx) {
    const root = resolve(ctx.cwd, String(args.path ?? '.'))
    const pattern = String(args.pattern)
    const re = globToRegExp(pattern)

    const matches: { file: string; mtime: number }[] = []
    async function walk(dir: string, depth: number): Promise<void> {
      if (depth > 24 || matches.length > 5000) return
      let entries: fs.Dirent[]
      try {
        entries = await fsp.readdir(dir, { withFileTypes: true })
      } catch {
        return
      }
      for (const e of entries) {
        if (e.name === 'node_modules' || e.name === '.git') continue
        const full = path.join(dir, e.name)
        if (e.isDirectory()) await walk(full, depth + 1)
        else {
          const rel = path.relative(root, full).replace(/\\/g, '/')
          if (re.test(rel)) {
            try {
              matches.push({ file: full, mtime: (await fsp.stat(full)).mtimeMs })
            } catch {
              /* skip */
            }
          }
        }
      }
    }
    await walk(root, 0)
    matches.sort((a, b) => b.mtime - a.mtime)
    return matches.length ? matches.map((m) => m.file).join('\n') : `No files match ${pattern} under ${root}`
  }
}

function globToRegExp(pattern: string): RegExp {
  const escaped = pattern
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*\*\//g, '(?:.*/)?')
    .replace(/\*\*/g, '.*')
    .replace(/\*/g, '[^/]*')
    .replace(/\?/g, '[^/]')
  return new RegExp(`^${escaped}$`, 'i')
}

/**
 * Content search. Uses ripgrep when it is on PATH (fast, respects .gitignore) and falls
 * back to a JS scan so the tool works on a machine that has no rg.
 */
const grepTool: Tool = {
  name: 'grep',
  description: 'Search file contents with a regular expression. Returns matching lines with file:line prefixes.',
  tier: 'read',
  parameters: schema(
    {
      pattern: str('Regular expression'),
      path: str('File or directory to search (default: working directory)'),
      glob: str('Only search files matching this glob, e.g. "*.ts"'),
      case_insensitive: bool('Ignore case'),
      max_results: int('Maximum matching lines to return (default 200)')
    },
    ['pattern']
  ),
  async run(args, ctx) {
    const root = resolve(ctx.cwd, String(args.path ?? '.'))
    const pattern = String(args.pattern)
    const max = Number(args.max_results ?? 200)

    const rgArgs = ['--line-number', '--no-heading', '--color=never', '--max-count', String(max)]
    if (args.case_insensitive) rgArgs.push('-i')
    if (args.glob) rgArgs.push('--glob', String(args.glob))
    rgArgs.push(pattern, root)

    try {
      const { stdout } = await exec('rg', rgArgs, { timeout: ctx.timeoutMs, maxBuffer: 32 * 1024 * 1024 })
      return stdout.trim() || `No matches for /${pattern}/ under ${root}`
    } catch (err) {
      const e = err as { code?: number; stdout?: string }
      // ripgrep exits 1 when there are simply no matches.
      if (e.code === 1) return `No matches for /${pattern}/ under ${root}`
      if (e.stdout) return e.stdout
      return fallbackGrep(root, pattern, Boolean(args.case_insensitive), max)
    }
  }
}

async function fallbackGrep(root: string, pattern: string, ignoreCase: boolean, max: number): Promise<string> {
  const re = new RegExp(pattern, ignoreCase ? 'i' : '')
  const out: string[] = []

  async function walk(dir: string, depth: number): Promise<void> {
    if (depth > 20 || out.length >= max) return
    let entries: fs.Dirent[]
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of entries) {
      if (out.length >= max) return
      if (e.name === 'node_modules' || e.name === '.git') continue
      const full = path.join(dir, e.name)
      if (e.isDirectory()) await walk(full, depth + 1)
      else {
        try {
          const st = await fsp.stat(full)
          if (st.size > 8 * 1024 * 1024) continue
          const text = await fsp.readFile(full, 'utf8')
          text.split(/\r?\n/).forEach((line, i) => {
            if (out.length < max && re.test(line)) out.push(`${full}:${i + 1}:${line.trim()}`)
          })
        } catch {
          /* binary or unreadable */
        }
      }
    }
  }

  const st = await fsp.stat(root)
  if (st.isFile()) {
    const text = await fsp.readFile(root, 'utf8')
    text.split(/\r?\n/).forEach((line, i) => {
      if (out.length < max && re.test(line)) out.push(`${root}:${i + 1}:${line.trim()}`)
    })
  } else {
    await walk(root, 0)
  }

  return out.length ? out.join('\n') : `No matches for /${pattern}/ under ${root}`
}

const deleteFile: Tool = {
  name: 'delete_file',
  description: 'Delete a file or directory. Directories require recursive=true.',
  tier: 'write',
  parameters: schema({ path: str('Path to delete'), recursive: bool('Delete directories recursively') }, ['path']),
  async run(args, ctx) {
    const target = resolve(ctx.cwd, String(args.path))
    const st = await fsp.stat(target)
    if (st.isDirectory() && !args.recursive) throw new Error(`${target} is a directory; pass recursive=true to delete it`)
    await fsp.rm(target, { recursive: Boolean(args.recursive), force: false })
    return `Deleted ${target}`
  }
}

const moveFile: Tool = {
  name: 'move_file',
  description: 'Move or rename a file or directory.',
  tier: 'write',
  parameters: schema({ from: str('Source path'), to: str('Destination path') }, ['from', 'to']),
  async run(args, ctx) {
    const from = resolve(ctx.cwd, String(args.from))
    const to = resolve(ctx.cwd, String(args.to))
    await fsp.mkdir(path.dirname(to), { recursive: true })
    await fsp.rename(from, to)
    return `Moved ${from} -> ${to}`
  }
}

const copyFile: Tool = {
  name: 'copy_file',
  description: 'Copy a file or directory.',
  tier: 'write',
  parameters: schema({ from: str('Source path'), to: str('Destination path') }, ['from', 'to']),
  async run(args, ctx) {
    const from = resolve(ctx.cwd, String(args.from))
    const to = resolve(ctx.cwd, String(args.to))
    await fsp.mkdir(path.dirname(to), { recursive: true })
    await fsp.cp(from, to, { recursive: true })
    return `Copied ${from} -> ${to}`
  }
}

const statTool: Tool = {
  name: 'stat_path',
  description: 'Get size, timestamps and type for a path.',
  tier: 'read',
  parameters: schema({ path: str('Path to inspect') }, ['path']),
  async run(args, ctx) {
    const target = resolve(ctx.cwd, String(args.path))
    const st = await fsp.stat(target)
    return JSON.stringify(
      {
        path: target,
        type: st.isDirectory() ? 'directory' : st.isFile() ? 'file' : 'other',
        bytes: st.size,
        modified: new Date(st.mtimeMs).toISOString(),
        created: new Date(st.birthtimeMs).toISOString()
      },
      null,
      2
    )
  }
}

export const filesystemTools: Tool[] = [
  readFile,
  writeFile,
  editFile,
  listDir,
  globTool,
  grepTool,
  statTool,
  moveFile,
  copyFile,
  deleteFile
]
