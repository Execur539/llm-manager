/**
 * Git.
 *
 * Split by tier rather than bundled behind one subcommand argument, because tier is what the
 * permission gate reads: reading a repository's state should cost nothing, while changing it
 * should ask. Routing everything through `run_command` meant `git status` was execute-tier and
 * stopped for approval — so orienting in a repository took several prompts before any work
 * began, and the user learned to approve shell commands reflexively.
 *
 * Every call goes through `execFile` with an argument array, never a shell string, so a branch
 * name or a commit message cannot turn into shell syntax.
 */

import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import type { Tool, ToolContext } from './base'
import { schema, str, int, bool } from './base'

const exec = promisify(execFile)

/** Output above this is truncated by the caller anyway; this stops a huge diff arriving at all. */
const MAX_BUFFER = 8 * 1024 * 1024

async function git(args: string[], ctx: ToolContext): Promise<string> {
  try {
    const { stdout, stderr } = await exec('git', args, {
      cwd: ctx.cwd,
      windowsHide: true,
      maxBuffer: MAX_BUFFER,
      timeout: ctx.timeoutMs,
      signal: ctx.signal
    })
    const out = `${stdout}${stderr && stderr.trim() ? `\n[stderr]\n${stderr.trimEnd()}` : ''}`.trim()
    return out || '(no output)'
  } catch (err) {
    const e = err as NodeJS.ErrnoException & { stdout?: string; stderr?: string; code?: string | number }
    if (e.code === 'ENOENT') {
      throw new Error('git is not installed or not on PATH. Install Git for Windows to use these tools.')
    }
    // git uses exit codes to mean things — 1 from `diff --quiet` is "there are changes" — so its
    // own output is far more useful to the model than the exit status.
    const detail = `${e.stdout ?? ''}${e.stderr ?? ''}`.trim()
    throw new Error(detail || (err instanceof Error ? err.message : String(err)))
  }
}

/** Refuse anything that is not plainly a ref/branch name, before it reaches git. */
function safeRef(value: unknown, label: string): string {
  const ref = String(value ?? '').trim()
  if (!ref) throw new Error(`${label} is required.`)
  if (ref.startsWith('-')) throw new Error(`${label} cannot start with "-"; that would be read as an option.`)
  return ref
}

// ---------------------------------------------------------------- read

const gitStatus: Tool = {
  name: 'git_status',
  description:
    'Show the working tree status of the repository in the working directory: branch, staged ' +
    'and unstaged changes, untracked files. Read-only, so it runs without asking.',
  tier: 'read',
  parameters: schema({}),
  async run(_args, ctx) {
    const branch = await git(['rev-parse', '--abbrev-ref', 'HEAD'], ctx).catch(() => '(no commits yet)')
    const status = await git(['status', '--porcelain=v1', '--branch'], ctx)
    return `On branch ${branch.trim()}\n\n${status}`
  }
}

const gitDiff: Tool = {
  name: 'git_diff',
  description:
    'Show changes as a unified diff. By default the unstaged working-tree changes; set staged ' +
    'for what is about to be committed, or give a ref to compare against. Read-only.',
  tier: 'read',
  parameters: schema({
    path: str('Limit to one file or directory'),
    staged: bool('Show staged changes instead of unstaged'),
    ref: str('Compare against this commit or branch instead of the working tree')
  }),
  async run(args, ctx) {
    const argv = ['diff']
    if (args.staged) argv.push('--staged')
    if (args.ref) argv.push(safeRef(args.ref, 'ref'))
    if (args.path) argv.push('--', String(args.path))
    const out = await git(argv, ctx)
    return out === '(no output)' ? 'No changes.' : out
  }
}

const gitLog: Tool = {
  name: 'git_log',
  description:
    'Recent commits, newest first: hash, author, relative date and subject. Use path to see the ' +
    'history of one file. Read-only.',
  tier: 'read',
  parameters: schema({
    limit: int('How many commits (default 20)'),
    path: str('Only commits touching this file or directory')
  }),
  async run(args, ctx) {
    const limit = Math.max(1, Math.min(200, Number(args.limit ?? 20)))
    const argv = ['log', `-n${limit}`, '--pretty=format:%h  %an  %ar  %s']
    if (args.path) argv.push('--', String(args.path))
    return git(argv, ctx)
  }
}

const gitShow: Tool = {
  name: 'git_show',
  description: 'Show one commit in full — its message and its diff. Read-only.',
  tier: 'read',
  parameters: schema({ ref: str('Commit hash, tag or branch (default HEAD)') }),
  async run(args, ctx) {
    return git(['show', args.ref ? safeRef(args.ref, 'ref') : 'HEAD'], ctx)
  }
}

// ---------------------------------------------------------------- write

const gitCommit: Tool = {
  name: 'git_commit',
  description:
    'Commit changes. Give paths to stage exactly those files, or set all to stage every tracked ' +
    'change. Never commits without staging something, so an empty commit cannot happen by ' +
    'accident. Asks for approval, and the prompt shows the message.',
  tier: 'write',
  parameters: schema(
    {
      message: str('Commit message. First line is the subject.'),
      paths: { type: 'array', items: { type: 'string' }, description: 'Files to stage; omit and set all instead' },
      all: bool('Stage every tracked modification')
    },
    ['message']
  ),
  async run(args, ctx) {
    const message = String(args.message ?? '').trim()
    if (!message) throw new Error('A commit message is required.')

    const paths = (args.paths as string[] | undefined) ?? []
    if (paths.length) await git(['add', '--', ...paths], ctx)
    else if (args.all) await git(['add', '-A'], ctx)
    else throw new Error('Nothing staged: pass paths, or set all to stage every tracked change.')

    const staged = await git(['diff', '--cached', '--name-only'], ctx)
    if (staged === '(no output)') return 'Nothing to commit — the staged set is empty.'

    const out = await git(['commit', '-m', message], ctx)
    return `${out}\n\nStaged:\n${staged}`
  }
}

const gitBranch: Tool = {
  name: 'git_branch',
  description: 'List branches, or create one. Creating does not switch to it; use git_checkout for that.',
  tier: 'write',
  parameters: schema({ name: str('Branch to create; omit to list branches') }),
  async run(args, ctx) {
    if (!args.name) return git(['branch', '--list', '-vv'], ctx)
    return git(['branch', safeRef(args.name, 'name')], ctx)
  }
}

const gitCheckout: Tool = {
  name: 'git_checkout',
  description:
    'Switch to a branch or commit, optionally creating the branch first. Refuses to discard ' +
    'uncommitted changes — commit or stash them first.',
  tier: 'write',
  parameters: schema({ ref: str('Branch, tag or commit to switch to'), create: bool('Create the branch first') }, ['ref']),
  async run(args, ctx) {
    const ref = safeRef(args.ref, 'ref')
    /*
     * `git switch`/`checkout` already refuses to clobber uncommitted work, but the message it
     * gives is easy for a model to read as a transient failure and retry. Checking first lets
     * the refusal say what to do about it.
     */
    const dirty = await git(['status', '--porcelain=v1'], ctx)
    if (dirty !== '(no output)') {
      return `Refusing to switch: there are uncommitted changes.\n\n${dirty}\n\nCommit them or stash them first.`
    }
    return git(args.create ? ['checkout', '-b', ref] : ['checkout', ref], ctx)
  }
}

const gitStash: Tool = {
  name: 'git_stash',
  description:
    'Save uncommitted changes aside (push), bring the most recent set back (pop), or list what ' +
    'is stashed. Nothing here deletes a stash.',
  tier: 'write',
  parameters: schema({ action: str("'push' (default), 'pop' or 'list'"), message: str('Label for a push') }),
  async run(args, ctx) {
    const action = String(args.action ?? 'push').toLowerCase()
    if (action === 'list') return git(['stash', 'list'], ctx)
    if (action === 'pop') return git(['stash', 'pop'], ctx)
    if (action !== 'push') throw new Error(`Unknown action "${action}". Use push, pop or list.`)
    const argv = ['stash', 'push']
    if (args.message) argv.push('-m', String(args.message))
    return git(argv, ctx)
  }
}

export const gitReadTools: Tool[] = [gitStatus, gitDiff, gitLog, gitShow]
export const gitWriteTools: Tool[] = [gitCommit, gitBranch, gitCheckout, gitStash]
export const gitTools: Tool[] = [...gitReadTools, ...gitWriteTools]
