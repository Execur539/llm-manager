/**
 * Permission engine.
 *
 * Tiered auto-approve (decided in Round 12):
 *   read    -> runs freely
 *   write   -> prompts
 *   execute -> prompts
 *
 * Above all of it sits a hard-block list for operations that are not recoverable. It is
 * overridable, but only through a deliberately buried setting that requires typing a
 * confirmation phrase — the point is that no *model* can talk its way past it, and no single
 * mis-click can either.
 *
 * Scope is machine-wide by decision, so this gate and the checkpoint system are the only
 * containment that exists. Paths and commands are resolved before display so an approval
 * prompt shows the real target rather than a misleading relative path or symlink.
 */

import path from 'node:path'
import fs from 'node:fs'
import type { PermissionDecision, PermissionRequest, ToolTier } from '@shared/types'

export interface PermissionRule {
  tool: string
  /** when set, the rule only matches this exact resolved command/argument string */
  exact?: string
  /** folder the rule applies to; rules are remembered per working directory */
  scope: string
}

/**
 * Operations that are refused in every permission mode unless hard blocks are disabled.
 *
 * Deliberately narrow: this is protection against a confused 7B model calling the wrong
 * tool, not an attempt to stop a determined user. Everything here is either unrecoverable
 * or disables the machine's own defences.
 */
const HARD_BLOCK_PATTERNS: { re: RegExp; reason: string }[] = [
  { re: /\bformat\s+[a-z]:/i, reason: 'Formats a drive' },
  { re: /\bdiskpart\b/i, reason: 'Low-level disk partitioning' },
  { re: /\b(mkfs|fdisk)\b/i, reason: 'Filesystem/partition destruction' },
  { re: /\brm\s+(-[a-z]*[rf][a-z]*\s+)+\/(\s|$)/i, reason: 'Recursive delete of the filesystem root' },
  { re: /\bRemove-Item\b[^\n]*\b-Recurse\b[^\n]*\b([a-z]:\\?|\\)\s*(["']|$)/i, reason: 'Recursive delete of a drive root' },
  { re: /\brd\s+\/s\s+\/q\s+[a-z]:\\?\s*$/i, reason: 'Recursive delete of a drive root' },
  { re: /\bSet-MpPreference\b[^\n]*\bDisableRealtimeMonitoring\s+\$?true/i, reason: 'Disables Windows Defender' },
  { re: /\bbcdedit\b/i, reason: 'Modifies the boot configuration' },
  { re: /\bbootrec\b/i, reason: 'Modifies the bootloader' },
  { re: /\b(vssadmin\s+delete\s+shadows)\b/i, reason: 'Deletes shadow copies, destroying restore points' },
  { re: /\bcipher\s+\/w/i, reason: 'Wipes free space irrecoverably' },
  { re: /\b(shutdown|Restart-Computer)\b/i, reason: 'Shuts down or restarts the machine' }
]

/** Directories the agent may never read, regardless of tier — app secrets live here. */
const SECRET_FILES = ['secrets.json']

export function isSecretPath(p: string, appDataDir: string): boolean {
  const resolved = path.resolve(p).toLowerCase()
  return SECRET_FILES.some((f) => resolved === path.join(appDataDir, f).toLowerCase())
}

export function checkHardBlock(commandOrPath: string): { blocked: boolean; reason?: string } {
  for (const { re, reason } of HARD_BLOCK_PATTERNS) {
    if (re.test(commandOrPath)) return { blocked: true, reason }
  }
  return { blocked: false }
}

/**
 * Render a fully-resolved, human-readable description of what a call will actually do.
 * This is what the approval prompt shows — the model's own phrasing never appears here,
 * so prompt-injected intent has nowhere to hide.
 */
export function describeCall(tool: string, args: Record<string, unknown>, cwd: string): string {
  const resolvePath = (v: unknown): string => {
    if (typeof v !== 'string') return String(v)
    try {
      const abs = path.resolve(cwd, v)
      // Resolve symlinks so the prompt shows the true destination.
      return fs.existsSync(abs) ? fs.realpathSync.native(abs) : abs
    } catch {
      return path.resolve(cwd, v)
    }
  }

  switch (tool) {
    case 'write_file':
      return `Write ${byteLabel(args.content)} to ${resolvePath(args.path)}`
    case 'edit_file':
      return `Edit ${resolvePath(args.path)}`
    case 'delete_file':
      return `Delete ${resolvePath(args.path)}`
    case 'move_file':
      return `Move ${resolvePath(args.from)} -> ${resolvePath(args.to)}`
    case 'copy_file':
      return `Copy ${resolvePath(args.from)} -> ${resolvePath(args.to)}`
    case 'run_command':
    case 'run_background':
      return `Run in ${args.cwd ? resolvePath(args.cwd) : cwd}:\n${String(args.command ?? '')}`
    case 'run_python':
      return `Execute Python (${byteLabel(args.code)})`
    case 'run_node':
      return `Execute Node (${byteLabel(args.code)})`
    case 'http_request':
      return `${String(args.method ?? 'GET')} ${String(args.url ?? '')}`
    case 'browser_navigate':
      return `Open ${String(args.url ?? '')} in the automation browser`
    case 'type_text':
      return `Type into the focused window: ${truncate(String(args.text ?? ''), 120)}`
    case 'click_mouse':
      return `Click at (${args.x}, ${args.y})`
    default: {
      const summary = Object.entries(args)
        .map(([k, v]) => `${k}=${truncate(typeof v === 'string' ? v : JSON.stringify(v), 80)}`)
        .join(', ')
      return `${tool}(${summary})`
    }
  }
}

function byteLabel(v: unknown): string {
  const s = typeof v === 'string' ? v : JSON.stringify(v ?? '')
  return `${Buffer.byteLength(s)} bytes`
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n)}...` : s
}

export class PermissionEngine {
  private rules: PermissionRule[] = []
  /**
   * Calls denied during the current turn.
   *
   * A model that has just been refused often asks for exactly the same thing again. Prompting
   * afresh each time turns one refusal into a barrage of identical dialogs, and the user cannot
   * make it stop except by aborting. An identical repeat is refused straight away, and the
   * refusal text tells the model the decision already stands.
   */
  private deniedThisTurn = new Set<string>()

  /** Called at the start of each turn: a new turn is a new chance to say yes. */
  resetTurn(): void {
    this.deniedThisTurn.clear()
  }

  constructor(
    private opts: {
      hardBlocksDisabled: () => boolean
      /** invoked when a decision is needed; resolves with the user's choice */
      ask: (req: PermissionRequest) => Promise<PermissionDecision>
    }
  ) {}

  /** Restore remembered rules (persisted per working directory). */
  load(rules: PermissionRule[]): void {
    this.rules = rules
  }

  export(): PermissionRule[] {
    return this.rules
  }

  private matches(tool: string, resolved: string, cwd: string): boolean {
    return this.rules.some(
      (r) =>
        r.tool === tool &&
        pathsEqual(r.scope, cwd) &&
        (r.exact === undefined || r.exact === resolved)
    )
  }

  /**
   * Decide whether a call may proceed. Read-tier calls never prompt.
   * Returns a reason string when denied so the model gets an explanation it can react to.
   */
  async authorise(
    tool: string,
    tier: ToolTier,
    args: Record<string, unknown>,
    cwd: string
  ): Promise<{ allowed: boolean; reason?: string }> {
    const resolved = describeCall(tool, args, cwd)

    // Hard blocks apply to every tier and outrank every remembered rule.
    const probe = [
      typeof args.command === 'string' ? args.command : '',
      typeof args.path === 'string' ? args.path : '',
      typeof args.to === 'string' ? args.to : ''
    ].join(' ')
    const hard = checkHardBlock(probe)
    if (hard.blocked && !this.opts.hardBlocksDisabled()) {
      return {
        allowed: false,
        reason: `Refused by the hard-block list: ${hard.reason}. This cannot be approved without disabling hard blocks in Settings.`
      }
    }

    if (tier === 'read') return { allowed: true }

    if (this.matches(tool, resolved, cwd)) return { allowed: true }

    const signature = `${tool}::${resolved}`
    if (this.deniedThisTurn.has(signature)) {
      return {
        allowed: false,
        reason:
          'You already requested this exact action in this turn and the user denied it. That ' +
          'decision stands — do not ask again. Take a different approach or explain why you cannot.'
      }
    }

    const decision = await this.opts.ask({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      tool,
      tier,
      args,
      resolved,
      hardBlocked: hard.blocked,
      blockReason: hard.reason
    })

    switch (decision) {
      case 'allow-once':
        return { allowed: true }
      case 'allow-tool':
        this.rules.push({ tool, scope: cwd })
        return { allowed: true }
      case 'allow-exact':
        this.rules.push({ tool, exact: resolved, scope: cwd })
        return { allowed: true }
      case 'deny':
      default:
        this.deniedThisTurn.add(signature)
        return {
          allowed: false,
          reason: 'The user denied this action. Do not retry it — choose another approach.'
        }
    }
  }
}

function pathsEqual(a: string, b: string): boolean {
  return path.resolve(a).toLowerCase() === path.resolve(b).toLowerCase()
}
