/**
 * MCP client.
 *
 * This is what makes the tool set open-ended rather than a fixed list: any Model Context
 * Protocol server the user connects contributes its tools to the agent's catalog, discovered
 * at connect time. Supports stdio and streamable-HTTP transports.
 *
 * Implemented directly against the JSON-RPC wire protocol rather than pulling in an SDK, so
 * there is one less dependency inside a bundle that is already large.
 *
 * MCP v2.1 makes tool sandboxing a spec requirement. We honour the servers' own declarations
 * and surface exactly what each one exposes, but a server still runs as a local process with
 * the user's rights — so connecting one is treated as a deliberate act, not a default.
 */

import { spawn, ChildProcess } from 'node:child_process'
import crypto from 'node:crypto'
import type { ToolDefinition, ToolTier } from '@shared/types'
import type { Tool, ToolContext } from './tools/base'
import { all, run } from '../storage/db'

const PROTOCOL_VERSION = '2025-06-18'

export interface McpServerConfig {
  id: string
  name: string
  transport: 'stdio' | 'http'
  command?: string
  args?: string[]
  url?: string
  env?: Record<string, string>
  enabled: boolean
}

interface McpToolSpec {
  name: string
  description?: string
  inputSchema?: Record<string, unknown>
  annotations?: { readOnlyHint?: boolean; destructiveHint?: boolean }
}

interface JsonRpcResponse {
  jsonrpc: '2.0'
  id?: number | string
  result?: unknown
  error?: { code: number; message: string }
}

/** One live connection to an MCP server. */
class McpConnection {
  private child: ChildProcess | null = null
  private nextId = 1
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>()
  private buffer = ''
  private sessionId: string | null = null
  tools: McpToolSpec[] = []
  connected = false
  lastError: string | null = null

  constructor(public config: McpServerConfig) {}

  async connect(): Promise<void> {
    try {
      if (this.config.transport === 'stdio') await this.connectStdio()
      else await this.connectHttp()

      await this.request('initialize', {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: { name: 'LLM Manager', version: '0.1.0' }
      })
      await this.notify('notifications/initialized', {})

      const listed = (await this.request('tools/list', {})) as { tools?: McpToolSpec[] }
      this.tools = listed.tools ?? []
      this.connected = true
      this.lastError = null
    } catch (err) {
      this.connected = false
      this.lastError = err instanceof Error ? err.message : String(err)
      throw err
    }
  }

  private async connectStdio(): Promise<void> {
    if (!this.config.command) throw new Error('stdio server needs a command')
    const child = spawn(this.config.command, this.config.args ?? [], {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
      env: { ...process.env, ...(this.config.env ?? {}) }
    })
    this.child = child

    child.stdout?.on('data', (d: Buffer) => this.onStdout(d.toString()))
    child.on('exit', () => {
      this.connected = false
      for (const [, p] of this.pending) p.reject(new Error('MCP server exited'))
      this.pending.clear()
    })
    child.on('error', (err) => {
      this.lastError = err.message
    })
  }

  private async connectHttp(): Promise<void> {
    if (!this.config.url) throw new Error('http server needs a url')
    // Streamable HTTP is request/response per call; nothing to hold open here.
  }

  /** stdio framing is newline-delimited JSON. */
  private onStdout(chunk: string): void {
    this.buffer += chunk
    const lines = this.buffer.split('\n')
    this.buffer = lines.pop() ?? ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      try {
        this.handleMessage(JSON.parse(trimmed) as JsonRpcResponse)
      } catch {
        /* server noise on stdout; ignore */
      }
    }
  }

  private handleMessage(msg: JsonRpcResponse): void {
    if (msg.id === undefined) return // a notification from the server
    const entry = this.pending.get(Number(msg.id))
    if (!entry) return
    this.pending.delete(Number(msg.id))
    if (msg.error) entry.reject(new Error(`${msg.error.code}: ${msg.error.message}`))
    else entry.resolve(msg.result)
  }

  private async request(method: string, params: unknown, timeoutMs = 30000): Promise<unknown> {
    const id = this.nextId++
    const payload = { jsonrpc: '2.0' as const, id, method, params }

    if (this.config.transport === 'http') {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        Accept: 'application/json, text/event-stream'
      }
      if (this.sessionId) headers['Mcp-Session-Id'] = this.sessionId
      const res = await fetch(this.config.url as string, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(timeoutMs)
      })
      const sid = res.headers.get('Mcp-Session-Id')
      if (sid) this.sessionId = sid
      if (!res.ok) throw new Error(`MCP HTTP ${res.status}`)

      const text = await res.text()
      // A streamable-HTTP server may answer as SSE even for a single call.
      const jsonText = text.startsWith('data:')
        ? text
            .split('\n')
            .filter((l) => l.startsWith('data:'))
            .map((l) => l.slice(5).trim())
            .join('')
        : text
      const parsed = JSON.parse(jsonText) as JsonRpcResponse
      if (parsed.error) throw new Error(parsed.error.message)
      return parsed.result
    }

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`MCP request ${method} timed out`))
      }, timeoutMs)

      this.pending.set(id, {
        resolve: (v) => {
          clearTimeout(timer)
          resolve(v)
        },
        reject: (e) => {
          clearTimeout(timer)
          reject(e)
        }
      })
      this.child?.stdin?.write(`${JSON.stringify(payload)}\n`)
    })
  }

  private async notify(method: string, params: unknown): Promise<void> {
    if (this.config.transport === 'http') return
    this.child?.stdin?.write(`${JSON.stringify({ jsonrpc: '2.0', method, params })}\n`)
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<string> {
    const result = (await this.request('tools/call', { name, arguments: args }, 120000)) as {
      content?: { type: string; text?: string }[]
      isError?: boolean
    }
    const text = (result.content ?? [])
      .map((c) => (c.type === 'text' ? c.text ?? '' : `[${c.type}]`))
      .join('\n')
    if (result.isError) throw new Error(text || 'MCP tool reported an error')
    return text || '(no output)'
  }

  close(): void {
    try {
      this.child?.kill()
    } catch {
      /* best effort */
    }
    this.child = null
    this.connected = false
  }
}

class McpManager {
  private connections = new Map<string, McpConnection>()

  /** Namespaced so an MCP tool can never shadow a built-in one. */
  private qualify(serverId: string, toolName: string): string {
    return `mcp__${serverId}__${toolName}`
  }

  private parse(name: string): { serverId: string; tool: string } | null {
    const m = name.match(/^mcp__([^_]+(?:_[^_]+)*?)__(.+)$/)
    return m ? { serverId: m[1], tool: m[2] } : null
  }

  listConfigs(): McpServerConfig[] {
    return all<{
      id: string
      name: string
      transport: string
      command: string | null
      args: string | null
      url: string | null
      env: string | null
      enabled: number
    }>('SELECT * FROM mcp_servers ORDER BY name').map((r) => ({
      id: r.id,
      name: r.name,
      transport: r.transport as 'stdio' | 'http',
      command: r.command ?? undefined,
      args: r.args ? (JSON.parse(r.args) as string[]) : undefined,
      url: r.url ?? undefined,
      env: r.env ? (JSON.parse(r.env) as Record<string, string>) : undefined,
      enabled: !!r.enabled
    }))
  }

  addServer(config: Omit<McpServerConfig, 'id'>): McpServerConfig {
    const id = crypto.randomBytes(4).toString('hex')
    run(
      'INSERT INTO mcp_servers (id, name, transport, command, args, url, env, enabled, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
      id,
      config.name,
      config.transport,
      config.command ?? null,
      config.args ? JSON.stringify(config.args) : null,
      config.url ?? null,
      config.env ? JSON.stringify(config.env) : null,
      config.enabled ? 1 : 0,
      Date.now()
    )
    return { ...config, id }
  }

  removeServer(id: string): void {
    this.connections.get(id)?.close()
    this.connections.delete(id)
    run('DELETE FROM mcp_servers WHERE id = ?', id)
  }

  setEnabled(id: string, enabled: boolean): void {
    run('UPDATE mcp_servers SET enabled = ? WHERE id = ?', enabled ? 1 : 0, id)
    if (!enabled) {
      this.connections.get(id)?.close()
      this.connections.delete(id)
    }
  }

  /** Connect every enabled server; failures are reported, not thrown. */
  async connectAll(): Promise<{ id: string; name: string; ok: boolean; tools: number; error?: string }[]> {
    const results: { id: string; name: string; ok: boolean; tools: number; error?: string }[] = []
    for (const config of this.listConfigs()) {
      if (!config.enabled) continue
      const existing = this.connections.get(config.id)
      if (existing?.connected) {
        results.push({ id: config.id, name: config.name, ok: true, tools: existing.tools.length })
        continue
      }
      const conn = new McpConnection(config)
      try {
        await conn.connect()
        this.connections.set(config.id, conn)
        results.push({ id: config.id, name: config.name, ok: true, tools: conn.tools.length })
      } catch (err) {
        results.push({
          id: config.id,
          name: config.name,
          ok: false,
          tools: 0,
          error: err instanceof Error ? err.message : String(err)
        })
      }
    }
    return results
  }

  status(): { id: string; name: string; connected: boolean; tools: string[]; error: string | null }[] {
    return this.listConfigs().map((c) => {
      const conn = this.connections.get(c.id)
      return {
        id: c.id,
        name: c.name,
        connected: conn?.connected ?? false,
        tools: conn?.tools.map((t) => t.name) ?? [],
        error: conn?.lastError ?? null
      }
    })
  }

  /**
   * MCP tools joining the agent's catalog.
   * Tier comes from the server's own annotations where present; anything not explicitly
   * marked read-only is treated as write, so an unannotated server errs toward prompting.
   */
  toolDefinitions(): ToolDefinition[] {
    const defs: ToolDefinition[] = []
    for (const [id, conn] of this.connections) {
      if (!conn.connected) continue
      for (const t of conn.tools) {
        const tier: ToolTier = t.annotations?.readOnlyHint ? 'read' : 'write'
        defs.push({
          name: this.qualify(id, t.name),
          description: `[${conn.config.name}] ${t.description ?? t.name}`,
          tier,
          parameters: t.inputSchema ?? { type: 'object', properties: {} }
        })
      }
    }
    return defs
  }

  /** Resolve a qualified name back to a runnable tool. */
  asTool(name: string): Tool | undefined {
    const parsed = this.parse(name)
    if (!parsed) return undefined
    const conn = this.connections.get(parsed.serverId)
    if (!conn?.connected) return undefined
    const spec = conn.tools.find((t) => t.name === parsed.tool)
    if (!spec) return undefined

    return {
      name,
      description: spec.description ?? spec.name,
      tier: spec.annotations?.readOnlyHint ? 'read' : 'write',
      parameters: spec.inputSchema ?? { type: 'object', properties: {} },
      run: async (args: Record<string, unknown>, _ctx: ToolContext) => conn.callTool(parsed.tool, args)
    }
  }

  closeAll(): void {
    for (const [, conn] of this.connections) conn.close()
    this.connections.clear()
  }
}

export const mcpManager = new McpManager()
