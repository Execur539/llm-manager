#!/usr/bin/env node
/**
 * A minimal MCP server over stdio, for exercising the client without a third-party dependency.
 *
 * Speaks JSON-RPC 2.0 framed as newline-delimited JSON, implements initialize / tools/list /
 * tools/call, and deliberately includes the awkward cases a real server produces:
 *   - a tool whose name collides with a built-in ("read_file"), to prove namespacing works
 *   - a tool that returns an error result rather than throwing
 *   - a tool that returns multiple content blocks
 *
 * Directives via argv let a scenario ask for degenerate behaviour:
 *   --no-tools       advertise an empty tool list
 *   --slow           delay every response by 400ms
 *   --die-on-call    exit the process when a tool is called
 */

import readline from 'node:readline'

const flags = new Set(process.argv.slice(2))
const delay = flags.has('--slow') ? 400 : 0

const TOOLS = flags.has('--no-tools')
  ? []
  : [
      {
        name: 'echo',
        description: 'Return the text you pass in.',
        inputSchema: {
          type: 'object',
          properties: { text: { type: 'string', description: 'text to echo' } },
          required: ['text']
        }
      },
      {
        // Deliberately collides with a built-in tool name.
        name: 'read_file',
        description: 'A server-side read that must not shadow the built-in read_file.',
        inputSchema: {
          type: 'object',
          properties: { path: { type: 'string' } },
          required: ['path']
        }
      },
      {
        name: 'always_fails',
        description: 'Always returns an error result.',
        inputSchema: { type: 'object', properties: {} }
      }
    ]

function send(msg) {
  const write = () => process.stdout.write(`${JSON.stringify(msg)}\n`)
  if (delay) setTimeout(write, delay)
  else write()
}

function result(id, value) {
  send({ jsonrpc: '2.0', id, result: value })
}

const rl = readline.createInterface({ input: process.stdin })

rl.on('line', (line) => {
  if (!line.trim()) return
  let msg
  try {
    msg = JSON.parse(line)
  } catch {
    return
  }

  const { id, method, params } = msg

  switch (method) {
    case 'initialize':
      return result(id, {
        protocolVersion: '2024-11-05',
        capabilities: { tools: {} },
        serverInfo: { name: 'mock-mcp', version: '1.0.0' }
      })

    case 'notifications/initialized':
      return // a notification: no reply

    case 'tools/list':
      return result(id, { tools: TOOLS })

    case 'tools/call': {
      const name = params?.name
      const args = params?.arguments ?? {}

      if (flags.has('--die-on-call')) process.exit(1)

      if (name === 'echo') {
        return result(id, { content: [{ type: 'text', text: `echo: ${args.text ?? ''}` }] })
      }
      if (name === 'read_file') {
        return result(id, {
          content: [
            { type: 'text', text: `mcp read of ${args.path ?? '(none)'}` },
            { type: 'text', text: 'second block' }
          ]
        })
      }
      if (name === 'always_fails') {
        // An error result, not a transport error — the client must surface it as a tool failure.
        return result(id, { isError: true, content: [{ type: 'text', text: 'this tool always fails' }] })
      }
      return send({ jsonrpc: '2.0', id, error: { code: -32601, message: `unknown tool: ${name}` } })
    }

    default:
      if (id !== undefined) {
        send({ jsonrpc: '2.0', id, error: { code: -32601, message: `unknown method: ${method}` } })
      }
  }
})
