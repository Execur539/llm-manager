/**
 * Agent primitives: sub-agent delegation, task lists, and persistent memory.
 *
 * Sub-agents are sequential and depth-limited. They exist mainly to keep the parent's context
 * clean: a subtask that would burn 30k tokens of tool output returns a paragraph instead.
 */

import crypto from 'node:crypto'
import type { Tool } from './base'
import { schema, str, bool } from './base'
import { addMemory, allMemory, deleteMemory, searchMemory, updateMemory } from '../memory'
import { all, run } from '../../storage/db'

export interface AgentToolDeps {
  spawnSubAgent: (prompt: string, cwd?: string) => Promise<string>
  canSpawn: () => boolean
  /** Put a question to the user and block until they answer. */
  askUser: (question: string, options: string[]) => Promise<string>
}

interface TaskRow {
  id: string
  chat_id: string
  text: string
  done: number
  ord: number
}

export function makeAgentTools(deps: AgentToolDeps): Tool[] {
  const delegate: Tool = {
    name: 'delegate',
    description:
      'Hand a self-contained subtask to a sub-agent and get back its result. Use this for work ' +
      'that would produce a lot of intermediate tool output you do not need to see, such as ' +
      '"find every place X is configured" or "figure out why the build fails". The sub-agent ' +
      'shares your tools and permissions but has its own context. It runs to completion before ' +
      'you continue.',
    tier: 'read',
    parameters: schema(
      {
        task: str('A complete, self-contained description of the subtask. The sub-agent cannot see your conversation.'),
        cwd: str('Working directory for the sub-agent (defaults to yours)')
      },
      ['task']
    ),
    async run(args) {
      if (!deps.canSpawn()) {
        throw new Error('Sub-agent depth limit reached; do this work directly instead of delegating further.')
      }
      const result = await deps.spawnSubAgent(String(args.task), args.cwd ? String(args.cwd) : undefined)
      return `[sub-agent result]\n${result}`
    }
  }

  const addTask: Tool = {
    name: 'add_tasks',
    description:
      'Record a list of tasks you intend to work through. Keeps multi-step work visible to the ' +
      'user and stops you losing track on long jobs.',
    tier: 'read',
    parameters: schema(
      {
        session_id: str('Current session id'),
        tasks: { type: 'array', items: { type: 'string' }, description: 'Task descriptions, in order' }
      },
      ['session_id', 'tasks']
    ),
    async run(args) {
      const tasks = (args.tasks as string[]) ?? []
      const sessionId = String(args.session_id)
      const existing = all<TaskRow>('SELECT * FROM tasks WHERE chat_id = ?', sessionId).length
      tasks.forEach((t, i) => {
        run(
          'INSERT INTO tasks (id, chat_id, text, done, ord) VALUES (?, ?, ?, 0, ?)',
          crypto.randomBytes(6).toString('hex'),
          sessionId,
          t,
          existing + i
        )
      })
      return `Added ${tasks.length} task(s).`
    }
  }

  const completeTask: Tool = {
    name: 'complete_task',
    description: 'Mark a task done by its text or id.',
    tier: 'read',
    parameters: schema({ session_id: str('Session id'), task: str('Task text or id') }, ['session_id', 'task']),
    async run(args) {
      const sessionId = String(args.session_id)
      const needle = String(args.task)
      run('UPDATE tasks SET done = 1 WHERE chat_id = ? AND (id = ? OR text = ?)', sessionId, needle, needle)
      const remaining = all<TaskRow>('SELECT * FROM tasks WHERE chat_id = ? AND done = 0', sessionId)
      return remaining.length
        ? `Done. Remaining:\n${remaining.map((t) => `- ${t.text}`).join('\n')}`
        : 'Done. All tasks complete.'
    }
  }

  const listTasks: Tool = {
    name: 'list_tasks',
    description: 'List the current task list with completion state.',
    tier: 'read',
    parameters: schema({ session_id: str('Session id') }, ['session_id']),
    async run(args) {
      const tasks = all<TaskRow>('SELECT * FROM tasks WHERE chat_id = ? ORDER BY ord', String(args.session_id))
      if (!tasks.length) return 'No tasks recorded.'
      return tasks.map((t) => `[${t.done ? 'x' : ' '}] ${t.text}`).join('\n')
    }
  }

  const remember: Tool = {
    name: 'remember',
    description:
      'Save a durable fact for future sessions — a user preference, a project convention, a ' +
      'command that works. Do not store secrets, and do not store things that are only true ' +
      'for the current conversation.',
    tier: 'read',
    parameters: schema({ text: str('The fact to remember, in one sentence') }, ['text']),
    async run(args) {
      const entry = addMemory(String(args.text))
      return `Remembered: ${entry.text}`
    }
  }

  const recall: Tool = {
    name: 'recall',
    description: 'Search your saved memories.',
    tier: 'read',
    parameters: schema({ query: str('Search text; omit to list everything') }),
    async run(args) {
      const results = args.query ? searchMemory(String(args.query)) : allMemory()
      if (!results.length) return 'No memories matched.'
      return results.map((m) => `${m.id}: ${m.text}`).join('\n')
    }
  }

  const forget: Tool = {
    name: 'forget',
    description: 'Delete or replace a saved memory by id.',
    tier: 'read',
    parameters: schema(
      { id: str('Memory id'), replace_with: str('New text; omit to delete'), confirm: bool('Confirm deletion') },
      ['id']
    ),
    async run(args) {
      if (args.replace_with) {
        updateMemory(String(args.id), String(args.replace_with))
        return `Updated memory ${args.id}.`
      }
      /*
       * `confirm` is checked rather than merely advertised.
       *
       * The parameter was declared and then ignored, so a model that read the schema, decided a
       * deletion needed confirming and left it unset had the memory deleted anyway — and this is
       * a read-tier tool, so nothing else asks. Deletion is the one irreversible thing here.
       */
      // Accepts the string too: a model whose template stringifies its arguments should not have
      // its explicit confirmation read as a refusal.
      if (args.confirm !== true && args.confirm !== 'true') {
        return `Not deleted. Call forget again with confirm=true to remove memory ${args.id}, or pass replace_with to edit it instead.`
      }
      deleteMemory(String(args.id))
      return `Deleted memory ${args.id}.`
    }
  }

  /**
   * Ask the user something, mid-turn.
   *
   * Read tier: asking changes nothing. The turn blocks on the answer, which is the whole value —
   * an agent that hits a genuine ambiguity previously had to either guess and possibly do the
   * wrong work, or stop and raise it in its final reply, which ends the turn and loses all its
   * context. Deliberately narrow in its description, because a model that asks about everything
   * is worse than one that asks about nothing.
   */
  const ask: Tool = {
    name: 'ask_user',
    description:
      'Ask the user a question and wait for their answer. Use this only when you genuinely cannot ' +
      'proceed without knowing — a choice between real alternatives, a missing detail only they ' +
      'have, or an ambiguous instruction where the readings lead to different work. Do not use it ' +
      'to confirm something you can determine yourself, to ask permission (writes and commands ' +
      'already prompt), or to narrate progress. Offer options when the choice is between a few ' +
      'known alternatives; they can always type something else.',
    tier: 'read',
    parameters: schema(
      {
        question: str('The question, in one or two plain sentences'),
        options: { type: 'array', items: { type: 'string' }, description: 'Suggested answers, if the choice is between a few' }
      },
      ['question']
    ),
    async run(args) {
      const question = String(args.question ?? '').trim()
      if (!question) throw new Error('A question is required.')
      const options = ((args.options as string[] | undefined) ?? []).map(String).filter(Boolean).slice(0, 6)
      const answer = await deps.askUser(question, options)
      return `The user answered: ${answer}`
    }
  }

  return [delegate, addTask, completeTask, listTasks, remember, recall, forget, ask]
}
