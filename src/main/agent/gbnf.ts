/**
 * JSON Schema -> GBNF grammar compiler.
 *
 * llama.cpp enforces a GBNF grammar at sampling time, which is what lets a small local model
 * emit *structurally valid* tool calls even when it was never trained for tool use. This is
 * the reliability decision from Round 13: constrain the shape, accept that argument-level
 * mistakes still happen, and let errors flow back into the conversation.
 *
 * Note the deliberate limit: this guarantees the call parses and has the right field types.
 * It does not guarantee the model picked the *right* tool or sensible values.
 */

interface JsonSchema {
  type?: string | string[]
  properties?: Record<string, JsonSchema>
  required?: string[]
  items?: JsonSchema
  enum?: (string | number | boolean)[]
  description?: string
  [k: string]: unknown
}

const PRIMITIVES = `
ws          ::= [ \\t\\n]*
string      ::= "\\"" char* "\\"" ws
char        ::= [^"\\\\\\x00-\\x1F] | "\\\\" (["\\\\/bfnrt] | "u" hex hex hex hex)
hex         ::= [0-9a-fA-F]
number      ::= "-"? int frac? exp? ws
int         ::= "0" | [1-9] [0-9]*
frac        ::= "." [0-9]+
exp         ::= [eE] [-+]? [0-9]+
boolean     ::= ("true" | "false") ws
null        ::= "null" ws
value       ::= string | number | boolean | null | array | object
array       ::= "[" ws (value ("," ws value)*)? "]" ws
object      ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
`.trim()

let ruleCounter = 0

function freshName(base: string): string {
  ruleCounter += 1
  return `${base}-${ruleCounter}`
}

function literal(s: string): string {
  return `"${s.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
}

/** Compile one schema node into a rule body, appending any helper rules it needs. */
function compileNode(schema: JsonSchema, rules: string[]): string {
  if (schema.enum && schema.enum.length) {
    const options = schema.enum
      .map((v) => (typeof v === 'string' ? `"\\"" ${literal(v)} "\\""` : literal(String(v))))
      .join(' | ')
    const name = freshName('enum')
    rules.push(`${name} ::= (${options}) ws`)
    return name
  }

  const type = Array.isArray(schema.type) ? schema.type[0] : schema.type

  switch (type) {
    case 'string':
      return 'string'
    case 'number':
    case 'integer':
      return 'number'
    case 'boolean':
      return 'boolean'
    case 'null':
      return 'null'
    case 'array': {
      const item = schema.items ? compileNode(schema.items, rules) : 'value'
      const name = freshName('arr')
      rules.push(`${name} ::= "[" ws (${item} ("," ws ${item})*)? "]" ws`)
      return name
    }
    case 'object': {
      const props = schema.properties ?? {}
      const keys = Object.keys(props)
      if (!keys.length) return 'object'

      const required = new Set(schema.required ?? [])
      const requiredKeys = keys.filter((k) => required.has(k))
      const optionalKeys = keys.filter((k) => !required.has(k))

      // Compiled once per key: compileNode appends helper rules, so calling it twice while
      // building two branches would emit the same helper twice under different names.
      const member = (k: string): string => `"\\"${k}\\"" ws ":" ws ${compileNode(props[k], rules)}`
      const requiredMembers = requiredKeys.map(member)
      const optionalMembers = optionalKeys.map(member)

      const pieces: string[] = []
      requiredMembers.forEach((m, i) => pieces.push(i === 0 ? m : `"," ws ${m}`))

      if (optionalMembers.length && requiredMembers.length) {
        // Something mandatory precedes them, so a leading comma is always correct and each
        // optional can be skipped independently.
        for (const m of optionalMembers) pieces.push(`("," ws ${m})?`)
      } else if (optionalMembers.length) {
        /*
         * All-optional objects need alternation, not a row of skippable groups.
         *
         * The old form emitted `(m1)? ("," ws m2)?`, which has no way to express "just m2": the
         * comma belongs to m2's group, so emitting it alone produces `{, "m2": …}` — not JSON —
         * and the valid `{"m2": …}` is refused by the grammar. Every tool whose arguments are
         * all optional was affected, and `browser_click` most visibly: its own code offers
         * selector *or* text, and the grammar made text-only unreachable.
         *
         * One alternative per starting member, each followed by the members after it as
         * skippable comma groups. The count is the number of optional keys, which is small.
         */
        const alternatives = optionalMembers.map((m, i) =>
          [m, ...optionalMembers.slice(i + 1).map((rest) => `("," ws ${rest})?`)].join(' ')
        )
        const tail = freshName('opt')
        rules.push(`${tail} ::= ${alternatives.join(' | ')}`)
        pieces.push(`${tail}?`)
      }

      const name = freshName('obj')
      rules.push(`${name} ::= "{" ws ${pieces.join(' ')} "}" ws`)
      return name
    }
    default:
      return 'value'
  }
}

/**
 * Build a grammar that forces the model to emit exactly one tool call of the form
 * `{"name": "<one of the tools>", "arguments": { ...that tool's schema... }}`.
 */
export function toolCallGrammar(tools: { name: string; parameters: Record<string, unknown> }[]): string {
  ruleCounter = 0
  const rules: string[] = []
  const branches: string[] = []

  for (const tool of tools) {
    const argsRule = compileNode(tool.parameters as JsonSchema, rules)
    const name = freshName('call')
    rules.push(
      `${name} ::= "{" ws "\\"name\\"" ws ":" ws "\\"${tool.name}\\"" ws "," ws "\\"arguments\\"" ws ":" ws ${argsRule} "}" ws`
    )
    branches.push(name)
  }

  const root = branches.length ? branches.join(' | ') : 'object'
  return [`root ::= ${root}`, ...rules, PRIMITIVES].join('\n')
}

/** Grammar for a plain JSON object matching a single schema — used for structured output. */
export function schemaGrammar(schema: Record<string, unknown>): string {
  ruleCounter = 0
  const rules: string[] = []
  const rootRule = compileNode(schema as JsonSchema, rules)
  return [`root ::= ${rootRule}`, ...rules, PRIMITIVES].join('\n')
}
