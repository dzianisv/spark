#!/usr/bin/env bun
// spark.ts — single-file AI coding agent using Ollama
// Zero external dependencies. Run: bun spark.ts

import { readdir, stat, readFile, writeFile, mkdir, unlink } from "node:fs/promises"
import { join, resolve, dirname, basename, relative } from "node:path"
import { homedir } from "node:os"
import { spawn } from "node:child_process"
import { createInterface, emitKeypressEvents } from "node:readline"
import { Transform } from "node:stream"

// ── Types ────────────────────────────────────────────────────────────────────

// Tool call ID correlation across providers:
//
// OpenAI Chat Completions API:
//   - Assistant response: tool_calls[].id (e.g. "call_12345xyz")
//   - Tool result message: { role: "tool", tool_call_id: "<id>", content: "..." }
//   - tool_call_id is REQUIRED for correlation
//
// Ollama native /api/chat:
//   - Assistant response: tool_calls[].id MAY be present
//   - Tool result message: { role: "tool", content: "...", tool_name: "func_name" }
//   - Ollama uses tool_name for correlation, tool_call_id is ignored
//
// Ollama OpenAI-compatible /v1/chat/completions:
//   - Follows OpenAI format, includes tool_calls[].id
//   - Tool result: { role: "tool", tool_call_id: "<id>", content: "..." }
//
// Vercel AI SDK (@ai-sdk/openai-compatible):
//   - Non-streaming: toolCall.id ?? generateId() (synthetic 16-char random if missing)
//   - Streaming: REQUIRES id on first delta chunk (throws InvalidResponseDataError if null)
//   - Sends results as: { role: "tool", tool_call_id: "<id>", content: "..." }
//
// Strategy: we send BOTH tool_name (for Ollama native) and tool_call_id (for OpenAI compat).
// Extra fields are ignored by each provider, so this is safe for both endpoints.
export interface Message {
  role: "system" | "user" | "assistant" | "tool"
  content: string
  tool_calls?: ToolCall[]
  thinking?: string
  tool_name?: string
  tool_call_id?: string
}

interface ToolCall {
  id: string
  function: { name: string; arguments: Record<string, unknown> }
}

interface ToolDef {
  type: "function"
  function: {
    name: string
    description: string
    parameters: {
      type: "object"
      properties: Record<string, unknown>
      required: string[]
    }
  }
}

interface Tool {
  definition: ToolDef
  execute(args: Record<string, unknown>): Promise<string>
}

// ── Ollama Client ────────────────────────────────────────────────────────────

const getOllamaUrl = () => process.env.OLLAMA_URL ?? "http://localhost:11434"
// Cache stores a Promise so concurrent cold calls for the same model share one /api/show request
const modelCapabilityCache = new Map<string, Promise<string[]>>()

async function fetchCapabilities(model: string): Promise<string[]> {
  try {
    const res = await fetch(`${getOllamaUrl()}/api/show`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: model }),
    })
    if (!res.ok) return []
    const data = (await res.json()) as { capabilities?: string[] }
    return data.capabilities ?? []
  } catch { return [] }
}

function ollamaCapabilities(model: string): Promise<string[]> {
  let p = modelCapabilityCache.get(model)
  if (!p) {
    p = fetchCapabilities(model)
    // only cache non-empty results — missing capabilities field means unknown, not unsupported
    p = p.then(caps => { if (caps.length > 0) modelCapabilityCache.set(model, Promise.resolve(caps)); return caps })
  }
  return p
}

interface StreamCallbacks {
  onThinking?: (chunk: string) => void
  onContent?: (chunk: string) => void
}

async function ollamaChatRaw(
  model: string,
  messages: Message[],
  tools: ToolDef[],
  callbacks: StreamCallbacks | undefined,
  format: unknown,
  think: boolean,
  signal?: AbortSignal,
): Promise<Message> {
  const body: Record<string, unknown> = { model, messages, tools, stream: true }
  if (think) body.think = true
  if (format) { body.format = format; body.options = { temperature: 0 } }
  const res = await fetch(`${getOllamaUrl()}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Ollama error ${res.status}: ${text}`)
  }

  const reader = res.body?.getReader()
  if (!reader) throw new Error("No response body")

  const decoder = new TextDecoder()
  let content = ""
  let thinking = ""
  let toolCalls: ToolCall[] = []
  let buf = ""

  while (true) {
    if (signal?.aborted) break
    const { done, value } = await reader.read()
    if (done) break

    buf += decoder.decode(value, { stream: true })
    const lines = buf.split("\n")
    buf = lines.pop() ?? "" // keep incomplete last line

    for (const line of lines) {
      if (!line.trim()) continue
      let chunk: { message?: { content?: string; thinking?: string; tool_calls?: ToolCall[] }; done?: boolean }
      try { chunk = JSON.parse(line) } catch { continue }

      const msg = chunk.message
      if (!msg) continue

      if (msg.thinking) {
        thinking += msg.thinking
        callbacks?.onThinking?.(msg.thinking)
      }
      if (msg.content) {
        content += msg.content
        callbacks?.onContent?.(msg.content)
      }
      if (msg.tool_calls?.length) {
        toolCalls.push(...msg.tool_calls)
      }
    }
  }

  // parse leftover buffer
  if (buf.trim()) {
    try {
      const chunk = JSON.parse(buf) as { message?: { content?: string; thinking?: string; tool_calls?: ToolCall[] } }
      if (chunk.message?.thinking) {
        thinking += chunk.message.thinking
        callbacks?.onThinking?.(chunk.message.thinking)
      }
      if (chunk.message?.content) {
        content += chunk.message.content
        callbacks?.onContent?.(chunk.message.content)
      }
      if (chunk.message?.tool_calls?.length) {
        toolCalls.push(...chunk.message.tool_calls)
      }
    } catch {}
  }

  const result: Message = { role: "assistant", content }
  if (thinking) result.thinking = thinking
  if (toolCalls.length) result.tool_calls = toolCalls
  return result
}

export async function ollamaChat(
  model: string,
  messages: Message[],
  tools: ToolDef[],
  callbacks?: StreamCallbacks,
  format?: unknown,
  signal?: AbortSignal,
  forceThink?: boolean,
): Promise<Message> {
  const caps = await ollamaCapabilities(model)
  const think = forceThink !== undefined
    ? (forceThink && caps.includes("thinking"))
    : (!format && caps.includes("thinking"))
  try {
    return await ollamaChatRaw(model, messages, tools, callbacks, format, think, signal)
  } catch (e) {
    // If the model reported thinking support but rejects think:true, evict and retry once
    if (think && String(e).includes("400")) {
      modelCapabilityCache.delete(model)
      return ollamaChatRaw(model, messages, tools, callbacks, format, false, signal)
    }
    throw e
  }
}

async function ollamaModels(): Promise<string[]> {
  const res = await fetch(`${getOllamaUrl()}/api/tags`)
  if (!res.ok) return []
  const data = (await res.json()) as { models: { name: string }[] }
  return data.models.map((m) => m.name)
}

// Rank models by coding capability — higher score = better pick
function modelScore(name: string): number {
  const n = name.toLowerCase()
  // coding-specific models — best
  if (n.includes("codellama") || n.includes("deepseek-coder") || n.includes("codegemma") || n.includes("starcoder"))
    return 1000
  // strong general models known for good tool use / coding
  if (n.includes("qwen3") || n.includes("qwen2.5-coder") || n.includes("llama3") || n.includes("gemma2") || n.includes("mistral") || n.includes("command-r"))
    return 500
  if (n.includes("qwen"))
    return 400
  if (n.includes("phi") || n.includes("solar"))
    return 300
  // tiny models — last resort
  if (n.includes("smollm") || n.includes("tinyllama"))
    return 10
  // everything else — middle tier
  return 200
}

// Extract param size from model name (e.g. "qwen3:14b" → 14, "qwen3:0.6b" → 0.6)
function modelParamSize(name: string): number {
  const m = name.match(/:(\d+\.?\d*)b/i)
  if (m) return parseFloat(m[1])
  // Common suffixes like ":latest" — treat as unknown mid-size
  return 7
}

function pickBestModel(models: string[]): string {
  // Sort by: 1) model family score (desc), 2) param size (desc, bigger = smarter)
  return [...models].sort((a, b) => {
    const scoreDiff = modelScore(b) - modelScore(a)
    if (scoreDiff !== 0) return scoreDiff
    return modelParamSize(b) - modelParamSize(a)
  })[0]
}

// ── Context Compaction ───────────────────────────────────────────────────────

export const CHARS_PER_TOKEN = 4
export function estimateTokens(messages: Message[]): number {
  let chars = 0
  for (const m of messages) {
    chars += m.content?.length ?? 0
    if (m.tool_calls) for (const c of m.tool_calls) chars += JSON.stringify(c.function).length
  }
  return Math.round(chars / CHARS_PER_TOKEN)
}

export function splitTurns(body: Message[]): Message[][] {
  if (body.length === 0) return []
  const turns: Message[][] = []
  let current: Message[] = []

  for (const msg of body) {
    if (msg.role === "user" && current.length > 0) {
      turns.push(current)
      current = []
    }
    current.push(msg)
  }
  if (current.length > 0) turns.push(current)
  return turns
}

export const TAIL_TURNS = 2
export const COMPACT_THRESHOLD = Number(process.env.SPARK_COMPACT_THRESHOLD) || 32000

const SUMMARY_SYSTEM =
  "You are a context summarization assistant for a coding agent session. Summarize ONLY the conversation history given. Do not answer or continue the conversation. Do not mention that you are summarizing. Preserve exact file paths, commands, code identifiers, and error strings verbatim. Be terse — bullets, not prose."

const SUMMARY_INSTRUCTION =
  "Summarize the conversation above into EXACTLY this markdown template. Keep every section header even if its content is empty. Use terse bullets.\n\n## Goal\n## Constraints & Preferences\n## Progress\n### Done\n### In Progress\n### Blocked\n## Key Decisions\n## Next Steps\n## Critical Context\n## Relevant Files"

export async function compactMessages(model: string, messages: Message[], signal?: AbortSignal): Promise<boolean> {
  const body = messages.slice(1)
  const turns = splitTurns(body)
  if (turns.length <= TAIL_TURNS) return false

  const tailTurns = turns.slice(-TAIL_TURNS)
  const headTurns = turns.slice(0, turns.length - TAIL_TURNS)
  const head = headTurns.flat()
  const tail = tailTurns.flat()

  // Render head into plain-text transcript
  const lines: string[] = []
  for (const msg of head) {
    const content = (msg.content ?? "").slice(0, 2000)
    if (msg.role === "tool") {
      const label = msg.tool_name ? `TOOL(${msg.tool_name})` : "TOOL"
      lines.push(`${label}: ${content}`)
    } else {
      lines.push(`${msg.role.toUpperCase()}: ${content}`)
      if (msg.tool_calls) {
        for (const c of msg.tool_calls) {
          const args = JSON.stringify(c.function.arguments).slice(0, 2000)
          lines.push(`[called tool: ${c.function.name}(${args})]`)
        }
      }
    }
  }
  const transcript = lines.join("\n")

  const reply = await ollamaChat(
    model,
    [
      { role: "system", content: SUMMARY_SYSTEM },
      { role: "user", content: transcript + "\n\n" + SUMMARY_INSTRUCTION },
    ],
    [],
    undefined,
    undefined,
    signal,
  )
  const summary = reply.content

  messages.splice(
    1,
    messages.length - 1,
    { role: "user", content: "[Earlier conversation compacted to summary]\n\n" + summary },
    ...tail,
  )
  return true
}

// ── Utilities ────────────────────────────────────────────────────────────────

function truncateOutput(output: string, max = 30_000): string {
  if (output.length <= max) return output
  const half = Math.floor(max / 2)
  return output.slice(0, half) + `\n\n...(${output.length - max} characters omitted)...\n\n` + output.slice(-half)
}

function lineTrimmedMatch(content: string, find: string): { start: number; end: number } | null {
  const contentLines = content.split("\n")
  const findLines = find.split("\n")
  // Remove trailing empty line from find if present
  if (findLines[findLines.length - 1] === "") findLines.pop()
  for (let i = 0; i <= contentLines.length - findLines.length; i++) {
    let matches = true
    for (let j = 0; j < findLines.length; j++) {
      if (contentLines[i + j].trim() !== findLines[j].trim()) { matches = false; break }
    }
    if (matches) {
      const before = contentLines.slice(0, i).join("\n")
      const start = i > 0 ? before.length + 1 : 0
      const matchedBlock = contentLines.slice(i, i + findLines.length).join("\n")
      return { start, end: start + matchedBlock.length }
    }
  }
  return null
}

// ── Tool Implementations ────────────────────────────────────────────────────

function makeReadFile(): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "ReadFile",
        description:
          "Read a file from disk. Returns numbered lines. For directories, lists entries. Use offset/limit for large files.",
        parameters: {
          type: "object",
          properties: {
            filePath: { type: "string", description: "Absolute or relative path" },
            offset: { type: "number", description: "1-indexed line to start from (default: 1)" },
            limit: { type: "number", description: "Max lines to return (default: 2000)" },
          },
          required: ["filePath"],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      const filePath = resolve(String(args.filePath))
      const offset = Math.max(1, Number(args.offset ?? 1))
      const limit = Math.max(1, Number(args.limit ?? 2000))

      const s = await stat(filePath).catch(() => null)
      if (!s) return `Error: file not found: ${filePath}`

      if (s.isDirectory()) {
        const entries = await readdir(filePath)
        return entries
          .map((e) => e)
          .sort()
          .join("\n")
      }

      const content = await readFile(filePath, "utf-8")
      const lines = content.split("\n")
      const total = lines.length
      if (offset > total) return `Error: offset ${offset} exceeds file length (${total} lines)`
      const slice = lines.slice(offset - 1, offset - 1 + limit)
      const numbered = slice.map((line, i) => {
        const num = offset + i
        const truncated = line.length > 2000 ? line.slice(0, 2000) + "... (truncated)" : line
        return `${num}: ${truncated}`
      })

      const shown = slice.length
      const header = shown < total ? `(Showing lines ${offset}-${offset + shown - 1} of ${total})` : ""
      return truncateOutput(numbered.join("\n") + (header ? `\n${header}` : ""))
    },
  }
}

async function checkedWrite(filePath: string, content: string): Promise<string | null> {
  if (!filePath.endsWith(".ts") && !filePath.endsWith(".tsx")) return null
  const tmpPath = join(dirname(filePath), `.spark-check-${Date.now()}.ts`)
  try {
    await writeFile(tmpPath, content, "utf-8")
    const proc = Bun.spawn(["bun", "--check", tmpPath], { stdout: "pipe", stderr: "pipe" })
    await proc.exited
    await unlink(tmpPath).catch(() => {})
    if (proc.exitCode !== 0) {
      const err = await new Response(proc.stderr).text()
      return `Syntax error: ${err.trim()}`
    }
    return null
  } catch (e) {
    await unlink(tmpPath).catch(() => {})
    return `Syntax error: ${String(e)}`
  }
}

function makeWriteFile(): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "WriteFile",
        description: `Write or patch a file. Two modes:
1. Full write: provide 'content' to replace entire file.
2. Patch: provide 'oldString' and 'newString' to find-and-replace a section (like OpenCode's edit tool). The oldString must match exactly once unless replaceAll is true.`,
        parameters: {
          type: "object",
          properties: {
            filePath: { type: "string", description: "Absolute or relative path" },
            content: { type: "string", description: "Full file content (for full write mode)" },
            oldString: { type: "string", description: "Text to find (for patch mode)" },
            newString: { type: "string", description: "Replacement text (for patch mode)" },
            replaceAll: { type: "boolean", description: "Replace all occurrences (default: false)" },
          },
          required: ["filePath"],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      const filePath = resolve(String(args.filePath))
      await mkdir(dirname(filePath), { recursive: true })

      // Patch mode
      if (args.oldString !== undefined) {
        const oldStr = String(args.oldString).replace(/\r\n/g, "\n")
        const newStr = String(args.newString ?? "").replace(/\r\n/g, "\n")
        let existing = await readFile(filePath, "utf-8").catch(() => null)

        if (existing === null) {
          if (oldStr === "") {
            await writeFile(filePath, newStr, "utf-8")
            return `Created new file: ${filePath}`
          }
          return `Error: file not found for patching: ${filePath}`
        }

        // Normalize line endings
        existing = existing.replace(/\r\n/g, "\n")

        if (args.replaceAll) {
          const result = existing.replaceAll(oldStr, newStr)
          await writeFile(filePath, result, "utf-8")
          const count = (existing.split(oldStr).length - 1)
          return `Replaced ${count} occurrence(s) in ${filePath}`
        }

        // Try exact match first
        const firstIdx = existing.indexOf(oldStr)
        if (firstIdx !== -1) {
          const lastIdx = existing.lastIndexOf(oldStr)
          if (firstIdx !== lastIdx)
            return `Error: found multiple matches for oldString. Use replaceAll or provide more context to make it unique.`
          const result = existing.slice(0, firstIdx) + newStr + existing.slice(firstIdx + oldStr.length)
          await writeFile(filePath, result, "utf-8")
          const oldLines = oldStr.split("\n").length
          const newLines = newStr.split("\n").length
          return `Patched ${filePath}: replaced ${oldLines} line(s) with ${newLines} line(s)`
        }

        // Fuzzy fallback: whitespace-insensitive line matching
        const fuzzy = lineTrimmedMatch(existing, oldStr)
        if (fuzzy) {
          const result = existing.slice(0, fuzzy.start) + newStr + existing.slice(fuzzy.end)
          await writeFile(filePath, result, "utf-8")
          const oldLines = oldStr.split("\n").length
          const newLines = newStr.split("\n").length
          return `Patched ${filePath} (fuzzy match): replaced ${oldLines} line(s) with ${newLines} line(s)`
        }

        return `Error: oldString not found in ${filePath}`
      }

      // Full write mode
      const content = String(args.content ?? "")
      const syntaxErr = await checkedWrite(filePath, content)
      if (syntaxErr) return syntaxErr
      const existing = await readFile(filePath, "utf-8").catch(() => null)
      await writeFile(filePath, content, "utf-8")

      if (existing === null) return `Created new file: ${filePath} (${content.split("\n").length} lines)`
      return `Wrote ${filePath} (${content.split("\n").length} lines)`
    },
  }
}

function makeBash(): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "Bash",
        description: "Execute a shell command and return stdout+stderr. Default timeout: 120s.",
        parameters: {
          type: "object",
          properties: {
            command: { type: "string", description: "Shell command to run" },
            workdir: { type: "string", description: "Working directory (optional)" },
            timeout: { type: "number", description: "Timeout in milliseconds (default: 120000)" },
          },
          required: ["command"],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      const command = String(args.command)
      const workdir = args.workdir ? resolve(String(args.workdir)) : process.cwd()
      const timeout = Number(args.timeout ?? 120_000)

      return new Promise<string>((done) => {
        const chunks: Buffer[] = []
        const proc = spawn("sh", ["-c", command], {
          cwd: workdir,
          stdio: ["ignore", "pipe", "pipe"],
          env: { ...process.env },
        })

        proc.stdout.on("data", (d: Buffer) => chunks.push(d))
        proc.stderr.on("data", (d: Buffer) => chunks.push(d))

        const timer = setTimeout(() => {
          proc.kill("SIGTERM")
          setTimeout(() => proc.kill("SIGKILL"), 3000)
        }, timeout)

        proc.on("close", (code) => {
          clearTimeout(timer)
          const output = Buffer.concat(chunks).toString("utf-8")
          const truncated = truncateOutput(output)
          const prefix = code !== 0 ? `[exit code: ${code}]\n` : ""
          done(prefix + truncated)
        })

        proc.on("error", (err) => {
          clearTimeout(timer)
          done(`Error spawning process: ${err.message}`)
        })
      })
    },
  }
}

function makeEval(): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "Eval",
        description:
          "Evaluate JavaScript or TypeScript code inside the agent process (Bun runtime). Returns the result of the last expression, or console output. Use for quick calculations, data transforms, or testing snippets.",
        parameters: {
          type: "object",
          properties: {
            code: { type: "string", description: "JS/TS code to evaluate" },
          },
          required: ["code"],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      const code = String(args.code)
      const logs: string[] = []
      const fakeConsole = {
        log: (...a: unknown[]) => logs.push(a.map(String).join(" ")),
        error: (...a: unknown[]) => logs.push("[stderr] " + a.map(String).join(" ")),
        warn: (...a: unknown[]) => logs.push("[warn] " + a.map(String).join(" ")),
        info: (...a: unknown[]) => logs.push(a.map(String).join(" ")),
      }
      try {
        const transpiler = new Bun.Transpiler({ loader: "ts" })
        const js = transpiler.transformSync(code)
        const fn = new Function("console", `return (async () => { ${js} })()`)
        const result = await fn(fakeConsole)
        const output = logs.length ? logs.join("\n") + "\n" : ""
        const resultStr = result !== undefined ? String(result) : ""
        return (output + resultStr).trim() || "(no output)"
      } catch (err: unknown) {
        const output = logs.length ? logs.join("\n") + "\n" : ""
        return output + `Error: ${err instanceof Error ? err.message : String(err)}`
      }
    },
  }
}

function makeGlob(): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "Glob",
        description: "Fast file pattern matching. Returns matching file paths. Use for finding files by name pattern (e.g. '**/*.ts', 'src/**/*.test.*'). Respects .gitignore.",
        parameters: {
          type: "object",
          properties: {
            pattern: { type: "string", description: "Glob pattern to match files against" },
            path: { type: "string", description: "Directory to search in. Defaults to current working directory." },
          },
          required: ["pattern"],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      const dir = resolve(String(args.path ?? process.cwd()))
      const glob = new Bun.Glob(String(args.pattern))
      const results: string[] = []
      for await (const file of glob.scan({ cwd: dir, onlyFiles: true, absolute: true })) {
        results.push(file)
        if (results.length >= 200) break
      }
      if (results.length === 0) return "No files matched the pattern."
      return results.join("\n")
    },
  }
}

function makeGrep(): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "Grep",
        description: "Search file contents using regex. Returns file paths and line numbers with matching lines. Use for finding where code/text patterns appear.",
        parameters: {
          type: "object",
          properties: {
            pattern: { type: "string", description: "Regex pattern to search for in file contents" },
            path: { type: "string", description: "Directory to search in. Defaults to current working directory." },
            include: { type: "string", description: "File pattern to filter (e.g. '*.ts', '*.{ts,tsx}')" },
          },
          required: ["pattern"],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      const dir = resolve(String(args.path ?? process.cwd()))
      const rgArgs = ["--line-number", "--no-heading", "--color=never", "--max-count=100"]
      if (args.include) rgArgs.push("--glob", String(args.include))
      rgArgs.push(String(args.pattern), dir)
      try {
        const proc = Bun.spawn(["rg", ...rgArgs], { stdout: "pipe", stderr: "pipe" })
        const output = await new Response(proc.stdout).text()
        const exitCode = await proc.exited
        if (exitCode === 1) return "No matches found."
        if (!output.trim()) return "No matches found."
        return truncateOutput(output.trim())
      } catch {
        return "Error: ripgrep (rg) not found. Install with: brew install ripgrep"
      }
    },
  }
}

function makeLoadSkill(skillMap: Map<string, string>): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "LoadSkill",
        description:
          "Load a skill by name. Returns the SKILL.md content which contains instructions for that skill. Available skills are listed in the system prompt.",
        parameters: {
          type: "object",
          properties: {
            name: { type: "string", description: "Skill name to load" },
          },
          required: ["name"],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      const name = String(args.name)
      const path = skillMap.get(name)
      if (!path) {
        const available = [...skillMap.keys()].sort().join(", ")
        return `Error: skill '${name}' not found. Available: ${available}`
      }
      const content = await readFile(path, "utf-8").catch(() => `Error reading skill: ${path}`)
      return `# Skill: ${name}\n\n${content}`
    },
  }
}

function makeTask(
  model: string,
  systemPrompt: string,
  tools: Tool[],
): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "Task",
        description:
          "Spawn a sub-agent to handle a complex task. The sub-agent has access to file, shell, and eval tools but cannot spawn further sub-agents. Use for delegating independent work.",
        parameters: {
          type: "object",
          properties: {
            description: { type: "string", description: "Short task description (3-5 words)" },
            prompt: { type: "string", description: "Detailed task instructions for the sub-agent" },
            timeout: { type: "number", description: "Timeout in milliseconds (default: 300000 = 5 min)" },
          },
          required: ["description", "prompt"],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      const prompt = String(args.prompt)
      const desc = String(args.description ?? "sub-task")
      const timeout = Number(args.timeout ?? 300_000)
      process.stderr.write(`\x1b[90m[Task: ${desc}]\x1b[0m\n`)

      // Sub-agent tools: everything EXCEPT Task (no recursive spawning)
      const subTools = tools.filter(t => t.definition.function.name !== "Task")

      const subMessages: Message[] = [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt },
      ]

      const toolDefs = subTools.map((t) => t.definition)
      const toolMap = new Map(subTools.map((t) => [t.definition.function.name, t]))
      const MAX_TURNS = 20
      const deadline = Date.now() + timeout
      let lastContent = ""

      for (let turn = 0; turn < MAX_TURNS; turn++) {
        if (Date.now() > deadline) {
          return `<task_result task="${desc}">\nTask timed out after ${timeout}ms\n</task_result>`
        }

        const reply = await ollamaChat(model, subMessages, toolDefs)  // no streaming callbacks for sub-agents
        subMessages.push(reply)
        lastContent = reply.content

        if (!reply.tool_calls?.length) {
          return `<task_result task="${desc}">\n${truncateOutput(reply.content)}\n</task_result>`
        }

        for (const call of reply.tool_calls) {
          if (Date.now() > deadline) {
            return `<task_result task="${desc}">\nTask timed out after ${timeout}ms\n</task_result>`
          }
          const tool = toolMap.get(call.function.name)
          const result = tool
            ? await tool.execute(call.function.arguments)
            : `Error: unknown tool '${call.function.name}'`
          subMessages.push({ role: "tool", content: result, tool_name: call.function.name, tool_call_id: call.id })
        }
      }

      return `<task_result task="${desc}">\nReached max turns (${MAX_TURNS}). Last response:\n${truncateOutput(lastContent)}\n</task_result>`
    },
  }
}

// ── Autopilot ────────────────────────────────────────────────────────────────

export const MAX_AUTOPILOT_REFLECTIONS = 50

export const AUTOPILOT_NUDGE = `<system-reminder>
Autopilot mode remains active. You have not called autopilot_exit yet.
If you were planning, stop planning and start implementing.
You aren't done until you have fully completed the task.

Do not call autopilot_exit if:
- You have open questions — make decisions and keep working
- You hit an error — try to resolve it
- There are remaining steps — complete them first

Continue executing autonomously. Keep moving forward.
</system-reminder>`

export const AUTOPILOT_SUMMARY_PROMPT = "Autopilot completed. Briefly summarize what was accomplished."

export function makeAutopilotExit(state: { exited: boolean }): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "autopilot_exit",
        description:
          "Call this ONLY when the task is fully complete to exit autopilot mode. Do not call it if you have open questions, hit a recoverable error, or have remaining steps.",
        parameters: {
          type: "object",
          properties: {
            summary: { type: "string", description: "Optional one-line summary of what was accomplished" },
          },
          required: [],
          additionalProperties: false,
        },
      },
    },
    async execute(_args: Record<string, unknown>): Promise<string> {
      state.exited = true
      return "Autopilot exit acknowledged. Provide a brief final summary."
    },
  }
}

function makePhaseAdvance(currentPhase: { value: number }): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "PhaseAdvance",
        description: "Advance to the next phase of the repair workflow (from locate to repair).",
        parameters: {
          type: "object",
          properties: {},
          required: [],
          additionalProperties: false,
        },
      },
    },
    async execute(_args: Record<string, unknown>): Promise<string> {
      const newPhase = currentPhase.value + 1
      currentPhase.value = newPhase
      return `Phase advanced to ${newPhase}. You are now in the repair phase — proceed with WriteFile edits and RunTests verification.`
    },
  }
}

// ── Skill Scanner ────────────────────────────────────────────────────────────

async function scanSkills(): Promise<Map<string, string>> {
  const skills = new Map<string, string>()
  const dirs = [join(homedir(), ".agents", "skills"), join(process.cwd(), ".agents", "skills")]

  for (const dir of dirs) {
    const entries = await readdir(dir).catch(() => [] as string[])
    for (const entry of entries) {
      const skillPath = join(dir, entry, "SKILL.md")
      const s = await stat(skillPath).catch(() => null)
      if (s?.isFile()) {
        // Extract description from frontmatter if present
        skills.set(entry, skillPath)
      }
    }
  }

  return skills
}

async function getSkillDescriptions(skillMap: Map<string, string>): Promise<string> {
  const lines: string[] = []
  for (const [name, path] of skillMap) {
    const content = await readFile(path, "utf-8").catch(() => "")
    // Try to extract description from YAML frontmatter
    const fmMatch = content.match(/^---\s*\n([\s\S]*?)\n---/)
    let desc = ""
    if (fmMatch) {
      const descMatch = fmMatch[1].match(/description:\s*(.+)/)
      if (descMatch) desc = descMatch[1].trim()
    }
    if (!desc) {
      // Fallback: first non-heading, non-empty line
      const bodyLines = content.split("\n").filter((l) => l.trim() && !l.startsWith("#") && !l.startsWith("---"))
      desc = bodyLines[0]?.trim().slice(0, 100) ?? ""
    }
    lines.push(`- ${name}: ${desc}`)
  }
  return lines.join("\n")
}

// ── Docker Sandbox ───────────────────────────────────────────────────────────

async function spawnSandbox(): Promise<void> {
  const scriptPath = import.meta.path          // absolute path to spark.ts on host
  const scriptDir = dirname(scriptPath)
  const cwd = resolve(process.cwd())

  // Rewrite localhost/127.0.0.1 in OLLAMA_URL to Docker's host alias
  const hostOllamaUrl = process.env.OLLAMA_URL ?? "http://localhost:11434"
  const dockerOllamaUrl = hostOllamaUrl.replace(/localhost|127\.0\.0\.1/g, "host.docker.internal")

  const volumes = [`${cwd}:/workspace`]
  let containerScript: string

  if (scriptDir === cwd) {
    containerScript = `/workspace/${basename(scriptPath)}`
  } else {
    // spark.ts lives outside the working directory — mount its dir separately
    volumes.push(`${scriptDir}:/spark`)
    containerScript = `/spark/${basename(scriptPath)}`
  }

  const dockerArgs = [
    "run", "--rm", "-it",
    "--add-host=host.docker.internal:host-gateway",  // Linux compat; macOS Docker Desktop has it built-in
    "-e", `OLLAMA_URL=${dockerOllamaUrl}`,
    "-e", "SPARK_SANDBOX=1",
    ...volumes.flatMap(v => ["-v", v]),
    "-w", "/workspace",
    "oven/bun:alpine",
    "bun", containerScript,
  ]

  console.log(`\x1b[90m[sandbox] docker ${dockerArgs.join(" ")}\x1b[0m\n`)

  await new Promise<void>((resolve) => {
    const proc = spawn("docker", dockerArgs, { stdio: "inherit" })
    proc.on("close", () => resolve())
    proc.on("error", (err) => {
      console.error(`\x1b[31mFailed to start Docker: ${err.message}\x1b[0m`)
      console.error(`\x1b[90mIs Docker running? Try: docker info\x1b[0m`)
      resolve()
    })
  })
}

// ── Agent Instructions Loader ────────────────────────────────────────────────

async function loadAgentInstructions(): Promise<string> {
  const parts: string[] = []

  // Global instructions first (lowest precedence)
  const globalPaths = [
    join(homedir(), ".agents", "AGENTS.md"),
    join(homedir(), ".agents", "agents.md"),
  ]
  for (const p of globalPaths) {
    const content = await readFile(p, "utf-8").catch(() => null)
    if (content) { parts.push(`# ${p}\n\n${content}`); break }
  }

  // Walk up from cwd to homedir (inclusive) looking for AGENTS.md / CLAUDE.md
  // Child dirs beat parents: we collect bottom-up then reverse before pushing.
  const candidates = ["AGENTS.md", "CLAUDE.md", "agents.md"]
  const found: string[] = []
  const home = homedir()
  let dir = process.cwd()
  // Only walk within the homedir subtree to avoid reading unrelated system configs
  if (dir.startsWith(home)) {
    while (dir !== dirname(dir)) {
      for (const name of candidates) {
        const p = join(dir, name)
        const content = await readFile(p, "utf-8").catch(() => null)
        if (content) { found.push(`# ${p}\n\n${content}`); break }
      }
      if (dir === home) break
      dir = dirname(dir)
    }
    // found is [cwd, parent, ..., home] — reverse so child instructions come last (highest precedence)
    parts.push(...found.reverse())
  }

  return parts.join("\n\n---\n\n")
}

// ── Git Context ──────────────────────────────────────────────────────────────

async function getGitContext(cwd: string): Promise<string> {
  const run = (cmd: string) =>
    new Promise<string>((res) => {
      const proc = spawn("sh", ["-c", cmd], { cwd, stdio: ["ignore", "pipe", "ignore"] }) // ignore stderr — only stdout
      const chunks: Buffer[] = []
      proc.stdout.on("data", (d: Buffer) => chunks.push(d))
      const timer = setTimeout(() => { proc.kill(); res("") }, 3000) // 3s timeout — git can hang on slow NFS/credentials
      proc.on("close", () => { clearTimeout(timer); res(Buffer.concat(chunks).toString("utf-8").trim()) })
      proc.on("error", () => { clearTimeout(timer); res("") })
    })

  const [branch, status, log] = await Promise.all([
    run("git branch --show-current 2>/dev/null"),
    run("git status --short 2>/dev/null | head -20"),
    run("git log --oneline -5 2>/dev/null"),
  ])

  if (!branch && !status && !log) return "" // not a git repo

  const parts: string[] = []
  if (branch) parts.push(`Branch: ${branch}`)
  else parts.push("Branch: (detached HEAD)")
  if (status) parts.push(`Changed files:\n${status}`)
  if (log) parts.push(`Recent commits:\n${log}`)
  return parts.join("\n")
}

// ── System Prompt Builder ────────────────────────────────────────────────────

function buildSystemPrompt(agentInstructions: string, skillList: string, model: string, gitContext: string): string {
  const cwd = process.cwd()
  const today = new Date().toLocaleDateString("en-CA") // YYYY-MM-DD
  const platform = process.platform

  return `You are spark, an interactive CLI coding agent. You solve coding tasks by reading, writing, and running code directly.

<env>
  Model: ${model}
  Working directory: ${cwd}
  Platform: ${platform}
  Today: ${today}${gitContext ? `\n${gitContext.replace(/</g, "&lt;").split("\n").map(l => `  ${l}`).join("\n")}` : ""}
</env>

## Behavior
- You have all the tools/functions to complete the task. Do not backdelegate to the user unless you are truly stuck.
- Be concise and direct. No preamble, no filler.
- When referring to code, use \`file_path:line_number\` references.
- Prefer editing existing files over creating new ones.
- If a task requires multiple independent tool calls, make them all at once.
- Verify your work — run the code, check the output, confirm it works.

## Available Skills
Use LoadSkill to read a skill's full instructions when a task matches.
${skillList || "(none)"}
${agentInstructions ? `\n## Agent Instructions\n${agentInstructions}` : ""}
`
}

// ── REPL ─────────────────────────────────────────────────────────────────────

const COLORS = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[90m",
  green: "\x1b[32m",
  cyan: "\x1b[36m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
  magenta: "\x1b[35m",
}

function printHeader(model: string, skillCount: number) {
  console.log(`${COLORS.cyan}${COLORS.bold}spark${COLORS.reset} ${COLORS.dim}— AI coding agent${COLORS.reset}`)
  console.log(`${COLORS.dim}Model:${COLORS.reset} ${model}  ${COLORS.dim}Skills:${COLORS.reset} ${skillCount}  ${COLORS.dim}Dir:${COLORS.reset} ${process.cwd()}`)
  const inSandbox = process.env.SPARK_SANDBOX === "1"
  const sandboxTag = inSandbox ? `  \x1b[33m[sandbox]\x1b[0m` : ""
  console.log(`${COLORS.dim}Commands: /models /clear /compact /goal /autopilot /quit${COLORS.reset}${sandboxTag}`)
  console.log()
}

const MODEL_FILE = join(".agents", "spark", "model")

async function saveModel(model: string): Promise<void> {
  const dir = dirname(MODEL_FILE)
  await mkdir(dir, { recursive: true }).catch(() => {})
  await writeFile(MODEL_FILE, model, "utf-8")
}

async function loadSavedModel(): Promise<string | null> {
  try {
    const saved = (await readFile(MODEL_FILE, "utf-8")).trim()
    return saved || null
  } catch {
    return null
  }
}

const GOAL_FILE = join(".agents", "spark", "goal")

async function saveGoal(goal: string): Promise<void> {
  const dir = dirname(GOAL_FILE)
  await mkdir(dir, { recursive: true }).catch(() => {})
  await writeFile(GOAL_FILE, goal, "utf-8")
}

async function loadGoal(): Promise<string | null> {
  try {
    const saved = (await readFile(GOAL_FILE, "utf-8")).trim()
    return saved || null
  } catch {
    return null
  }
}

async function clearGoal(): Promise<void> {
  try {
    await unlink(GOAL_FILE)
  } catch {}
}

const AUTOPILOT_COUNT_FILE = join(".agents", "spark", "autopilot-count")

async function loadAutopilotCount(): Promise<number> {
  try {
    return parseInt(await readFile(AUTOPILOT_COUNT_FILE, "utf-8"), 10) || 0
  } catch {
    return 0
  }
}

async function saveAutopilotCount(n: number): Promise<void> {
  const dir = dirname(AUTOPILOT_COUNT_FILE)
  await mkdir(dir, { recursive: true }).catch(() => {})
  await writeFile(AUTOPILOT_COUNT_FILE, String(n), "utf-8")
}

async function selectModel(models: string[], current: string, rl: ReturnType<typeof createInterface>): Promise<string> {
  console.log(`\n${COLORS.cyan}Available models:${COLORS.reset}`)
  models.forEach((m, i) => {
    const marker = m === current ? ` ${COLORS.green}(current)${COLORS.reset}` : ""
    console.log(`  ${COLORS.dim}${i + 1}.${COLORS.reset} ${m}${marker}`)
  })

  const answer = await new Promise<string>((res) => {
    rl.question(`${COLORS.yellow}Select model (1-${models.length}): ${COLORS.reset}`, res)
  })

  const idx = parseInt(answer, 10) - 1
  if (idx >= 0 && idx < models.length) {
    const picked = models[idx]
    await saveModel(picked)
    console.log(`${COLORS.green}Switched to ${picked}${COLORS.reset}`)
    return picked
  }
  console.log(`${COLORS.dim}Keeping ${current}${COLORS.reset}`)
  return current
}

// ── Goal Supervisor ──────────────────────────────────────────────────────────

/**
 * Injects the active goal into the system prompt as a MANDATORY completion
 * requirement with an evidence rule — mirrors Claude Code's agents-supervisor
 * buildGoalRequirementSection(). A bare "I'm done" claim without tool output
 * evidence does NOT satisfy this block.
 */
export function buildGoalBlock(goal: string): string {
  if (!goal.trim()) return ""
  return `

## GOAL (mandatory completion requirement)

MANDATORY: The following goal MUST be demonstrably met before the task is complete:

  "${goal.trim()}"

Evidence rule: a claim that this goal is met MUST be backed by evidence already
in the conversation — commands run and their output, tests executed, files created
and verified. A bare assertion does NOT count as evidence. If you believe the goal
is met, show the proof (command output, file contents, test results).`
}

/**
 * Builds an escalating supervisor feedback message based on how many checks
 * have already fired. Escalation levels mirror CC's buildEscalatingFeedback():
 *   checks 1-2: gentle nudge
 *   checks 3-4: firmer, asks for plan change
 *   checks 5-9: strong — detect planning loop, demand action
 *   checks 10+: final warning, ask for different approach
 */
export function buildSupervisorFeedback(checkCount: number, goal: string, feedback: string): string {
  const base = feedback ? `${feedback} ` : ""
  if (checkCount <= 2) {
    return `[supervisor] Goal not yet reached. ${base}Keep working toward the goal: ${goal}`
  }
  if (checkCount <= 4) {
    return `[supervisor] Still not reached after ${checkCount} checks. ${base}If you have been only reading files, start writing. Make a concrete change now. Goal: ${goal}`
  }
  if (checkCount <= 9) {
    return `[supervisor] STOP PLANNING — ${checkCount} checks have fired with no completion. ${base}Do NOT read more files. Pick one concrete action and execute it immediately. Goal: ${goal}`
  }
  return `[supervisor] WARNING: ${checkCount} supervisor cycles without completion. ${base}You must either complete the goal NOW or call autopilot_exit and explain why it cannot be done. Goal: ${goal}`
}



const VERDICT_SCHEMA = {
  type: "object",
  properties: {
    reached: { type: "boolean" },
    feedback: { type: "string" },
  },
  required: ["reached", "feedback"],
} as const

export function parseVerdict(text: string): GoalVerdict {
  try {
    const parsed = JSON.parse(text)
    return { reached: Boolean(parsed.reached), feedback: String(parsed.feedback ?? "") }
  } catch {
    return { reached: false, feedback: "" } // fail-safe: keep working rather than falsely declare done
  }
}

export async function deriveGoal(
  model: string,
  messages: Message[],
  existingGoal: string | null,
  signal?: AbortSignal,
): Promise<string> {
  const INJECTED_PREFIXES = ["[supervisor]", "[autopilot]", "[System:"]
  const userTurns = messages
    .filter(m => m.role === "user" && typeof m.content === "string")
    .filter(m => !INJECTED_PREFIXES.some(p => String(m.content).startsWith(p)))
    .slice(-6)
    .map(m => String(m.content).slice(0, 400))
    .join("\n---\n")

  const systemPrompt =
    "You are a goal synthesizer for an AI coding agent. " +
    "Produce a single precise, measurable, actionable goal in one sentence. " +
    "Describe exactly what 'done' looks like. " +
    "Reply with ONLY the goal text — no explanation, no preamble, no quotes."

  const userPrompt = existingGoal
    ? `Stated goal: ${existingGoal}\n\n${userTurns ? `Recent conversation:\n${userTurns}\n\n` : ""}Refine into a single precise, verifiable goal. What exactly does success look like?`
    : `Conversation:\n${userTurns || "(no prior messages)"}\n\nWhat is the single most important goal to accomplish? State it precisely.`

  const reply = await ollamaChat(
    model,
    [{ role: "system", content: systemPrompt }, { role: "user", content: userPrompt }],
    [], undefined, undefined, signal,
  )

  const derived = (reply.content ?? "").trim().slice(0, 300)
  return derived || existingGoal || "Complete the current task"
}

export async function checkGoal(model: string, goal: string, messages: Message[], signal?: AbortSignal): Promise<GoalVerdict> {
  const lastAssistant = [...messages].reverse().find(m => m.role === "assistant")?.content ?? ""
  // Skip injected supervisor/system messages when finding the last real user message.
  const INJECTED_PREFIXES = ["[supervisor]", "[autopilot]", "[System:"]
  const lastUser = [...messages].reverse().find(
    m => m.role === "user" &&
    typeof m.content === "string" &&
    !INJECTED_PREFIXES.some(p => (m.content as string).startsWith(p))
  )?.content ?? ""
  // Include the last few tool results as evidence for the judge — this is what
  // buildGoalBlock's evidence rule asks for: "commands run and their output".
  const recentToolOutput = messages
    .filter(m => m.role === "tool" && typeof m.content === "string")
    .slice(-3)
    .map(m => `[tool:${(m as {tool_name?: string}).tool_name ?? "?"}] ${String(m.content).slice(0, 400)}`)
    .join("\n")
  const judgeSystem = "You are a strict goal supervisor for a coding agent. You judge whether a GOAL is fully accomplished based on the conversation. Be skeptical of unverified claims."
  const evidenceSection = recentToolOutput ? `\nRecent tool output (evidence):\n${recentToolOutput}\n` : ""
  const judgeUser = `GOAL: ${goal}\n\nLast user message:\n${typeof lastUser === "string" ? lastUser : ""}${evidenceSection}\nAgent's last message:\n${typeof lastAssistant === "string" ? lastAssistant : ""}\n\nIs the GOAL fully reached? Respond with ONLY a JSON object: {"reached": true|false, "feedback": "<if not reached, one concrete next action to push the agent forward; empty string if reached>"}`
  const reply = await ollamaChat(model, [
    { role: "system", content: judgeSystem },
    { role: "user", content: judgeUser },
  ], [], undefined, VERDICT_SCHEMA, signal)
  return parseVerdict(reply.content ?? "")
}

// Bracketed paste filter: sits between process.stdin and readline.
// Intercepts ESC[200~...ESC[201~ paste sequences, encodes internal newlines
// as NUL (\x00) so readline sees one "line" per paste, then promptMultiline
// decodes NUL back to \n. Normal keystrokes pass through unchanged.
function makePasteFilter(): Transform {
  const BP_START = "\x1b[200~"
  const BP_END   = "\x1b[201~"
  let pasteBuf = ""
  let pasting  = false

  return new Transform({
    decodeStrings: false,
    transform(chunk: Buffer | string, _enc: string, cb: (err: null, data?: string) => void) {
      let s = Buffer.isBuffer(chunk) ? chunk.toString("utf8") : chunk
      let out = ""
      while (s.length > 0) {
        if (pasting) {
          const end = s.indexOf(BP_END)
          if (end === -1) { pasteBuf += s; s = "" }
          else {
            pasteBuf += s.slice(0, end)
            out += pasteBuf.replace(/\n/g, "\x00") + "\n"
            pasteBuf = ""; pasting = false
            // consume one trailing \n (Enter pressed after paste) to avoid empty submission
            const after = s.slice(end + BP_END.length)
            s = after.startsWith("\n") ? after.slice(1) : after
          }
        } else {
          const start = s.indexOf(BP_START)
          if (start === -1) { out += s; s = "" }
          else { out += s.slice(0, start); pasting = true; pasteBuf = ""; s = s.slice(start + BP_START.length) }
        }
      }
      cb(null, out || undefined)
    },
  })
}

async function main() {
  // Docker sandbox: re-spawn inside Alpine container then exit
  if (process.argv.includes("--sandbox")) {
    await spawnSandbox()
    process.exit(0)
  }

  // 1. Check Ollama
  const models = await ollamaModels()
  if (models.length === 0) {
    console.error(`${COLORS.red}Error: No Ollama models found. Is Ollama running?${COLORS.reset}`)
    console.error(`${COLORS.dim}Start it with: ollama serve${COLORS.reset}`)
    process.exit(1)
  }

  // Try saved model first, fall back to auto-pick
  const savedModel = await loadSavedModel()
  let currentModel: string
  let thinkingEnabled = false
  if (savedModel && models.includes(savedModel)) {
    currentModel = savedModel
    console.log(`${COLORS.dim}Restored saved model: ${savedModel}${COLORS.reset}`)
  } else {
    currentModel = pickBestModel(models)
  }

  // 2. Load skills, agent instructions, and git context in parallel
  const [skillMap, agentInstructions, gitContext] = await Promise.all([
    scanSkills(),
    loadAgentInstructions(),
    getGitContext(process.cwd()),
  ])
  const skillList = await getSkillDescriptions(skillMap)

  // 3. Build tools (core tools first, then system prompt, then Task uses lazy ref)
  const readFileTool = makeReadFile()
  const writeFileTool = makeWriteFile()
  const bashTool = makeBash()
  const evalTool = makeEval()
  const globTool = makeGlob()
  const grepTool = makeGrep()
  const loadSkillTool = makeLoadSkill(skillMap)
  const coreTools = [readFileTool, writeFileTool, bashTool, evalTool, globTool, grepTool, loadSkillTool]

  // 4. Build system prompt
  const systemPrompt = buildSystemPrompt(agentInstructions, skillList, currentModel, gitContext)

  // 5. Build full tool set with Task having a lazy systemPrompt reference
  const buildTools = () => {
    const taskTool = makeTask(currentModel, systemPrompt, coreTools)
    return [...coreTools, taskTool]
  }

  // 6. Init conversation
  const messages: Message[] = [{ role: "system", content: systemPrompt }]
  let goal: string | null = await loadGoal()
  if (goal) {
    messages[0].content += buildGoalBlock(goal)
  }
  let autopilot = false

  printHeader(currentModel, skillMap.size)

  // 7. REPL loop
  const isTTY = process.stdin.isTTY ?? false

  // Pipe stdin through the paste filter so multi-line pastes arrive as one submission.
  const pasteFilter = makePasteFilter()
  process.stdin.pipe(pasteFilter)
  if (isTTY) {
    process.stdout.write("\x1b[?2004h") // request bracketed paste from terminal
    process.on("exit", () => process.stdout.write("\x1b[?2004l"))
  }

  const rl = createInterface({ input: pasteFilter, output: process.stdout })
  if (isTTY) emitKeypressEvents(pasteFilter, rl)

  const prompt = () =>
    new Promise<string>((res) => {
      rl.question(`${COLORS.green}> ${COLORS.reset}`, res)
    })

  // Collect input: bare Enter submits; trailing \ continues; bracketed pastes
  // arrive as a single line with NUL-encoded newlines (decoded here to \n).
  // Slash commands and paste blocks are never continued.
  const promptMultiline = async (): Promise<string> => {
    const lines: string[] = []
    while (true) {
      const raw  = await prompt()
      const line = raw.replace(/\x00/g, "\n") // decode paste newlines
      const isPaste   = raw.includes("\x00")
      const isCommand = lines.length === 0 && line.trimStart().startsWith("/")
      if (!isPaste && !isCommand && raw.endsWith("\\")) {
        lines.push(raw.slice(0, -1))
        process.stdout.write(`${COLORS.green}... ${COLORS.reset}`)
      } else {
        lines.push(line)
        return lines.join("\n")
      }
    }
  }

  // Open $EDITOR on a temp file and return its contents.
  // Rejects if the editor exits non-zero (cancelled) or cannot be spawned.
  const openEditor = (): Promise<string> =>
    new Promise((res, rej) => {
      const tmp = join(homedir(), `.spark_edit_${Date.now()}.txt`)
      const editor = process.env.EDITOR ?? process.env.VISUAL ?? "vi"
      const child = spawn(editor, [tmp], { stdio: "inherit" })
      let errored = false
      child.on("error", (err) => { errored = true; rej(err) })
      child.on("close", (code) => {
        if (errored) return // error event already rejected
        if (code !== 0) return rej(new Error(`editor exited ${code}`))
        readFile(tmp, "utf8")
          .then((t) => unlink(tmp).catch(() => {}).then(() => res(t.trim())))
          .catch(rej)
      })
    })

  // Exit cleanly on Ctrl+C when waiting at the prompt (no agent turn active).
  const promptSigint = () => { process.stdout.write("\n"); rl.close(); process.exit(0) }

  while (true) {
    process.once("SIGINT", promptSigint)
    const input = await promptMultiline()
    process.removeListener("SIGINT", promptSigint)
    const trimmed = input.trim()
    if (!trimmed) continue

    // Handle commands
    if (trimmed === "/quit" || trimmed === "/exit") {
      console.log(`${COLORS.dim}Bye!${COLORS.reset}`)
      rl.close()
      process.exit(0)
    }

    if (trimmed === "/clear") {
      messages.length = 1 // keep system prompt
      console.log(`${COLORS.dim}Conversation cleared.${COLORS.reset}\n`)
      continue
    }

    if (trimmed === "/models") {
      const freshModels = await ollamaModels()
      currentModel = await selectModel(freshModels, currentModel, rl)
      continue
    }

    if (trimmed.startsWith("/model ")) {
      const name = trimmed.slice("/model ".length).trim()
      if (name) {
        currentModel = name
        console.log(`Model set to ${name}`)
      } else {
        console.log(`Usage: /model <name>`)
      }
      continue
    }

    if (trimmed.startsWith("/think")) {
      const arg = trimmed.slice("/think".length).trim()
      if (arg === "on" || arg === "1" || arg === "true") {
        thinkingEnabled = true
        console.log(`Thinking mode ON — model will use extended reasoning (slower)`)
      } else if (arg === "off" || arg === "0" || arg === "false" || arg === "") {
        thinkingEnabled = false
        console.log(`Thinking mode OFF`)
      } else {
        console.log(`Usage: /think on|off`)
      }
      continue
    }

    if (trimmed === "/goal") {
      console.log(goal ? `goal: ${goal}` : `(no goal set)`)
      continue
    }

    if (trimmed.startsWith("/goal ")) {
      const arg = trimmed.slice("/goal ".length).trim()
      if (arg === "clear") {
        goal = null
        await clearGoal()
        // Also strip the MANDATORY goal block from system prompt if present
        messages[0].content = messages[0].content.replace(/\n\n## GOAL \(mandatory[\s\S]*$/, "")
        console.log(`goal cleared`)
      } else {
        goal = arg
        await saveGoal(arg)
        console.log(`goal set: ${arg}`)
        // Update system prompt with MANDATORY block, replace any existing one
        messages[0].content = messages[0].content.replace(/\n\n## GOAL \(mandatory[\s\S]*$/, "")
        messages[0].content += buildGoalBlock(arg)
        messages.push({ role: "user", content: `My current goal: ${arg}\n\nThis goal is mandatory — I need you to work toward it and provide evidence when it is achieved.` })
      }
      continue
    }

    if (trimmed === "/compact") {
      const before = estimateTokens(messages)
      const compacted = await compactMessages(currentModel, messages)
      if (!compacted) {
        const turns = splitTurns(messages.slice(1))
        console.log(`${COLORS.dim}nothing to compact (${turns.length} turns)${COLORS.reset}`)
      } else {
        const after = estimateTokens(messages)
        console.log(`↯ compacted: ${before} → ${after} est. tokens (kept system + last ${TAIL_TURNS} turns)`)
      }
      continue
    }

    if (trimmed === "/autopilot") {
      console.log(autopilot ? `autopilot: ON` : `autopilot: OFF`)
      continue
    }

    let skipGenericPush = false

    if (trimmed === "/edit") {
      let edited: string
      try {
        edited = await openEditor()
      } catch {
        console.log(`${COLORS.dim}editor failed or cancelled${COLORS.reset}`)
        continue
      }
      if (!edited) { console.log(`${COLORS.dim}(empty, skipped)${COLORS.reset}`); continue }
      messages.push({ role: "user", content: edited })
      skipGenericPush = true
      // fall through to agent loop
    }

    if (trimmed.startsWith("/autopilot ")) {
      const arg = trimmed.slice("/autopilot ".length).trim()
      if (arg === "off") {
        autopilot = false
        console.log(`autopilot OFF`)
        continue
      }
      autopilot = true
      skipGenericPush = true

      // Include any inline task arg as staging context for goal derivation,
      // but don't push it as a raw message — the kick-off message below replaces it.
      const stagingMessages: Message[] = arg && arg !== "on"
        ? [...messages, { role: "user", content: arg }]
        : messages

      process.stdout.write(`${COLORS.dim}deriving goal…${COLORS.reset}\r`)
      const derived = await deriveGoal(currentModel, stagingMessages, goal)
      goal = derived
      await saveGoal(derived)

      // Inject MANDATORY goal block into system prompt
      messages[0].content = messages[0].content.replace(/\n\n## GOAL \(mandatory[\s\S]*$/, "")
      messages[0].content += buildGoalBlock(derived)

      const objectiveN = (await loadAutopilotCount()) + 1
      await saveAutopilotCount(objectiveN)

      const preview = derived.length > 80 ? derived.slice(0, 80) + "…" : derived
      process.stdout.write("                              \r")
      console.log(`copilot: ${COLORS.green}●${COLORS.reset} Started autopilot objective #${objectiveN}: ${preview}`)

      // Kick-off message anchors the agent to the refined goal from turn 1
      messages.push({
        role: "user",
        content: `[autopilot] Objective #${objectiveN}: ${derived}\nWork toward this goal autonomously. Use tools. Provide evidence (command output, file contents, test results) when done. Call autopilot_exit only when the goal is fully achieved and verified.`,
      })
      // fall through into the agent loop below by NOT continuing
    }

    // Per-turn git context refresh: inject lightweight git state before each LLM call
    let perTurnGitBlock: string | null = null
    if (!skipGenericPush) {
      const runGit = (cmd: string) =>
        new Promise<string>((res) => {
          const proc = spawn("sh", ["-c", cmd], { cwd: process.cwd(), stdio: ["ignore", "pipe", "ignore"] })
          const chunks: Buffer[] = []
          proc.stdout.on("data", (d: Buffer) => chunks.push(d))
          proc.on("close", () => res(Buffer.concat(chunks).toString("utf-8").trim()))
          proc.on("error", () => res(""))
        })
      const [diffStat, statusShort] = await Promise.all([
        runGit("git diff --stat HEAD 2>/dev/null"),
        runGit("git status --short 2>/dev/null"),
      ])
      if (diffStat || statusShort) {
        perTurnGitBlock = `[Git: ${diffStat || "(no diff)"}\n${statusShort || ""}]`.trim()
      }
    }

    // Add user message
    if (!skipGenericPush) {
      const content = perTurnGitBlock ? `${perTurnGitBlock}\n\n${trimmed}` : trimmed
      messages.push({ role: "user", content })
    }

    // Agent loop: call model, handle tool calls, repeat until text response
    const tools = buildTools()
    const autopilotState = { exited: false, summarized: false }
    if (autopilot) tools.push(makeAutopilotExit(autopilotState))
    let reflections = 0
    const toolDefsForCall = tools.map((t) => t.definition)
    const toolMap = new Map(tools.map((t) => [t.definition.function.name, t]))
    const MAX_TOOL_ROUNDS = autopilot ? 200 : 30
    // Supervisor check cap: high for autopilot (user expects long run), small for
    // interactive turns (prevents runaway if judge keeps returning reached:false).
    const MAX_GOAL_CHECKS = autopilot ? 50 : 5
    let goalChecks = 0

    // Per-turn abort controller — Esc/Ctrl+C aborts and returns to prompt; Ctrl+Q exits
    let turnAbort = new AbortController()
    const sigintHandler = () => {
      process.stdout.write(`\n${COLORS.yellow}⚡ Interrupted${COLORS.reset}\n`)
      turnAbort.abort()
    }
    const keypressHandler = (_: unknown, key: { name?: string; ctrl?: boolean } | undefined) => {
      if (!key) return
      if (key.name === "escape") {
        process.stdout.write(`\n${COLORS.yellow}⚡ Interrupted${COLORS.reset}\n`)
        turnAbort.abort()
      } else if (key.ctrl && key.name === "q") {
        process.stdout.write(`\n${COLORS.dim}Bye!${COLORS.reset}\n`)
        rl.close()
        process.exit(0)
      }
    }
    process.on("SIGINT", sigintHandler)
    if (isTTY) {
      process.stdin.setRawMode(true)
      pasteFilter.on("keypress", keypressHandler)
    }

    // Doom loop detection: track last 3 tool calls (name + stringified args)
    const recentToolCalls: string[] = []

    supervise: while (true) {
      for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
        if (estimateTokens(messages) > COMPACT_THRESHOLD) {
          const before = estimateTokens(messages)
          if (await compactMessages(currentModel, messages, turnAbort.signal))
            console.log(`↯ auto-compacted: ${before} → ${estimateTokens(messages)} est. tokens`)
        }

        try {
          let thinkingStarted = false
          let contentStarted = false

          const reply = await ollamaChat(currentModel, messages, toolDefsForCall, {
            onThinking(chunk) {
              if (!thinkingStarted) {
                process.stdout.write(`${COLORS.dim}`)
                thinkingStarted = true
              }
              process.stdout.write(chunk)
            },
            onContent(chunk) {
              if (thinkingStarted && !contentStarted) {
                process.stdout.write(`${COLORS.reset}\n\n`)
              }
              if (!contentStarted) contentStarted = true
              process.stdout.write(chunk)
            },
          }, undefined, turnAbort.signal, thinkingEnabled)

          if (thinkingStarted && !contentStarted) process.stdout.write(`${COLORS.reset}`)

          if (turnAbort.signal.aborted) break

          messages.push(reply)

          // No tool calls → finalize streamed response
          if (!reply.tool_calls?.length) {
            if (contentStarted || thinkingStarted) process.stdout.write("\n\n")
            if (autopilot && !autopilotState.exited && reflections < MAX_AUTOPILOT_REFLECTIONS) {
              reflections++
              console.log(`${COLORS.dim}↻ autopilot reflection ${reflections}/${MAX_AUTOPILOT_REFLECTIONS}${COLORS.reset}`)
              messages.push({ role: "user", content: AUTOPILOT_NUDGE })
              continue
            }
            break
          }

          // If there were tool calls, close any open styling
          if (thinkingStarted || contentStarted) process.stdout.write(`${COLORS.reset}\n`)

          // Execute tool calls
          for (const call of reply.tool_calls) {
            if (turnAbort.signal.aborted) break

            const toolName = call.function.name
            const toolArgs = call.function.arguments
            const tool = toolMap.get(toolName)

            console.log(`${COLORS.magenta}[${toolName}]${COLORS.reset} ${COLORS.dim}${formatToolArgs(toolArgs)}${COLORS.reset}`)

            let result: string
            if (!tool) {
              result = `Error: unknown tool '${toolName}'`
            } else {
              try {
                result = await tool.execute(toolArgs)
              } catch (err: unknown) {
                result = `Error: ${err instanceof Error ? err.message : String(err)}`
              }
            }

            if (turnAbort.signal.aborted) break

            // Show truncated result
            const preview = result.length > 500 ? result.slice(0, 500) + `\n${COLORS.dim}...(${result.length} chars total)${COLORS.reset}` : result
            console.log(`${COLORS.dim}${preview}${COLORS.reset}`)

            messages.push({ role: "tool", content: result, tool_name: toolName, tool_call_id: call.id })

            // Doom loop detection: track identical tool+args combos
            const callSig = toolName + JSON.stringify(toolArgs)
            recentToolCalls.push(callSig)
            if (recentToolCalls.length > 3) recentToolCalls.shift()
            if (recentToolCalls.length === 3 && recentToolCalls[0] === recentToolCalls[1] && recentToolCalls[1] === recentToolCalls[2]) {
              messages.push({ role: "user", content: `[System: You have called ${toolName} with identical arguments 3 times in a row. Change your approach — try a different tool or different arguments.]` })
              recentToolCalls.length = 0
            }
          }
          if (turnAbort.signal.aborted) break
          if (autopilotState.exited && !autopilotState.summarized) {
            autopilotState.summarized = true
            autopilot = false // exit returns to normal mode (blog: switch to build); re-arm with /autopilot
            messages.push({ role: "user", content: AUTOPILOT_SUMMARY_PROMPT })
          }
        } catch (err: unknown) {
          if (turnAbort.signal.aborted) break
          const msg = err instanceof Error ? err.message : String(err)
          console.error(`${COLORS.red}Error: ${msg}${COLORS.reset}`)
          // Push error as assistant message so conversation doesn't break
          messages.push({ role: "assistant", content: `I encountered an error: ${msg}` })
          break
        }
      }

      if (!goal) break supervise
      goalChecks++
      if (goalChecks > MAX_GOAL_CHECKS) {
        console.log(`${COLORS.yellow}⚠ supervisor: reached ${MAX_GOAL_CHECKS} check limit${autopilot ? "" : " — use /autopilot for longer runs"}${COLORS.reset}`)
        break supervise
      }
      let verdict: GoalVerdict
      try {
        verdict = await checkGoal(currentModel, goal, messages, turnAbort.signal)
      } catch (err: unknown) {
        console.log(`⚠ supervisor check failed: ${err instanceof Error ? err.message : String(err)} — stopping`)
        break supervise
      }
      if (turnAbort.signal.aborted) break supervise
      if (verdict.reached) { console.log(`✓ supervisor: goal reached`); break supervise }
      // Soft milestone warnings
      if (goalChecks === 5)  console.log(`${COLORS.yellow}⚠ supervisor: 5 checks — consider rephrasing goal or changing approach${COLORS.reset}`)
      if (goalChecks === 10) console.log(`${COLORS.yellow}⚠ supervisor: 10 checks — if stuck, interrupt with Esc${COLORS.reset}`)
      const nudge = buildSupervisorFeedback(goalChecks, goal, verdict.feedback)
      console.log(`↻ supervisor (check ${goalChecks}): ${verdict.feedback}`)
      messages.push({ role: "user", content: nudge })
    }

    process.off("SIGINT", sigintHandler)
    if (isTTY) {
      try { process.stdin.setRawMode(false) } catch {}
      pasteFilter.off("keypress", keypressHandler)
    }
  }
}

function formatToolArgs(args: Record<string, unknown>): string {
  const parts: string[] = []
  for (const [k, v] of Object.entries(args)) {
    const val = typeof v === "string" ? (v.length > 80 ? v.slice(0, 80) + "..." : v) : JSON.stringify(v)
    parts.push(`${k}=${val}`)
  }
  return parts.join(" ")
}

// ── Entry ────────────────────────────────────────────────────────────────────

if (import.meta.main) {
  main().catch((err) => {
    console.error(`${COLORS.red}Fatal: ${err.message}${COLORS.reset}`)
    process.exit(1)
  })
}
