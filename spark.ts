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

interface ModelInfo {
  capabilities: string[]
  contextLength: number  // native max context length from model_info
}

// Cache stores a Promise so concurrent cold calls for the same model share one /api/show request
const modelInfoCache = new Map<string, Promise<ModelInfo>>()

async function fetchModelInfo(model: string): Promise<ModelInfo> {
  try {
    const res = await fetch(`${getOllamaUrl()}/api/show`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: model }),
    })
    if (!res.ok) return { capabilities: [], contextLength: 0 }
    const data = (await res.json()) as {
      capabilities?: string[]
      model_info?: Record<string, unknown>
    }
    // model_info has arch-prefixed keys e.g. "qwen35.context_length", "llama.context_length"
    const info = data.model_info ?? {}
    const ctxEntry = Object.entries(info).find(([k]) => k.endsWith(".context_length"))
    const contextLength = typeof ctxEntry?.[1] === "number" ? ctxEntry[1] : 0
    return { capabilities: data.capabilities ?? [], contextLength }
  } catch { return { capabilities: [], contextLength: 0 } }
}

function ollamaModelInfo(model: string): Promise<ModelInfo> {
  let p = modelInfoCache.get(model)
  if (!p) {
    // Store the promise synchronously so concurrent cold calls share the same fetch.
    p = fetchModelInfo(model).then(info => {
      if (info.capabilities.length === 0 && info.contextLength === 0)
        modelInfoCache.delete(model)  // don't cache failures — allow retry
      return info
    })
    modelInfoCache.set(model, p)
  }
  return p
}

/** Capabilities for a model (cached). */
async function ollamaCapabilities(model: string): Promise<string[]> {
  return (await ollamaModelInfo(model)).capabilities
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
  nativeCtxLength = 0,  // from model_info; 0 = unknown
): Promise<Message> {
  const serializedMessages = messages.map(m =>
    m.thinking ? { ...m, thinking: undefined } : m
  )
  const body: Record<string, unknown> = { model, messages: serializedMessages, tools, stream: true }
  if (think) {
    body.think = true
    // num_ctx must cover both input and output. Use the model's native context
    // length (from /api/show model_info) capped to a sane max, then ensure
    // we leave at least 32768 tokens for the think block + response.
    // SPARK_NUM_CTX overrides everything.
    const envCtx = Number(process.env.SPARK_NUM_CTX) || 0
    const THINK_OUTPUT_RESERVE = 32768  // tokens reserved for think + response
    const MAX_CTX = 131072              // cap: 128k is enough for any current model
    let numCtx: number
    if (envCtx > 0) {
      numCtx = envCtx
    } else if (nativeCtxLength > 0) {
      // Use the model's native limit, but cap to MAX_CTX to avoid OOM
      numCtx = Math.min(nativeCtxLength, MAX_CTX)
    } else {
      // Unknown model — safe fallback: input budget (COMPACT_THRESHOLD tokens)
      // + output reserve. No * 4: COMPACT_THRESHOLD is already in tokens.
      numCtx = Math.min(32000 + THINK_OUTPUT_RESERVE, MAX_CTX)
    }
    // Qwen3 recommended sampling for thinking mode: DO NOT use greedy (temp=0).
    body.options = { num_ctx: numCtx, temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0 }
  }
  if (format && !think) { body.format = format; body.options = { temperature: 0 } }
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
    if (done || signal?.aborted) break

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
  const info = await ollamaModelInfo(model)
  const caps = info.capabilities
  const think = forceThink !== undefined
    ? (forceThink && caps.includes("thinking"))
    : (!format && caps.includes("thinking"))
  try {
    return await ollamaChatRaw(model, messages, tools, callbacks, format, think, signal, info.contextLength)
  } catch (e) {
    // If the model reported thinking support but rejects think:true, evict and retry once
    if (think && String(e).includes("400")) {
      modelInfoCache.delete(model)
      return ollamaChatRaw(model, messages, tools, callbacks, format, false, signal, 0)
    }
    throw e
  }
}

type ToolCallAccum = { id: string; name: string; args: string }

export async function openaiChatRaw(
  model: string,
  messages: Message[],
  tools: ToolDef[],
  callbacks: StreamCallbacks | undefined,
  signal?: AbortSignal,
): Promise<Message> {
  const serialized = messages.map(m => {
    const base = m.thinking ? { ...m, thinking: undefined } : { ...m }
    if (base.role === "tool") {
      // OpenAI doesn't want tool_name on tool result messages
      const { tool_name, ...rest } = base as typeof base & { tool_name?: string }
      void tool_name
      return rest
    }
    return base
  })

  const res = await fetch(`${process.env.SPARK_API_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.SPARK_API_KEY || ""}`,
    },
    body: JSON.stringify({ model, messages: serialized, tools, stream: true }),
    signal,
  })

  if (res.status === 401) {
    throw new Error("OpenAI 401: token expired or invalid. Re-run aurora-setup.ts")
  }
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`OpenAI error ${res.status}: ${text}`)
  }

  const contentType = res.headers.get("content-type") ?? ""

  // Non-streaming fallback (Aurora forces JSON when tools are present)
  if (contentType.includes("application/json")) {
    const data = (await res.json()) as {
      choices: [{
        message: {
          role: string
          content?: string
          tool_calls?: { id: string; type: string; function: { name: string; arguments: string } }[]
        }
      }]
    }
    const msg = data.choices[0].message
    const content = msg.content ?? ""
    if (content) callbacks?.onContent?.(content)
    const toolCalls: ToolCall[] = (msg.tool_calls ?? []).map(tc => ({
      id: tc.id,
      function: {
        name: tc.function.name,
        arguments: (() => { try { return JSON.parse(tc.function.arguments) } catch { return {} } })(),
      },
    }))
    const result: Message = { role: "assistant", content }
    if (toolCalls.length) result.tool_calls = toolCalls
    return result
  }

  // SSE streaming path
  const reader = res.body?.getReader()
  if (!reader) throw new Error("No response body")

  const decoder = new TextDecoder()
  let content = ""
  let buf = ""
  const toolCallMap = new Map<number, ToolCallAccum>()

  while (true) {
    if (signal?.aborted) break
    const { done, value } = await reader.read()
    if (done || signal?.aborted) break

    buf += decoder.decode(value, { stream: true })
    const lines = buf.split("\n")
    buf = lines.pop() ?? ""

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || !trimmed.startsWith("data:")) continue
      const payload = trimmed.slice(5).trim()
      if (payload === "[DONE]") break

      let chunk: {
        choices: [{
          delta: {
            content?: string
            tool_calls?: { index: number; id?: string; type?: string; function?: { name?: string; arguments?: string } }[]
          }
          finish_reason?: string | null
        }]
      }
      try { chunk = JSON.parse(payload) } catch { continue }

      const delta = chunk.choices?.[0]?.delta
      if (!delta) continue

      if (delta.content) {
        content += delta.content
        callbacks?.onContent?.(delta.content)
      }

      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index
          if (!toolCallMap.has(idx)) {
            toolCallMap.set(idx, { id: tc.id ?? "", name: tc.function?.name ?? "", args: "" })
          } else {
            const accum = toolCallMap.get(idx)!
            if (tc.id) accum.id = tc.id
            if (tc.function?.name) accum.name = tc.function.name
          }
          if (tc.function?.arguments) {
            toolCallMap.get(idx)!.args += tc.function.arguments
          }
        }
      }
    }
  }

  const toolCalls: ToolCall[] = []
  for (const [, accum] of [...toolCallMap.entries()].sort(([a], [b]) => a - b)) {
    let args: Record<string, unknown>
    try { args = JSON.parse(accum.args) } catch { args = {} }
    toolCalls.push({ id: accum.id, function: { name: accum.name, arguments: args } })
  }

  const result: Message = { role: "assistant", content }
  if (toolCalls.length) result.tool_calls = toolCalls
  return result
}

async function openaiModels(): Promise<string[]> {
  try {
    const res = await fetch(`${process.env.SPARK_API_URL}/v1/models`, {
      headers: { Authorization: `Bearer ${process.env.SPARK_API_KEY || ""}` },
    })
    if (!res.ok) return []
    const data = (await res.json()) as { data?: { id: string }[] }
    return (data.data ?? []).map(m => m.id)
  } catch { return [] }
}

async function openaiModelInfo(model: string): Promise<ModelInfo> {
  const CTX: Record<string, number> = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "o1": 200000,
    "o1-mini": 128000,
    "o3": 200000,
    "o3-mini": 200000,
    "o4-mini": 200000,
    "gpt-5": 200000,
    "gpt-5-mini": 128000,
  }
  // Prefix match for versioned names like "gpt-4o-2024-11-20"
  const ctx = CTX[model] ?? Object.entries(CTX).find(([k]) => model.startsWith(k))?.[1] ?? 128000
  return { capabilities: [], contextLength: ctx }
}

export async function openaiChat(
  model: string,
  messages: Message[],
  tools: ToolDef[],
  callbacks?: StreamCallbacks,
  format?: unknown,
  signal?: AbortSignal,
  forceThink?: boolean,
): Promise<Message> {
  return openaiChatRaw(model, messages, tools, callbacks, signal)
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
  // Filter out non-chat models (embedding-only, etc.) — they'd fail /api/chat immediately.
  const chatModels = models.filter(m => !m.toLowerCase().includes("embed"))
  const pool = chatModels.length > 0 ? chatModels : models  // fallback if only embeddings
  // Sort by: 1) model family score (desc), 2) param size (desc, bigger = smarter)
  return [...pool].sort((a, b) => {
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
    if (typeof m.thinking === "string") chars += m.thinking.length
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
    false,  // compaction is plain summarisation — no thinking needed
  )
  const summary = reply.content
  if (signal?.aborted || !summary.trim()) return false

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

function truncateLines(output: string, maxLines = 200): string {
  const lines = output.split("\n")
  if (lines.length <= maxLines) return output
  const kept = lines.slice(0, maxLines)
  const omitted = lines.length - maxLines
  return kept.join("\n") + `\n[...${omitted} lines omitted]`
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
          "Read a file from disk. Returns numbered lines with a header showing total line count and current window. For directories, lists entries. Use offset/limit for large files.",
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
      const end = offset - 1 + shown
      const header = `# ${filePath} | ${total} lines total | showing ${offset}-${end}`
      return truncateOutput(header + "\n" + numbered.join("\n"))
    },
  }
}

async function checkedWrite(filePath: string, content: string): Promise<string | null> {
  const ext = filePath.split(".").pop()?.toLowerCase() ?? ""
  const checkExts = ["ts", "tsx", "js", "jsx"]
  if (checkExts.includes(ext)) {
    try {
      const loader = (ext === "tsx" || ext === "jsx") ? ext : (ext === "ts" ? "ts" : "js") as "ts" | "js" | "tsx" | "jsx"
      new Bun.Transpiler({ loader }).transformSync(content)
    } catch (e: unknown) {
      return `Error: syntax check failed before writing ${filePath}:\n${String(e).slice(0, 1000)}`
    }
  }
  return null
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
            const checkErr = await checkedWrite(filePath, newStr)
            if (checkErr) return checkErr
            await writeFile(filePath, newStr, "utf-8")
            return `Created new file: ${filePath}`
          }
          return `Error: file not found for patching: ${filePath}`
        }

        // Normalize line endings
        existing = existing.replace(/\r\n/g, "\n")

        if (args.replaceAll) {
          const result = existing.replaceAll(oldStr, newStr)
          const checkErr = await checkedWrite(filePath, result)
          if (checkErr) return checkErr
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
          const checkErr = await checkedWrite(filePath, result)
          if (checkErr) return checkErr
          await writeFile(filePath, result, "utf-8")
          const oldLines = oldStr.split("\n").length
          const newLines = newStr.split("\n").length
          return `Patched ${filePath}: replaced ${oldLines} line(s) with ${newLines} line(s)`
        }

        // Fuzzy fallback: whitespace-insensitive line matching
        const fuzzy = lineTrimmedMatch(existing, oldStr)
        if (fuzzy) {
          const fuzzy2 = lineTrimmedMatch(existing.slice(fuzzy.end), oldStr)
          if (fuzzy2) {
            return `Error: Multiple fuzzy matches found for oldString — cannot safely patch. Make oldString more specific.`
          }
          const result = existing.slice(0, fuzzy.start) + newStr + existing.slice(fuzzy.end)
          const checkErr = await checkedWrite(filePath, result)
          if (checkErr) return checkErr
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
          const truncated = truncateOutput(truncateLines(output))
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
          "Evaluate JavaScript or TypeScript code in an isolated Bun subprocess. Returns the result of the last expression, or console output. Use for quick calculations, data transforms, or testing snippets.",
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
      // Run in a child process so process.exit() / infinite loops can't kill the REPL.
      // Wrap in an IIFE that prints the return value so the agent sees the result.
      const wrapped = `const __r = await (async()=>{ ${code} })(); if (__r !== undefined) console.log(__r)`
      return new Promise<string>((done) => {
        const proc = spawn("bun", ["--eval", wrapped], {
          cwd: process.cwd(),
          stdio: ["ignore", "pipe", "pipe"],
          env: process.env as Record<string, string>,
        })
        const chunks: Buffer[] = []
        const errChunks: Buffer[] = []
        proc.stdout!.on("data", (d: Buffer) => chunks.push(d))
        proc.stderr!.on("data", (d: Buffer) => errChunks.push(d))
        const TIMEOUT_MS = 30_000
        const timer = setTimeout(() => { proc.kill(); done("Error: Eval timed out after 30s") }, TIMEOUT_MS)
        proc.on("close", (code: number | null) => {
          clearTimeout(timer)
          const out = Buffer.concat(chunks).toString("utf-8").trim()
          const err = Buffer.concat(errChunks).toString("utf-8").trim()
          if (code !== 0) done(`Error (exit ${code ?? "killed"}): ${err || out || "(no output)"}`)
          else if (err) done(out ? `${out}\n[stderr] ${err}` : `[stderr] ${err}`)
          else done(out || "(no output)")
        })
        proc.on("error", (err: Error) => { clearTimeout(timer); done(`Error spawning eval: ${err.message}`) })
      })
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
      rgArgs.push("-e", String(args.pattern), "--", dir)
      try {
        const proc = Bun.spawn(["rg", ...rgArgs], { stdout: "pipe", stderr: "pipe" })
        const [output, errText] = await Promise.all([
          new Response(proc.stdout).text(),
          new Response(proc.stderr).text(),
        ])
        const exitCode = await proc.exited
        if (exitCode >= 2) return `Error: rg failed: ${errText.trim() || "unknown error"}`
        if (exitCode === 1 || !output.trim()) return "No matches found."
        return truncateLines(output.trim())
      } catch {
        return "Error: ripgrep (rg) not found. Install with: brew install ripgrep"
      }
    },
  }
}


function makeListSymbols(): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "ListSymbols",
        description: "List top-level symbols (functions, classes, constants, types) defined in a file with their line numbers. Use to understand a file's structure without reading all content.",
        parameters: {
          type: "object",
          properties: {
            filePath: { type: "string", description: "Absolute or relative path to the file" },
          },
          required: ["filePath"],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      const filePath = resolve(String(args.filePath))
      const content = await readFile(filePath, "utf-8").catch(() => null)
      if (!content) return `Error: file not found: ${filePath}`

      const ext = filePath.split(".").pop()?.toLowerCase() ?? ""
      const lines = content.split("\n")
      const symbols: string[] = []

      if (["ts", "tsx", "js", "jsx"].includes(ext)) {
        // Match: export function/class/const/type/interface/enum, and non-export top-level function/class
        const patterns = [
          /^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)/,
          /^(?:export\s+)?class\s+(\w+)/,
          /^(?:export\s+)?(?:const|let|var)\s+(\w+)/,
          /^(?:export\s+)?(?:type|interface|enum)\s+(\w+)/,
        ]
        lines.forEach((line, i) => {
          for (const pat of patterns) {
            const m = line.match(pat)
            if (m) { symbols.push(`${i + 1}: ${line.trim().slice(0, 80)}`); break }
          }
        })
      } else if (["py"].includes(ext)) {
        lines.forEach((line, i) => {
          if (/^(?:async\s+)?def\s+\w+|^class\s+\w+/.test(line)) {
            symbols.push(`${i + 1}: ${line.trim().slice(0, 80)}`)
          }
        })
      } else {
        // Generic: any line starting with word char at col 0 that looks like a definition
        lines.forEach((line, i) => {
          if (/^\w[\w\s*]*\(/.test(line) && !line.startsWith(" ") && !line.startsWith("\t")) {
            symbols.push(`${i + 1}: ${line.trim().slice(0, 80)}`)
          }
        })
      }

      if (symbols.length === 0) return `No top-level symbols found in ${filePath}`
      return `# ${filePath} — ${symbols.length} symbols\n` + symbols.join("\n")
    },
  }
}

function makeFindSymbol(): Tool {
  return {
    definition: {
      type: "function",
      function: {
        name: "FindSymbol",
        description: "Find where a function, class, or variable is defined across the codebase. Returns file paths and line numbers. Faster than Grep for finding symbol definitions.",
        parameters: {
          type: "object",
          properties: {
            name: { type: "string", description: "Symbol name to search for (exact identifier)" },
            path: { type: "string", description: "Directory to search in (default: cwd)" },
            include: { type: "string", description: "File pattern filter e.g. '*.ts'" },
          },
          required: ["name"],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      const name = String(args.name)
      const dir = resolve(String(args.path ?? process.cwd()))
      const include = args.include ? String(args.include) : undefined

      // Escape name to prevent regex injection
      const safeName = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      // Build a pattern that matches common definition forms
      const defPattern = `(function\\s+${safeName}|class\\s+${safeName}|const\\s+${safeName}\\s*=|let\\s+${safeName}\\s*=|var\\s+${safeName}\\s*=|def\\s+${safeName}|type\\s+${safeName}\\s*=|interface\\s+${safeName}[\\s{]|enum\\s+${safeName}[\\s{])`

      const rgArgs = ["--line-number", "--no-heading", "--color=never", "-e", defPattern]
      if (include) rgArgs.push("--glob", include)
      rgArgs.push(dir)

      try {
        const proc = Bun.spawn(["rg", ...rgArgs], { stdout: "pipe", stderr: "pipe" })
        const [output, errText] = await Promise.all([
          new Response(proc.stdout).text(),
          new Response(proc.stderr).text(),
        ])
        const exitCode = await proc.exited
        if (exitCode >= 2) return `Error: rg failed: ${errText.trim() || "unknown error"}`
        if (exitCode === 1 || !output.trim()) return `No definition found for '${name}'`
        const lines = output.trim().split("\n").slice(0, 50)
        return lines.join("\n")
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

// ── Parallel Task Registry ────────────────────────────────────────────────────
//
// Research: how GitHub Copilot CLI, opencode, and Claude Code implement parallel
// subagents and mid-run message passing — and how spark maps to each pattern.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 1. GITHUB COPILOT CLI                                                       │
// │    (the process running spark — /fleet /tasks /sidekicks /subagents)        │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// Spawn (non-blocking, background):
//   task(name, prompt, agent_type, mode:"background")
//   → returns agent_id immediately; agent runs concurrently on the JS event loop
//   → agent_type selects a named specialist: "explore", "general-purpose",
//     "rubber-duck", "code-review", "security-review", or custom agents under
//     ~/.agents/ (e.g. "gsd-code-reviewer", "gsd-executor", "gsd-planner")
//
// Await result:
//   read_agent(agent_id, wait:true, timeout?)
//   → blocks until the agent's current turn completes; returns full turn history
//   → read_agent(agent_id, since_turn:N) for delta reads (only turns after N)
//   → called automatically when the completion notification arrives
//
// Mid-run message injection (= task_steer equivalent):
//   write_agent(agent_id, message)
//   → delivers message as a new user turn to a running OR idle background agent
//   → if agent is running: queued, delivered after the current turn completes
//   → if agent is idle: wakes the agent to process the message as a new turn
//   → supports full multi-turn back-and-forth: write → wait → write again
//   → confirmed: used in this session for research refinement and follow-up
//
// List/enumerate:
//   list_agents(include_completed?)
//   → all active and completed background agents; status and agent_id
//   → statuses: running, idle, completed, failed, cancelled
//   → "(steerable)" = accepts write_agent; "(one-shot)" = read-only
//
// Completion notification:
//   → automatic event-driven notification when a background agent finishes
//   → no polling needed; notification arrives as the next agent turn
//   → pattern: launch → do independent work → wait for notification → read_agent
//
// Parallel pattern (used throughout this session):
//   task(A, mode:"background") → id_A    // start agent A
//   task(B, mode:"background") → id_B    // start agent B concurrently
//   // do independent work while both run …
//   read_agent(id_A, wait:true)           // collect A
//   read_agent(id_B, wait:true)           // collect B
//
// /fleet  — enables parallel subagent execution mode (auto-fans out work)
// /tasks  — view and manage running subagents and shell commands
//
// Agent statefulness:
//   → agents retain full conversation history across write_agent turns
//   → idle agents (done with last turn) re-engage on the next write_agent
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 2. OPENCODE                                                                  │
// │    (packages/opencode/src/tool/task.ts + task-interrupt.ts + interrupt.ts)  │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// Spawn:
//   Task({ description, prompt, subagent_type, directory?, task_id? })
//   → creates child Session with parentID; runs via ops.prompt() (Effect fiber)
//   → returns task_id (session UUID) in the tool result string
//   → task_id passed back to Task() resumes the prior session (stateful)
//   → subagent_type: "general", "explore", "scout", or user-defined agents
//
// Parallel (session/prompt.ts:1561, workflow/index.ts:405):
//   Effect.forEach(subtasks, handleSubtask, { concurrency: 16 })
//   → up to 16 parallel Effect fibers per turn
//   → workflow ctx.parallel(tasks, { concurrencyLimit? }) — worker pool via
//     withConcurrency(): shared queue drained by Promise.all of N workers
//
// Mid-run injection — task_steer (task-interrupt.ts):
//   task_steer({ task_id, reason })
//   → Interrupt.request({ sessionID, intent:"steer", reason, origin:"parent" })
//   → writes to pending Map<SessionID, Pending> (in-memory shared singleton)
//   → cancel beats steer: if cancel already queued, steer is silently dropped
//   → child calls Interrupt.consume(sessionID) at EVERY turn start
//   → steer → injects renderSteer(reason) as user msg, then continues normally
//   → renderSteer: <steer>\n<reason>…</reason>\n</steer>
//
// Graceful cancel — task_cancel:
//   → Interrupt.request(intent:"cancel") → renderCancel injected
//   → child gets CANCEL_GRACE_TURNS=2 turns to wrap up before forced stop
//   → renderCancel: <cancel>\n<reason>…</reason>\n</cancel>
//
// Hard abort — task_abort:
//   → Interrupt.abortChild() → Effect interrupt (no grace, immediate kill)
//
// Completion notification (session/status.ts):
//   → session.idle Bus event fires when child transitions to idle status
//   → plugin event handler: event.type === "session.idle" → runSupervisor()
//   → waitForResponse() polling fallback: 2s interval, checks time.completed
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 3. CLAUDE CODE — agents-supervisor                                          │
// │    (agents-supervisor/core/assessment.mjs, bin/on-stop.mjs)                │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// Spawn:
//   CC native Task() tool — creates a child context window for the subagent
//   → no task_id / no resume; each Task call is a fresh context
//   → subagent inherits core tools; Task itself excluded (no recursive spawning)
//
// Parallel:
//   Caller launches multiple subagent Promise chains simultaneously (Promise.all)
//   → no concurrency limit; all chains run on the JS event loop
//   → supervisor itself is strictly sequential: activeSupervisors Set<string>
//     ensures only one supervisor loop runs per session at a time
//
// Mid-run injection / completion:
//   → Stop hook: CC calls bin/on-stop.mjs synchronously on EVERY agent stop
//   → hook writes JSON to stdout:
//       { decision:"block", reason:"<feedback>" }  → CC injects as user turn
//       { decision:"proceed" }                     → agent allowed to stop
//   → stop_hook_active guard prevents infinite hook loops
//   → OpenCode variant: supervisor uses client.session.promptAsync() over HTTP
//   → no real-time steering; supervisor only acts AFTER the agent stops
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 4. GITHUB COPILOT CLOUD AGENT                                               │
// │    (docs.github.com/en/copilot/concepts/agents/cloud-agent)                │
// └─────────────────────────────────────────────────────────────────────────────┘
//
//   → Runs in GitHub Actions ephemeral VM; task IS the session (1 issue = 1 job)
//   → No Task() tool; no in-session subagent spawning
//   → Parallelism at session level only: multiple issues → multiple Actions jobs
//   → Mid-run injection: human types in GitHub chat UI; not agent-programmable
//   → Automations: trigger-based (cron / issue-opened / PR-opened); sequential
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 5. COMPARISON                                                               │
// └─────────────────────────────────────────────────────────────────────────────┘
//
//  Dimension        │ Copilot CLI          │ opencode             │ CC agents-sup     │ spark
// ──────────────────┼──────────────────────┼──────────────────────┼───────────────────┼──────────────────
//  Spawn API        │ task(agent_type,      │ Task({subagent_type, │ Task() native     │ Task({description,
//                   │   mode:"background") │   task_id?,...})     │   (no resume)     │   prompt})
//  Parallel         │ JS event loop;        │ Effect.forEach       │ Promise.all       │ Promise.all via
//                   │   /fleet mode        │   concurrency:16     │   (no limit)      │   TaskWait
//  Mid-run inject   │ write_agent(id, msg) │ task_steer →         │ Stop hook stdout  │ (timeout/abort)
//                   │   (running or idle)  │   Interrupt.Service  │   {block, reason} │
//  Await result     │ read_agent(id,        │ ops.prompt() await   │ Promise.then      │ TaskWait([ids])
//                   │   wait:true)         │   + session.idle evt │   inline          │   Promise.all
//  Cancel           │ write_agent("stop…") │ task_cancel →        │ n/a (hook stops)  │ timeout abort
//                   │                      │   CANCEL_GRACE=2     │                   │
//  Notify           │ auto event-driven    │ session.idle Bus     │ CC calls hook     │ Promise resolve
//  Resume session   │ write_agent wakes    │ task_id param        │ no                │ no
//                   │   idle agent         │   reuses session     │                   │
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ 6. SPARK IMPLEMENTATION CHOICES                                             │
// └─────────────────────────────────────────────────────────────────────────────┘
//
//   - Task() fires a background subagent Promise; returns task_id immediately
//   - TaskWait([ids]) runs Promise.all → N agents run concurrently
//   - No steer/cancel tools: timeout + user abort (Esc/Ctrl+C) handle it
//   - No resume: local Ollama sessions are cheap; just spawn a fresh Task
//   - No Effect fibers or Bus events: JS Promise + Map is sufficient for Ollama

export interface SubAgentHandle {
  id: string
  description: string
  promise: Promise<string>
  status: "running" | "done" | "cancelled" | "failed"
  abort: AbortController
}

// Module-level registry — shared across all makeTaskTools() calls in a session
export const taskRegistry = new Map<string, SubAgentHandle>()

export function makeTaskTools(
  model: string,
  systemPrompt: string,
  tools: Tool[],
  chatFn: typeof ollamaChat = ollamaChat,  // injectable for deterministic tests
): Tool[] {
  // Sub-agent tools: everything EXCEPT Task-family (no recursive spawning)
  const taskNames = new Set(["Task", "TaskWait"])
  const subTools = tools.filter(t => !taskNames.has(t.definition.function.name))
  const toolDefs = subTools.map(t => t.definition)
  const toolMap = new Map(subTools.map(t => [t.definition.function.name, t]))

  const MAX_TURNS = 20

  // ── Inner agent loop (runs in background via fire-and-forget Promise) ────
  async function runSubAgent(handle: SubAgentHandle, prompt: string): Promise<string> {
    const subMessages: Message[] = [
      { role: "system", content: systemPrompt },
      { role: "user", content: prompt },
    ]
    let lastContent = ""

    try {
      for (let turn = 0; turn < MAX_TURNS; turn++) {
        const reply = await chatFn(model, subMessages, toolDefs, undefined, undefined, handle.abort.signal)
        subMessages.push(reply)
        lastContent = reply.content

        if (!reply.tool_calls?.length) break

        for (const call of reply.tool_calls) {
          if (handle.abort.signal.aborted) break
          const tool = toolMap.get(call.function.name)
          const result = tool
            ? await tool.execute(call.function.arguments)
            : `Error: unknown tool '${call.function.name}'`
          subMessages.push({ role: "tool", content: result, tool_name: call.function.name, tool_call_id: call.id })
        }
        if (handle.abort.signal.aborted) break
      }
    } catch (err: unknown) {
      if (handle.abort.signal.aborted) {
        if (handle.status === "running") handle.status = "cancelled"
        return truncateOutput(lastContent)
      }
      const msg = err instanceof Error ? err.message : String(err)
      if (handle.status === "running") handle.status = "failed"
      return `Error: ${msg}`
    }

    if (handle.status === "running") handle.status = "done"
    return truncateOutput(lastContent)
  }

  // ── Task tool: start a background subagent, return task_id immediately ───
  const taskTool: Tool = {
    definition: {
      type: "function",
      function: {
        name: "Task",
        description:
          "Start a background subagent to handle an independent task. Returns a task_id immediately. " +
          "Use TaskWait to collect results. Spawn multiple Tasks then TaskWait([id1,id2,...]) to run in parallel. " +
          "Sub-agents cannot spawn further sub-agents.",
        parameters: {
          type: "object",
          properties: {
            description: { type: "string", description: "Short task description (3-5 words)" },
            prompt: { type: "string", description: "Detailed task instructions for the sub-agent" },
            timeout: { type: "number", description: "Timeout in ms (default 300000 = 5min); aborts and cancels the subagent on expiry" },
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
      const id = crypto.randomUUID()

      const handle: SubAgentHandle = {
        id,
        description: desc,
        promise: Promise.resolve(""),
        status: "running",
        abort: new AbortController(),
      }

      // Timeout: abort the in-flight chatFn. Aborting the shared controller makes
      // every subsequent chatFn call throw, so the agent stops immediately; the
      // catch in runSubAgent reports it as "cancelled" using cancelReason.
      const timeoutId = setTimeout(() => {
        if (handle.status === "running") {
          handle.status = "cancelled"
          handle.abort.abort()
        }
      }, timeout)

      process.stderr.write(`\x1b[90m🔧 [Task: ${desc}] started (id: ${id})\x1b[0m\n`)
      handle.promise = runSubAgent(handle, prompt).finally(() => clearTimeout(timeoutId))
      taskRegistry.set(id, handle)

      return `task_id: ${id}\nTask "${desc}" started in background. Call TaskWait(["${id}"]) to collect the result, or call more Tasks first to run them in parallel.`
    },
  }

  // ── TaskWait: block until one or more tasks complete ─────────────────────
  // opencode equivalent: the Effect.promise that wraps ops.prompt() in task.ts
  const taskWaitTool: Tool = {
    definition: {
      type: "function",
      function: {
        name: "TaskWait",
        description:
          "Wait for one or more background tasks to complete. Pass the task_ids returned by Task. " +
          "Returns all results when all tasks are done. Run tasks in parallel by calling Task multiple times before TaskWait.",
        parameters: {
          type: "object",
          properties: {
            task_ids: { type: "array", items: { type: "string" }, description: "Array of task_id strings returned by Task. Also accepts a single task_id string." },
            task_id: { type: "string", description: "Alias: single task_id string (use task_ids array for multiple)" },
          },
          required: [],
          additionalProperties: false,
        },
      },
    },
    async execute(args) {
      // Defensive: accept task_id (singular string), task_ids (array), or a bare string
      // Small models often pass task_id:"uuid" instead of task_ids:["uuid"]
      let ids: string[]
      if (Array.isArray(args.task_ids)) {
        ids = args.task_ids as string[]
      } else if (typeof args.task_ids === "string") {
        ids = [args.task_ids]
      } else if (typeof args.task_id === "string") {
        ids = [args.task_id as string]
      } else {
        ids = []
      }
      if (!ids.length) return "Error: task_ids must be a non-empty array"

      // allSettled so one failed task doesn't discard sibling results
      const settled = await Promise.allSettled(
        ids.map(async (id) => {
          const handle = taskRegistry.get(id)
          if (!handle) return `task_id ${id}: not found`
          const result = await handle.promise
          taskRegistry.delete(id)  // evict after collecting to prevent unbounded growth
          return `<task_result task_id="${id}" description="${handle.description}" status="${handle.status}">\n${result}\n</task_result>`
        }),
      )

      return settled
        .map(r => r.status === "fulfilled" ? r.value : `<task_result status="failed">\nError: ${(r as PromiseRejectedResult).reason}\n</task_result>`)
        .join("\n\n")
    },
  }

  return [taskTool, taskWaitTool]
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
    const fd = await readFile(path, "utf-8").catch(() => "")
    // Only read enough to extract description — first 40 lines covers any frontmatter
    const preview = fd.split("\n").slice(0, 40).join("\n")
    const fmMatch = preview.match(/^---\s*\n([\s\S]*?)\n---/)
    let desc = ""
    if (fmMatch) {
      const descMatch = fmMatch[1].match(/description:\s*(.+)/)
      if (descMatch) desc = descMatch[1].trim()
    }
    if (!desc) {
      const bodyLines = preview.split("\n").filter((l) => l.trim() && !l.startsWith("#") && !l.startsWith("---"))
      desc = bodyLines[0]?.trim().slice(0, 120) ?? ""
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

  // Use spark-sandbox image (Bun+Node+Go+Aurora); fall back to plain bun if not yet built.
  const useLocalImage = await new Promise<boolean>(res => {
    const p = spawn("docker", ["image", "inspect", "spark-sandbox"], { stdio: "ignore" })
    p.on("close", code => res(code === 0))
    p.on("error", () => res(false))
  })
  const image = useLocalImage ? "spark-sandbox" : "oven/bun:alpine"

  // Pass CHATGPT_ACCESS_TOKEN so Aurora can start inside the container.
  const tokenEnv = process.env.CHATGPT_ACCESS_TOKEN
    ? ["-e", `CHATGPT_ACCESS_TOKEN=${process.env.CHATGPT_ACCESS_TOKEN}`]
    : []

  // Forward explicit API config when already set by the user.
  const apiEnv = process.env.SPARK_API_URL
    ? ["-e", `SPARK_API_URL=${process.env.SPARK_API_URL}`, "-e", `SPARK_API_KEY=${process.env.SPARK_API_KEY ?? ""}`]
    : []

  const dockerArgs = [
    "run", "--rm", "-it",
    "--add-host=host.docker.internal:host-gateway",  // Linux compat; macOS Docker Desktop has it built-in
    "-e", `OLLAMA_URL=${dockerOllamaUrl}`,
    "-e", "SPARK_SANDBOX=1",
    ...tokenEnv,
    ...apiEnv,
    ...volumes.flatMap(v => ["-v", v]),
    "-w", "/workspace",
    image,
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
- After patching code, run the relevant test command (e.g. bun test, pytest, go test) to verify correctness before declaring done.

## Reasoning
Before writing code or making decisions, think through the problem:
- Identify the root cause, not just the symptom
- Consider at least two approaches and evaluate trade-offs
- Check for edge cases and failure modes before committing to a solution
- Verify your reasoning matches what the code actually does, not what you expect it to do

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
//
// DESIGN CONTEXT: how GitHub Copilot and Claude Code handle goal supervision,
// and how spark.ts adapts those ideas for a local single-file Ollama agent.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │ GitHub Copilot Cloud Agent (docs.github.com/en/copilot/how-tos/use-    │
// │ copilot-agents/cloud-agent)                                             │
// │                                                                         │
// │ Copilot has no "/goal" command. The task IS the goal — it comes from    │
// │ an issue, chat message, PR comment, CLI prompt, or API call. Copilot   │
// │ runs autonomously in a cloud sandbox, creates a branch, makes changes, │
// │ and opens a PR. Completion is defined by: PR created + built-in code   │
// │ review + security scan pass. There is no interactive judge loop —      │
// │ the agent either finishes and ships a PR, or stalls and times out.     │
// │                                                                         │
// │ Key features:                                                           │
// │ - Built-in Copilot code review (second-opinion on every PR)            │
// │ - Security validation (hardcoded secrets, insecure deps)               │
// │ - Automations: trigger on schedule or GitHub events (issue opened etc) │
// │ - No explicit attempt cap exposed to user; cloud infra enforces limits  │
// └─────────────────────────────────────────────────────────────────────────┘
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │ Claude Code agents-supervisor plugin (github.com/dzianisv/             │
// │ agents-supervisor — core/goal.mjs, opencode/supervisor-impl.ts)        │
// │                                                                         │
// │ 3 supervisor modes:                                                     │
// │   reflection (default) — judge fires on every Stop hook, no goal       │
// │   goal                 — /supervisor:goal <condition> sets explicit     │
// │                          goal; judge checks it each turn               │
// │   autopilot            — skip judge, always continue until budget gone  │
// │                                                                         │
// │ Goal state (per-session JSON, mode 0600):                               │
// │   { condition, status, attempts, tokenBaseline, startedAt, deadline }   │
// │   DEFAULT_MAX_ATTEMPTS = 16                                             │
// │   DEFAULT_MAX_GOAL_DURATION_MS = 30 * 60 * 1000  (30 min deadline)     │
// │                                                                         │
// │ Goal injection (buildGoalRequirementSection):                           │
// │   "## GOAL (mandatory completion requirement)"                          │
// │   + evidence rule: "bare assertion does NOT count as evidence"          │
// │                                                                         │
// │ 2-stage judge:                                                          │
// │   Stage 1: agent self-assessment JSON (status/stuck/missing/evidence)  │
// │   Stage 2: external judge prompt validates the self-assessment          │
// │                                                                         │
// │ Escalating feedback (buildEscalatingFeedback):                          │
// │   - Planning loop detected (read-only tools, no writes) → hard STOP    │
// │   - Action loop detected (same commands repeated) → change approach    │
// │   - Normal: gentle → missing/next_actions list → stuck warning         │
// │                                                                         │
// │ Antipatterns mined from 227 real agent stops (78% premature):          │
// │   PERMISSION-SEEKING, STOPPED-WITH-TODOS, VERIFICATION-DEFERRAL,       │
// │   FALSE-COMPLETE                                                        │
// └─────────────────────────────────────────────────────────────────────────┘
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │ spark.ts — simplified adaptation for local Ollama                      │
// │                                                                         │
// │ 2 states instead of 3 modes:                                            │
// │   goal string (null = off)  — set via /goal or synthesised by LLM      │
// │   autopilot bool            — extends tool budget to 200 rounds,       │
// │                               adds autopilot_exit tool                 │
// │                                                                         │
// │ Key differences from CC:                                                │
// │ - No deadline (30-min wall would kill long Ollama tasks)               │
// │ - No self-assessment stage (adds a full LLM round per check)           │
// │ - No per-session JSON (single .agents/spark/goal file, plain text)     │
// │ - deriveGoal(): LLM synthesises goal from conversation — CC requires   │
// │   the user to write the condition manually                             │
// │ - Bounded check cap: interactive=5, autopilot=50 (CC default: 16)      │
// │ - "copilot: ● Started autopilot objective #N" display format           │
// └─────────────────────────────────────────────────────────────────────────┘

/**
 * Injects the active goal into the system prompt as a MANDATORY completion
 * requirement with an evidence rule.
 *
 * Design: mirrors Claude Code agents-supervisor `buildGoalRequirementSection()`
 * (core/goal.mjs). The key insight from that implementation: the goal block
 * must be framed as MANDATORY — not advisory — and must include an evidence
 * rule so the agent knows it cannot just assert "done". Without this framing,
 * small local models frequently declare completion without verification.
 *
 * Unlike CC, we also tell the agent WHERE to put the evidence (in its reply),
 * because spark's judge only reads the final assistant message — not raw tool
 * logs. The agent is responsible for surfacing proof in prose.
 *
 * Ref: ~/workspace/agents-supervisor/core/goal.mjs → buildGoalRequirementSection()
 */
export function buildGoalBlock(goal: string): string {
  if (!goal.trim()) return ""
  return `

## GOAL (mandatory completion requirement)

MANDATORY: The following goal MUST be demonstrably met before the task is complete:

  "${goal.trim()}"

Evidence rule: when you believe this goal is met, you MUST include the evidence
directly in your response — paste the relevant command output, test results, or
file contents. A bare assertion ("I'm done", "tests pass") without showing the
output does NOT count as evidence and will NOT be accepted as completion.`
}

/**
 * Builds an escalating supervisor feedback message based on how many judge
 * checks have already fired for the current turn.
 *
 * Design: simplified port of Claude Code agents-supervisor
 * `buildEscalatingFeedback()` (core/feedback.mjs). The CC version detects
 * planning loops (read-only tool pattern) and action loops (repeated identical
 * commands) and emits hard STOP messages. spark.ts uses a simpler count-based
 * escalation without loop detection (doom-loop detection in the tool call layer
 * already handles the action-loop case).
 *
 * Escalation levels:
 *   1-2:  gentle nudge — keep working, include goal text
 *   3-4:  firmer — "if you've been reading files, start writing"
 *   5-9:  strong — "STOP PLANNING, pick one action and execute it"
 *   10+:  final warning — "complete NOW or call autopilot_exit"
 *
 * The feedback string from checkGoal() is prepended so the agent sees
 * BOTH the judge's specific next-step suggestion AND the escalating tone.
 *
 * Ref: ~/workspace/agents-supervisor/core/feedback.mjs → buildEscalatingFeedback()
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



/** Verdict from the goal judge. Kept minimal — CC uses a richer schema with
 *  status/stuck/missing[]/next_actions[] but that requires a larger model to
 *  reliably produce. For small local Ollama models, a binary reached+feedback
 *  is more reliable and fast enough for the check interval. */
export interface GoalVerdict { reached: boolean; feedback: string }

const VERDICT_SCHEMA = {
  type: "object",
  properties: {
    reached: { type: "boolean" },
    feedback: { type: "string" },
  },
  required: ["reached", "feedback"],
} as const

/** Parses the judge's JSON response. Fails safe to reached=false so the agent
 *  keeps working rather than falsely declaring done on a malformed reply. */
export function parseVerdict(text: string): GoalVerdict {
  try {
    const parsed = JSON.parse(text)
    return { reached: Boolean(parsed.reached), feedback: String(parsed.feedback ?? "") }
  } catch {
    return { reached: false, feedback: "" }
  }
}

/**
 * LLM-derives or refines a goal from the conversation context.
 *
 * This is spark's main divergence from Claude Code: CC requires the user to
 * write the goal condition manually (/supervisor:goal <condition>). spark
 * synthesises it automatically from recent conversation turns, making the
 * system work without the user knowing the exact goal syntax.
 *
 * If existingGoal is set (via /goal or a prior /autopilot run), the model
 * refines it to be more specific and verifiable. Otherwise it derives a goal
 * from scratch. Either way the output is a single actionable sentence.
 *
 * Injected supervisor/system messages are filtered before the context window
 * so they don't pollute the goal synthesis (they contain "Goal not yet reached"
 * which would mislead the synthesiser into making a negative goal).
 *
 * GitHub Copilot equivalent: the user's issue/chat message IS the goal —
 * no synthesis step needed because the task is already written by the user.
 */

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
    [], undefined, undefined, signal, false,  // goal derivation — no thinking
  )

  const derived = (reply.content ?? "").trim().slice(0, 300)
  return derived || existingGoal || "Complete the current task"
}

export async function checkGoal(model: string, goal: string, messages: Message[], signal?: AbortSignal): Promise<GoalVerdict> {
  /**
   * Goal judge — evaluates whether the agent's last response demonstrates
   * that the goal has been met.
   *
   * DESIGN: spark uses a 1-stage judge (single LLM call → JSON verdict).
   * Claude Code uses a 2-stage judge:
   *   Stage 1: agent self-assessment → JSON with status/stuck/missing/evidence
   *   Stage 2: external judge validates the self-assessment
   * The 2-stage approach is more accurate but costs 2× inference rounds and
   * requires a model large enough to reliably produce the structured schema.
   * For small local Ollama models, a single focused judge call is more reliable.
   *
   * CONTEXT WINDOW: judge sees last real user message + last assistant message.
   * - Injected [supervisor]/[autopilot]/[System:] messages are skipped when
   *   finding lastUser — they contain "Goal not yet reached" which would
   *   confuse the judge into a false negative.
   * - Tool result messages are NOT included — the agent is responsible for
   *   pasting evidence into its prose response (buildGoalBlock enforces this).
   *   Including raw tool logs inflates the prompt and confuses small models;
   *   CC's stop-hook judge reads the full transcript naturally, but our judge
   *   receives a constructed prompt.
   *
   * GitHub Copilot equivalent: completion check is external (PR created + CI
   * green + code review pass) rather than an LLM judge call.
   *
   * Ref: ~/workspace/agents-supervisor/core/assessment.mjs → buildJudgePrompt()
   *      ~/workspace/blogposts/claude-code-stop-hook-reflection-judge.md
   */
  const lastAssistant = [...messages].reverse().find(m => m.role === "assistant")?.content ?? ""
  // Skip injected supervisor/system messages when finding the last real user message.
  // Includes <system-reminder> (AUTOPILOT_NUDGE) and "Autopilot completed" (AUTOPILOT_SUMMARY_PROMPT)
  // so the judge always sees the original task message, not autopilot bookkeeping noise.
  const INJECTED_PREFIXES = ["[supervisor]", "[autopilot]", "[System:", "<system-reminder>", "Autopilot completed"]
  const lastUser = [...messages].reverse().find(
    m => m.role === "user" &&
    typeof m.content === "string" &&
    !INJECTED_PREFIXES.some(p => (m.content as string).startsWith(p))
  )?.content ?? ""
  // The judge evaluates the agent's final claim — the agent is responsible for
  // including evidence (test output, command results) in its response text.
  // buildGoalBlock enforces this on the agent side. We don't re-parse tool msgs
  // here because: (a) they're noisy/truncated, (b) good agents already summarise
  // them in prose, (c) it inflates the judge prompt for small local models.
  const judgeSystem = "You are a strict goal supervisor for a coding agent. You judge whether a GOAL is fully accomplished based on the conversation. Be skeptical of unverified claims."
  const judgeUser = `GOAL: ${goal}\n\nLast user message:\n${typeof lastUser === "string" ? lastUser : ""}\n\nAgent's last message:\n${typeof lastAssistant === "string" ? lastAssistant : ""}\n\nIs the GOAL fully reached? Respond with ONLY a JSON object: {"reached": true|false, "feedback": "<if not reached, one concrete next action to push the agent forward; empty string if reached>"}`
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

  // 1. Check backend (OpenAI-compat if SPARK_API_URL set, else Ollama)
  const isOpenAI = Boolean(process.env.SPARK_API_URL)

  let models: string[]
  if (isOpenAI) {
    models = await openaiModels()
    if (models.length === 0) {
      console.error(`${COLORS.red}Error: No models returned from ${process.env.SPARK_API_URL}. Is Aurora running?${COLORS.reset}`)
      process.exit(1)
    }
  } else {
    models = await ollamaModels()
    if (models.length === 0) {
      console.error(`${COLORS.red}Error: No Ollama models found. Is Ollama running?${COLORS.reset}`)
      console.error(`${COLORS.dim}Start it with: ollama serve${COLORS.reset}`)
      process.exit(1)
    }
  }

  // Try saved model first, fall back to auto-pick
  const savedModel = await loadSavedModel()
  let currentModel: string
  let thinkingEnabled: boolean | undefined = undefined  // undefined = auto (on for capable models)
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
  const listSymbolsTool = makeListSymbols()
  const findSymbolTool = makeFindSymbol()
  const coreTools = [readFileTool, writeFileTool, bashTool, evalTool, globTool, grepTool, listSymbolsTool, findSymbolTool, loadSkillTool]

  // 4. Build system prompt
  const systemPrompt = buildSystemPrompt(agentInstructions, skillList, currentModel, gitContext)

  // 5. Build full tool set with Task-family tools having a lazy systemPrompt reference
  const chatFn = isOpenAI ? openaiChat : ollamaChat
  const buildTools = () => {
    const taskTools = makeTaskTools(currentModel, systemPrompt, coreTools, chatFn)
    return [...coreTools, ...taskTools]
  }

  // 6. Init conversation
  // baseSystemPrompt never contains the goal block — goal is appended dynamically.
  // This avoids the fragile regex-strip approach: a user's AGENTS.md containing
  // "## GOAL (mandatory" would match the strip regex and corrupt the system prompt.
  const baseSystemPrompt = systemPrompt
  const setGoalBlock = (g: string | null) => {
    messages[0].content = baseSystemPrompt + (g ? buildGoalBlock(g) : "")
  }

  const messages: Message[] = [{ role: "system", content: systemPrompt }]
  let goal: string | null = await loadGoal()
  if (goal) setGoalBlock(goal)
  let autopilot = false
  let alwaysAutopilot = true
  let goalIsExplicit = false

  printHeader(currentModel, skillMap.size)
  if (isOpenAI) {
    console.log(`${COLORS.dim}Backend: OpenAI-compat  ${process.env.SPARK_API_URL}${COLORS.reset}`)
  }

  // Announce restored goal after header so user knows why the supervisor may kick in
  if (goal) {
    console.log(`${COLORS.dim}Restored goal: ${goal}  (use /goal clear to remove)${COLORS.reset}\n`)
  }

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
      rl.question(`👤 ${COLORS.green}> ${COLORS.reset}`, res)
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
        process.stdout.write(`👤 ${COLORS.green}... ${COLORS.reset}`)
      } else {
        lines.push(line)
        return lines.join("\n")
      }
    }
  }

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
      autopilot = false
      console.log(`${COLORS.dim}Conversation cleared.${COLORS.reset}\n`)
      continue
    }

    if (trimmed === "/models") {
      try {
        const freshModels = isOpenAI ? await openaiModels() : await ollamaModels()
        currentModel = await selectModel(freshModels, currentModel, rl)
      } catch (err: unknown) {
        console.log(`⚠ /models failed: ${err instanceof Error ? err.message : String(err)}`)
      }
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
        console.log(`Thinking mode ON — forced on`)
      } else if (arg === "off" || arg === "0" || arg === "false") {
        thinkingEnabled = false
        console.log(`Thinking mode OFF — forced off`)
      } else if (arg === "auto" || arg === "") {
        thinkingEnabled = undefined
        console.log(`Thinking mode AUTO — enabled for capable models (default)`)
      } else {
        console.log(`Usage: /think on|off|auto`)
      }
      continue
    }

    if (trimmed === "/goal") {
      console.log(goal ? `goal: ${goal}` : `(no goal set)`)
      continue
    }

    // /goal — set or clear the completion goal
    if (trimmed.startsWith("/goal ")) {
      const arg = trimmed.slice("/goal ".length).trim()
      if (arg === "clear") {
        goal = null
        goalIsExplicit = false
        await clearGoal()
        setGoalBlock(null)
        console.log(`goal cleared`)
      } else {
        goal = arg
        goalIsExplicit = true
        await saveGoal(arg)
        console.log(`goal set: ${arg}`)
        setGoalBlock(arg)
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
        console.log(`♻️  ${COLORS.cyan}compacted${COLORS.reset} ${COLORS.dim}${before} → ${after} est. tokens (kept system + last ${TAIL_TURNS} turns)${COLORS.reset}`)
      }
      continue
    }

    if (trimmed === "/autopilot") {
      console.log(`always-autopilot: ${alwaysAutopilot ? "ON" : "OFF"} | current turn: ${autopilot ? "ON" : "OFF"}`)
      continue
    }

    let skipGenericPush = false

    // /autopilot — autonomous goal-driven loop
    if (trimmed.startsWith("/autopilot ")) {
      const arg = trimmed.slice("/autopilot ".length).trim()
      if (arg === "off") {
        autopilot = false
        alwaysAutopilot = false
        console.log(`autopilot OFF`)
        continue
      }

      if (arg === "on") {
        alwaysAutopilot = true
        console.log(`autopilot ON (always-on mode)`)
        continue
      }

      try {
        autopilot = true
        alwaysAutopilot = true
        skipGenericPush = true

        // Include any inline task arg as staging context for goal derivation,
        // but don't push it as a raw message — the kick-off message below replaces it.
        const stagedArg = arg
        const stagingMessages: Message[] = stagedArg && stagedArg !== "on"
          ? [...messages, { role: "user", content: stagedArg }]
          : messages

        process.stdout.write(`${COLORS.dim}deriving goal…${COLORS.reset}\r`)
        const derived = await deriveGoal(currentModel, stagingMessages, goal)
        goal = derived
        await saveGoal(derived)

        setGoalBlock(derived)

        const objectiveN = (await loadAutopilotCount()) + 1
        await saveAutopilotCount(objectiveN)

        const preview = derived.length > 80 ? derived.slice(0, 80) + "…" : derived
        process.stdout.write("                              \r")
        // Display format mirrors GitHub Copilot's cloud agent "● Started" indicator
        console.log(`copilot: ${COLORS.green}●${COLORS.reset} Started autopilot objective #${objectiveN}: ${preview}`)

        // Kick-off message anchors the agent to the refined goal from turn 1
        messages.push({
          role: "user",
          content: `[autopilot] Objective #${objectiveN}: ${derived}\nWork toward this goal autonomously. Use tools. Provide evidence (command output, file contents, test results) when done. Call autopilot_exit only when the goal is fully achieved and verified.`,
        })
        // fall through into the agent loop below by NOT continuing
      } catch (err: unknown) {
        autopilot = false
        process.stdout.write("                              \r")
        console.log(`⚠ /autopilot failed: ${err instanceof Error ? err.message : String(err)}`)
        continue
      }
    }

    // Always-autopilot: derive goal from user message if none is set
    if (alwaysAutopilot && !skipGenericPush) {
      autopilot = true
      if (!goal) {
        // Derive goal from this user message in a subagent call
        const stagingMessages: Message[] = [...messages, { role: "user", content: trimmed }]
        process.stdout.write(`${COLORS.dim}deriving goal…${COLORS.reset}\r`)
        try {
          const derived = await deriveGoal(currentModel, stagingMessages, null)
          if (derived) {
            goal = derived
            await saveGoal(derived)
            setGoalBlock(derived)
            const preview = derived.length > 80 ? derived.slice(0, 80) + "…" : derived
            process.stdout.write("                              \r")
            console.log(`${COLORS.dim}goal: ${preview}${COLORS.reset}`)
          }
        } catch {
          process.stdout.write("                              \r")
        }
      }
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
    // Also abort all running sub-agents so TaskWait unblocks immediately.
    const abortTurn = () => {
      turnAbort.abort()
      for (const h of taskRegistry.values()) {
        if (h.status === "running") { h.abort.abort() }
      }
    }
    const sigintHandler = () => {
      process.stdout.write(`\n${COLORS.yellow}⚡ Interrupted${COLORS.reset}\n`)
      abortTurn()
    }
    const keypressHandler = (_: unknown, key: { name?: string; ctrl?: boolean } | undefined) => {
      if (!key) return
      if (key.name === "escape" || (key.ctrl && key.name === "c")) {
        process.stdout.write(`\n${COLORS.yellow}⚡ Interrupted${COLORS.reset}\n`)
        abortTurn()
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

    // supervise: loop — the goal supervisor outer loop
    //
    // Structure: the inner for-loop runs the agent (model + tool calls) for up
    // to MAX_TOOL_ROUNDS, then exits. The outer supervise: while(true) calls
    // checkGoal() and either breaks (done / cap hit) or pushes a supervisor
    // nudge and re-enters the inner loop for another batch of tool rounds.
    //
    // This mirrors Claude Code's Stop-hook pattern at a high level:
    //   CC:    agent finishes turn → Stop hook fires → judge decides DONE|CONTINUE
    //          → if CONTINUE, re-prompts the live session with feedback
    //   spark: agent exhaust tool rounds → checkGoal fires inline → if not
    //          reached, push escalating nudge → re-enter inner loop
    //
    // Key parameters:
    //   MAX_TOOL_ROUNDS: 30 (interactive) | 200 (autopilot) — inner loop cap
    //   MAX_GOAL_CHECKS: 5  (interactive) | 50  (autopilot) — outer loop cap
    //
    // CC uses DEFAULT_MAX_ATTEMPTS=16 with a 30-min deadline regardless of mode.
    // spark separates interactive vs autopilot because interactive users expect
    // the REPL to return control quickly; autopilot users expect long runs.
    //
    // Loop exits:
    //   1. !goal           — no goal set, single pass only
    //   2. goalChecks > cap — soft limit reached (print warning, return to REPL)
    //   3. verdict.reached  — judge confirmed goal met
    //   4. checkGoal throws — Ollama error, stop safely
    //   5. turnAbort.signal.aborted — user pressed Esc/Ctrl+C
    supervise: while (true) {
      recentToolCalls.length = 0  // reset doom-loop detector each supervisor cycle
      for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
        try {
          if (estimateTokens(messages) > COMPACT_THRESHOLD) {
            const before = estimateTokens(messages)
            if (await compactMessages(currentModel, messages, turnAbort.signal))
              console.log(`\n♻️  ${COLORS.cyan}auto-compacted${COLORS.reset} ${COLORS.dim}${before} → ${estimateTokens(messages)} est. tokens (kept system + last ${TAIL_TURNS} turns)${COLORS.reset}`)
          }
        } catch (compactErr) {
          if (turnAbort.signal.aborted) break
          console.error(`${COLORS.dim}⚠ compaction failed: ${compactErr instanceof Error ? compactErr.message : String(compactErr)} — continuing${COLORS.reset}`)
        }

        try {
          let thinkingStarted = false
          let contentStarted = false

          // Show spinner while waiting for first token
          process.stdout.write(`🤖 ${COLORS.dim}…${COLORS.reset}`)
          let spinnerCleared = false
          const clearSpinner = () => {
            if (!spinnerCleared) {
              process.stdout.write(`\r${COLORS.dim}   \r${COLORS.reset}`)
              spinnerCleared = true
            }
          }

          const reply = await (isOpenAI ? openaiChat : ollamaChat)(currentModel, messages, toolDefsForCall, {
            onThinking(chunk) {
              clearSpinner()
              if (!thinkingStarted) {
                process.stdout.write(`🤖 ${COLORS.dim}`)
                thinkingStarted = true
              }
              process.stdout.write(chunk)
            },
            onContent(chunk) {
              clearSpinner()
              if (thinkingStarted && !contentStarted) {
                process.stdout.write(`${COLORS.reset}\n\n🤖 `)
              }
              if (!contentStarted) {
                if (!thinkingStarted) process.stdout.write(`🤖 `)
                contentStarted = true
              }
              process.stdout.write(chunk)
            },
          }, undefined, turnAbort.signal, thinkingEnabled)

          clearSpinner()

          if (thinkingStarted && !contentStarted) process.stdout.write(`${COLORS.reset}`)

          if (turnAbort.signal.aborted) break

          // Echo-suppression: some small models prefix their response by repeating
          // the last user message verbatim. Strip it so it doesn't clutter the output.
          if (reply.content && messages.length >= 2) {
            const lastUserMsg = [...messages].reverse().find(m => m.role === "user")
            if (lastUserMsg && typeof lastUserMsg.content === "string") {
              const userText = lastUserMsg.content.trim()
              if (reply.content.trimStart().startsWith(userText)) {
                reply.content = reply.content.trimStart().slice(userText.length).trimStart()
              }
            }
          }

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

            console.log(`🔧 ${COLORS.magenta}[${toolName}]${COLORS.reset} ${COLORS.dim}${formatToolArgs(toolArgs)}${COLORS.reset}`)

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
          if (turnAbort.signal.aborted) {
            // Strip the assistant tool_calls message if we never pushed matching tool results —
            // an unanswered tool_calls entry confuses the model on the next turn.
            const last = messages[messages.length - 1]
            if (last?.role === "assistant" && last.tool_calls?.length) messages.pop()
            break
          }
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
      // Trust an explicit autopilot_exit — agent signalled done, skip the judge.
      if (autopilotState.exited) { console.log(`✓ supervisor: agent called autopilot_exit`); break supervise }
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

    autopilot = false
    // Re-arm autopilot for the next turn; clear derived goal so next message gets a fresh one
    if (alwaysAutopilot && !goalIsExplicit) {
      goal = null
      await clearGoal()
      setGoalBlock(null)
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
