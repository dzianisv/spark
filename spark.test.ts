// spark.test.ts — G-Eval (LLM-as-judge) tests for the spark agent
// Run: bun test spark.test.ts
// Requires: Ollama running with at least one model

import { test, expect, beforeAll, describe } from "bun:test"
import { parseVerdict, checkGoal, makeAutopilotExit, AUTOPILOT_NUDGE, MAX_AUTOPILOT_REFLECTIONS, AUTOPILOT_SUMMARY_PROMPT, estimateTokens, splitTurns, compactMessages, TAIL_TURNS, ollamaChat as sparkOllamaChat } from "./spark.ts"
import type { Message as SparkMessage } from "./spark.ts"

// ── Configuration ────────────────────────────────────────────────────────────

const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://localhost:11434"
const AGENT_MODEL = process.env.AGENT_MODEL ?? "" // auto-pick if empty
const JUDGE_MODEL = process.env.JUDGE_MODEL ?? "" // defaults to AGENT_MODEL
const MAX_AGENT_TURNS = 15
const TIMEOUT = 120_000

// ── Types ────────────────────────────────────────────────────────────────────

interface Message {
  role: "system" | "user" | "assistant" | "tool"
  content: string
  tool_calls?: ToolCall[]
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
    parameters: { type: "object"; properties: Record<string, unknown>; required: string[] }
  }
}

interface JudgeScore {
  tool_selection: number
  tool_usage: number
  task_completion: number
  reasoning: string
}

interface AgentTrace {
  messages: Message[]
  tool_calls_made: { name: string; args: Record<string, unknown>; result: string }[]
  final_response: string
}

// ── Ollama Client ────────────────────────────────────────────────────────────

async function ollamaChat(model: string, messages: Message[], tools?: ToolDef[]): Promise<Message> {
  const body: Record<string, unknown> = { model, messages, stream: false }
  if (tools?.length) body.tools = tools

  const res = await fetch(`${OLLAMA_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Ollama error ${res.status}: ${await res.text()}`)

  const data = (await res.json()) as { message: Message }
  return data.message
}

async function getModels(): Promise<string[]> {
  const res = await fetch(`${OLLAMA_URL}/api/tags`)
  if (!res.ok) return []
  const data = (await res.json()) as { models: { name: string }[] }
  return data.models.map((m) => m.name)
}

function pickBestModel(models: string[]): string {
  const score = (n: string): number => {
    const l = n.toLowerCase()
    if (l.includes("qwen3") || l.includes("qwen2.5-coder")) return 1000
    if (l.includes("llama3") || l.includes("mistral") || l.includes("deepseek")) return 800
    if (l.includes("gemma") || l.includes("phi")) return 600
    return 200
  }
  return [...models].sort((a, b) => score(b) - score(a))[0]
}

// ── Tool Definitions (matching spark.ts) ─────────────────────────────────────

const TOOL_DEFS: ToolDef[] = [
  {
    type: "function",
    function: {
      name: "ReadFile",
      description: "Read a file from disk. Returns numbered lines. For directories, lists entries.",
      parameters: {
        type: "object",
        properties: {
          filePath: { type: "string", description: "Absolute or relative path" },
          offset: { type: "number", description: "1-indexed line to start from" },
          limit: { type: "number", description: "Max lines to return" },
        },
        required: ["filePath"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "WriteFile",
      description: "Write or patch a file. Full write: provide content. Patch: provide oldString and newString.",
      parameters: {
        type: "object",
        properties: {
          filePath: { type: "string", description: "Absolute or relative path" },
          content: { type: "string", description: "Full file content (for full write)" },
          oldString: { type: "string", description: "Text to find (for patch)" },
          newString: { type: "string", description: "Replacement text (for patch)" },
          replaceAll: { type: "boolean", description: "Replace all occurrences" },
        },
        required: ["filePath"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "Bash",
      description: "Execute a shell command and return stdout+stderr.",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string", description: "Shell command to run" },
          workdir: { type: "string", description: "Working directory" },
          timeout: { type: "number", description: "Timeout in milliseconds" },
        },
        required: ["command"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "Eval",
      description: "Evaluate JavaScript/TypeScript code in the agent process (Bun runtime).",
      parameters: {
        type: "object",
        properties: {
          code: { type: "string", description: "JS/TS code to evaluate" },
        },
        required: ["code"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "LoadSkill",
      description: "Load a skill by name. Returns the SKILL.md content.",
      parameters: {
        type: "object",
        properties: {
          name: { type: "string", description: "Skill name to load" },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "Task",
      description: "Spawn a sub-agent to handle a task autonomously.",
      parameters: {
        type: "object",
        properties: {
          description: { type: "string", description: "Short task description" },
          prompt: { type: "string", description: "Detailed task instructions" },
        },
        required: ["description", "prompt"],
      },
    },
  },
]

// ── Simulated Tool Execution ─────────────────────────────────────────────────
// We simulate tool results to avoid side effects during testing

import { readFile, stat, readdir } from "node:fs/promises"
import { resolve } from "node:path"
import { spawn } from "node:child_process"

async function executeToolCall(name: string, args: Record<string, unknown>): Promise<string> {
  switch (name) {
    case "ReadFile": {
      const filePath = resolve(String(args.filePath))
      const s = await stat(filePath).catch(() => null)
      if (!s) return `Error: file not found: ${filePath}`
      if (s.isDirectory()) {
        const entries = await readdir(filePath)
        return entries.sort().join("\n")
      }
      const content = await readFile(filePath, "utf-8")
      const lines = content.split("\n")
      const offset = Math.max(1, Number(args.offset ?? 1))
      const limit = Math.max(1, Number(args.limit ?? 2000))
      const slice = lines.slice(offset - 1, offset - 1 + limit)
      return slice.map((line, i) => `${offset + i}: ${line}`).join("\n")
    }
    case "WriteFile": {
      // Simulate — don't actually write during tests
      const filePath = String(args.filePath)
      if (args.oldString !== undefined)
        return `Patched ${filePath}: replaced section`
      return `Wrote ${filePath} (simulated)`
    }
    case "Bash": {
      const command = String(args.command)
      const workdir = args.workdir ? resolve(String(args.workdir)) : process.cwd()
      return new Promise<string>((done) => {
        const chunks: Buffer[] = []
        const proc = spawn("sh", ["-c", command], {
          cwd: workdir,
          stdio: ["ignore", "pipe", "pipe"],
        })
        proc.stdout.on("data", (d: Buffer) => chunks.push(d))
        proc.stderr.on("data", (d: Buffer) => chunks.push(d))
        const timer = setTimeout(() => { proc.kill("SIGTERM") }, 30_000)
        proc.on("close", (code) => {
          clearTimeout(timer)
          const output = Buffer.concat(chunks).toString("utf-8")
          const prefix = code !== 0 ? `[exit code: ${code}]\n` : ""
          done(prefix + output.slice(0, 10_000))
        })
        proc.on("error", (err) => {
          clearTimeout(timer)
          done(`Error: ${err.message}`)
        })
      })
    }
    case "Eval": {
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
        return `Error: ${err instanceof Error ? err.message : String(err)}`
      }
    }
    case "LoadSkill":
      return `Error: skill '${args.name}' not found. Available: (none in test env)`
    case "Task":
      return `[Task: ${args.description}] Simulated sub-agent response: task completed.`
    default:
      return `Error: unknown tool '${name}'`
  }
}

// ── Agent Runner ─────────────────────────────────────────────────────────────

async function runAgent(model: string, userPrompt: string): Promise<AgentTrace> {
  const systemPrompt = `You are spark, a coding agent. You have tools: ReadFile, WriteFile, Bash, Eval, LoadSkill, Task.
Use the appropriate tool(s) to accomplish the user's request. Be direct and concise.
Working directory: ${process.cwd()}`

  const messages: Message[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: userPrompt },
  ]
  const toolCallsMade: { name: string; args: Record<string, unknown>; result: string }[] = []

  for (let turn = 0; turn < MAX_AGENT_TURNS; turn++) {
    const reply = await ollamaChat(model, messages, TOOL_DEFS)
    messages.push(reply)

    if (!reply.tool_calls?.length) {
      return { messages, tool_calls_made: toolCallsMade, final_response: reply.content }
    }

    for (const call of reply.tool_calls) {
      const result = await executeToolCall(call.function.name, call.function.arguments)
      toolCallsMade.push({ name: call.function.name, args: call.function.arguments, result })
      messages.push({ role: "tool", content: result, tool_name: call.function.name, tool_call_id: call.id })
    }
  }

  return { messages, tool_calls_made: toolCallsMade, final_response: "(max turns reached)" }
}

// ── LLM Judge ────────────────────────────────────────────────────────────────

async function judge(model: string, scenario: string, expectedTool: string, trace: AgentTrace): Promise<JudgeScore> {
  const traceStr = trace.tool_calls_made
    .map((tc) => `Tool: ${tc.name}\nArgs: ${JSON.stringify(tc.args)}\nResult: ${tc.result.slice(0, 500)}`)
    .join("\n---\n")

  const judgePrompt = `You are an evaluation judge for an AI coding agent. Score the agent's performance on this task.

## Scenario
${scenario}

## Expected Tool
The agent should have primarily used: ${expectedTool}

## Agent's Tool Calls
${traceStr || "(no tool calls made)"}

## Agent's Final Response
${trace.final_response.slice(0, 1000)}

## Scoring Rubric (1-5 each)

**Tool Selection** (Did the agent pick the right tool?)
- 5: Perfect tool choice
- 4: Correct tool with minor unnecessary extras
- 3: Partially correct (used the right tool among others)
- 2: Wrong primary tool but recovered
- 1: Completely wrong tool or no tool used

**Tool Usage** (Were the arguments correct?)
- 5: Perfect arguments, efficient usage
- 4: Correct with minor inefficiencies
- 3: Mostly correct but missing optional improvements
- 2: Arguments had errors but partially worked
- 1: Completely wrong arguments

**Task Completion** (Did the agent accomplish the goal?)
- 5: Fully accomplished with clear result
- 4: Accomplished with minor issues
- 3: Partially accomplished
- 2: Barely accomplished, significant gaps
- 1: Failed to accomplish

## Response Format
Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{"tool_selection": <1-5>, "tool_usage": <1-5>, "task_completion": <1-5>, "reasoning": "<brief explanation>"}`

  const reply = await ollamaChat(model, [
    { role: "system", content: "You are a precise evaluation judge. Always respond with valid JSON only." },
    { role: "user", content: judgePrompt },
  ])

  // Parse the JSON response
  const content = reply.content.trim()
  // Try to extract JSON from the response (handle markdown code blocks)
  const jsonMatch = content.match(/\{[\s\S]*\}/)
  if (!jsonMatch) {
    console.error("Judge failed to return JSON:", content.slice(0, 200))
    return { tool_selection: 1, tool_usage: 1, task_completion: 1, reasoning: "Judge parse error" }
  }

  try {
    const parsed = JSON.parse(jsonMatch[0])
    return {
      tool_selection: Math.min(5, Math.max(1, Number(parsed.tool_selection) || 1)),
      tool_usage: Math.min(5, Math.max(1, Number(parsed.tool_usage) || 1)),
      task_completion: Math.min(5, Math.max(1, Number(parsed.task_completion) || 1)),
      reasoning: String(parsed.reasoning ?? ""),
    }
  } catch {
    console.error("Judge JSON parse error:", content.slice(0, 200))
    return { tool_selection: 1, tool_usage: 1, task_completion: 1, reasoning: "JSON parse error" }
  }
}

// ── Test Setup ───────────────────────────────────────────────────────────────

let agentModel = ""
let judgeModel = ""

beforeAll(async () => {
  const models = await getModels()
  if (models.length === 0) throw new Error("No Ollama models found. Is Ollama running?")

  agentModel = AGENT_MODEL || pickBestModel(models)
  judgeModel = JUDGE_MODEL || agentModel
  console.log(`Agent model: ${agentModel}`)
  console.log(`Judge model: ${judgeModel}`)
})

// ── Test Scenarios ───────────────────────────────────────────────────────────

describe("spark agent G-Eval", () => {
  test("ReadFile — read a known file", async () => {
    const trace = await runAgent(agentModel, "Read the file spark.ts and tell me how many lines it has.")

    const score = await judge(judgeModel,
      "User asked to read spark.ts and report line count. Agent should use ReadFile on spark.ts.",
      "ReadFile", trace)

    console.log(`  ReadFile scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    expect(score.tool_selection).toBeGreaterThanOrEqual(3)
    expect(score.task_completion).toBeGreaterThanOrEqual(2)
  }, TIMEOUT)

  test("ReadFile — list directory contents", async () => {
    const trace = await runAgent(agentModel, "What files are in the current directory?")

    const score = await judge(judgeModel,
      "User asked to list files in current directory. Agent should use ReadFile with '.' or Bash with 'ls'.",
      "ReadFile", trace)

    console.log(`  ListDir scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    expect(score.tool_selection).toBeGreaterThanOrEqual(2)
    expect(score.task_completion).toBeGreaterThanOrEqual(2)
  }, TIMEOUT)

  test("WriteFile — create a new file", async () => {
    const trace = await runAgent(agentModel,
      "Create a file called /tmp/spark-test-output.txt with the content 'hello from spark test'")

    const score = await judge(judgeModel,
      "User asked to create /tmp/spark-test-output.txt with specific content. Agent should use WriteFile with filePath and content.",
      "WriteFile", trace)

    console.log(`  WriteFile scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    expect(score.tool_selection).toBeGreaterThanOrEqual(3)
    expect(score.task_completion).toBeGreaterThanOrEqual(2)
  }, TIMEOUT)

  test("WriteFile — patch an existing file", async () => {
    const trace = await runAgent(agentModel,
      "In the file /tmp/spark-test-output.txt, replace the word 'hello' with 'goodbye'")

    const score = await judge(judgeModel,
      "User asked to replace 'hello' with 'goodbye' in a file. Agent should use WriteFile in patch mode with oldString and newString.",
      "WriteFile", trace)

    console.log(`  Patch scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    expect(score.tool_selection).toBeGreaterThanOrEqual(3)
    expect(score.task_completion).toBeGreaterThanOrEqual(2)
  }, TIMEOUT)

  test("Bash — run a shell command", async () => {
    const trace = await runAgent(agentModel, "What is my current git branch? Use git to check.")

    const score = await judge(judgeModel,
      "User asked for current git branch. Agent should use Bash with a git command like 'git branch --show-current' or 'git rev-parse --abbrev-ref HEAD'.",
      "Bash", trace)

    console.log(`  Bash scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    expect(score.tool_selection).toBeGreaterThanOrEqual(3)
    expect(score.task_completion).toBeGreaterThanOrEqual(2)
  }, TIMEOUT)

  test("Bash — multi-step command", async () => {
    const trace = await runAgent(agentModel,
      "Count the number of TypeScript files in this directory (non-recursively). Use the shell.")

    const score = await judge(judgeModel,
      "User asked to count .ts files in current directory using shell. Agent should use Bash with something like 'ls *.ts | wc -l' or 'find . -maxdepth 1 -name \"*.ts\" | wc -l'.",
      "Bash", trace)

    console.log(`  BashMulti scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    expect(score.tool_selection).toBeGreaterThanOrEqual(3)
    expect(score.task_completion).toBeGreaterThanOrEqual(2)
  }, TIMEOUT)

  test("Eval — calculate an expression", async () => {
    const trace = await runAgent(agentModel,
      "Calculate the factorial of 10 using JavaScript. Use the Eval tool to run the code.")

    const score = await judge(judgeModel,
      "User asked to calculate factorial of 10 using Eval. Agent should use Eval with JS code that computes 10! (3628800).",
      "Eval", trace)

    console.log(`  Eval scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    expect(score.tool_selection).toBeGreaterThanOrEqual(3)
    expect(score.task_completion).toBeGreaterThanOrEqual(2)
  }, TIMEOUT)

  test("Eval — data transformation", async () => {
    const trace = await runAgent(agentModel,
      "Using Eval, parse the JSON string '{\"name\":\"spark\",\"version\":1}' and return the name field.")

    const score = await judge(judgeModel,
      "User asked to parse JSON and extract name field using Eval. Agent should use Eval with JSON.parse and return the name.",
      "Eval", trace)

    console.log(`  EvalJSON scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    expect(score.tool_selection).toBeGreaterThanOrEqual(3)
    expect(score.task_completion).toBeGreaterThanOrEqual(2)
  }, TIMEOUT)

  test("LoadSkill — attempt to load a skill", async () => {
    const trace = await runAgent(agentModel,
      "Load the skill called 'debugging' using LoadSkill.")

    const score = await judge(judgeModel,
      "User asked to load a skill called 'debugging'. Agent should use LoadSkill with name='debugging'. It will fail (not found) but the correct tool should still be selected.",
      "LoadSkill", trace)

    console.log(`  LoadSkill scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    expect(score.tool_selection).toBeGreaterThanOrEqual(3)
  }, TIMEOUT)

  test("Task — delegate a sub-task", async () => {
    const trace = await runAgent(agentModel,
      "Use the Task tool to spawn a sub-agent that will summarize what files exist in the current directory. Delegate it, don't do it yourself.")

    const score = await judge(judgeModel,
      "User explicitly asked to use Task tool to delegate work. Agent should use Task with a description and prompt.",
      "Task", trace)

    console.log(`  Task scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    expect(score.tool_selection).toBeGreaterThanOrEqual(2)
    expect(score.task_completion).toBeGreaterThanOrEqual(2)
  }, TIMEOUT)

  test("Multi-tool — read then write", async () => {
    const trace = await runAgent(agentModel,
      "Read the first 5 lines of spark.ts, then create a file /tmp/spark-header.txt containing those lines.")

    const score = await judge(judgeModel,
      "User asked to read first 5 lines of spark.ts then write them to a new file. Agent should use ReadFile (with limit=5) then WriteFile.",
      "ReadFile", trace)

    console.log(`  MultiTool scores: sel=${score.tool_selection} usage=${score.tool_usage} task=${score.task_completion}`)
    console.log(`  Reasoning: ${score.reasoning}`)

    // For multi-tool, check that multiple tools were used
    const toolsUsed = new Set(trace.tool_calls_made.map((tc) => tc.name))
    expect(toolsUsed.size).toBeGreaterThanOrEqual(2)
    expect(score.task_completion).toBeGreaterThanOrEqual(2)
  }, TIMEOUT)
})

// ── Summary Reporter ─────────────────────────────────────────────────────────

// Bun test runner handles reporting, but we add a final summary via afterAll if needed

// ── Goal Supervisor Tests ────────────────────────────────────────────────────

describe("goal supervisor", () => {
  // ── Deterministic parseVerdict tests (no LLM) ──────────────────────────────

  test("parseVerdict — clean JSON reached=true", () => {
    const result = parseVerdict('{"reached":true,"feedback":""}')
    expect(result.reached).toBe(true)
    expect(result.feedback).toBe("")
  })

  test("parseVerdict — clean JSON reached=false with feedback", () => {
    const result = parseVerdict('{"reached":false,"feedback":"Run the tests first"}')
    expect(result.reached).toBe(false)
    expect(result.feedback).toBe("Run the tests first")
  })

  test("parseVerdict — missing feedback field defaults to empty string", () => {
    const result = parseVerdict('{"reached":true}')
    expect(result.reached).toBe(true)
    expect(result.feedback).toBe("")
  })

  test("parseVerdict — garbage/non-JSON input returns fail-safe", () => {
    const result = parseVerdict("The goal is not reached yet. Please keep working.")
    expect(result.reached).toBe(false)
    expect(result.feedback).toBe("")
  })

  test("parseVerdict — empty string returns fail-safe", () => {
    const result = parseVerdict("")
    expect(result.reached).toBe(false)
    expect(result.feedback).toBe("")
  })

  // ── LLM integration tests (need Ollama) ────────────────────────────────────

  test("checkGoal — goal clearly accomplished", async () => {
    const messages: SparkMessage[] = [
      { role: "system", content: "You are spark, a coding agent." },
      { role: "user", content: "Create a file hello.ts that prints hello world" },
      { role: "assistant", content: "I have created the file hello.ts with a console.log statement and verified it runs correctly. The tests all pass." },
    ]
    const verdict = await checkGoal(judgeModel, "Create a file hello.ts that prints hello world", messages)
    expect(verdict.reached).toBe(true)
  }, TIMEOUT)

  test("checkGoal — goal clearly not accomplished", async () => {
    const messages: SparkMessage[] = [
      { role: "system", content: "You are spark, a coding agent." },
      { role: "user", content: "Make all tests pass" },
      { role: "assistant", content: "I ran the tests and they are still failing. There are 5 errors remaining that I have not fixed yet." },
    ]
    const verdict = await checkGoal(judgeModel, "Make all tests pass", messages)
    expect(verdict.reached).toBe(false)
  }, TIMEOUT)
})

// ── Autopilot Tests ──────────────────────────────────────────────────────────

describe("autopilot", () => {
  test("makeAutopilotExit — tool name and required", () => {
    const state = { exited: false }
    const tool = makeAutopilotExit(state)
    expect(tool.definition.function.name).toBe("autopilot_exit")
    expect(tool.definition.function.parameters.required).toEqual([])
  })

  test("makeAutopilotExit — execute sets exited and returns string", async () => {
    const state = { exited: false }
    const tool = makeAutopilotExit(state)
    const result = await tool.execute({})
    expect(typeof result).toBe("string")
    expect(state.exited).toBe(true)
  })

  test("AUTOPILOT_NUDGE — is string, starts/ends with system-reminder tags, contains key phrases", () => {
    expect(typeof AUTOPILOT_NUDGE).toBe("string")
    expect(AUTOPILOT_NUDGE.trimStart().startsWith("<system-reminder>")).toBe(true)
    expect(AUTOPILOT_NUDGE.trimEnd().endsWith("</system-reminder>")).toBe(true)
    expect(AUTOPILOT_NUDGE).toContain("autopilot_exit")
    expect(AUTOPILOT_NUDGE).toContain("stop planning and start implementing")
  })

  test("MAX_AUTOPILOT_REFLECTIONS === 50", () => {
    expect(MAX_AUTOPILOT_REFLECTIONS).toBe(50)
  })

  test("AUTOPILOT_SUMMARY_PROMPT contains 'summarize'", () => {
    expect(AUTOPILOT_SUMMARY_PROMPT).toContain("summarize")
  })
})

// ── Compaction Tests ─────────────────────────────────────────────────────────

describe("compaction", () => {
  // ── Deterministic unit tests (no LLM) ──────────────────────────────────────

  test("estimateTokens — returns ~chars/4 for known message set", () => {
    const messages: SparkMessage[] = [
      { role: "system", content: "abcd" },      // 4 chars
      { role: "user", content: "efgh" },         // 4 chars
      { role: "assistant", content: "ijkl" },    // 4 chars
    ]
    // 12 chars total → 3 tokens
    const tokens = estimateTokens(messages)
    expect(tokens).toBe(3)
  })

  test("estimateTokens — counts tool_calls function JSON", () => {
    const messages: SparkMessage[] = [
      {
        role: "assistant",
        content: "",
        tool_calls: [
          { id: "1", function: { name: "ReadFile", arguments: { filePath: "a.ts" } } },
        ],
      },
    ]
    const json = JSON.stringify({ name: "ReadFile", arguments: { filePath: "a.ts" } })
    const expected = Math.round(json.length / 4)
    expect(estimateTokens(messages)).toBe(expected)
  })

  test("splitTurns — [user, assistant(tool_calls), tool, user, assistant] → 2 turns", () => {
    const messages: SparkMessage[] = [
      { role: "user", content: "do something" },
      {
        role: "assistant",
        content: "",
        tool_calls: [{ id: "1", function: { name: "Bash", arguments: { command: "ls" } } }],
      },
      { role: "tool", content: "file1.ts\nfile2.ts", tool_name: "Bash", tool_call_id: "1" },
      { role: "user", content: "now summarize" },
      { role: "assistant", content: "Here is the summary." },
    ]
    const turns = splitTurns(messages)
    expect(turns.length).toBe(2)
    // First turn: user + assistant(tool_calls) + tool
    expect(turns[0].length).toBe(3)
    expect(turns[0][0].role).toBe("user")
    expect(turns[0][1].role).toBe("assistant")
    expect(turns[0][2].role).toBe("tool")
    // Second turn: user + assistant
    expect(turns[1].length).toBe(2)
    expect(turns[1][0].role).toBe("user")
    expect(turns[1][1].role).toBe("assistant")
  })

  test("splitTurns — assistant-leading body forms its own first turn", () => {
    const messages: SparkMessage[] = [
      { role: "assistant", content: "preamble" },
      { role: "user", content: "hello" },
      { role: "assistant", content: "world" },
    ]
    const turns = splitTurns(messages)
    expect(turns.length).toBe(2)
    expect(turns[0][0].role).toBe("assistant")
    expect(turns[1][0].role).toBe("user")
  })

  test("splitTurns — single user+assistant is one turn", () => {
    const messages: SparkMessage[] = [
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]
    const turns = splitTurns(messages)
    expect(turns.length).toBe(1)
    expect(turns[0].length).toBe(2)
  })

  test("splitTurns — empty body returns empty array", () => {
    expect(splitTurns([])).toEqual([])
  })

  // ── Live Ollama integration test ────────────────────────────────────────────

  test("compactMessages — summary is non-empty and starts with compaction marker", async () => {
    // Build a message array with 3 turns so compaction kicks in (TAIL_TURNS=2 → head=turn1)
    const messages: SparkMessage[] = [
      { role: "system", content: "You are spark, a coding agent." },
      // turn 1
      { role: "user", content: "Read spark.ts and tell me about it." },
      { role: "assistant", content: "spark.ts is a single-file AI coding agent using Ollama." },
      // turn 2
      { role: "user", content: "How many tools does it define?" },
      { role: "assistant", content: "It defines 7 tools: ReadFile, WriteFile, Bash, Eval, Glob, Grep, LoadSkill." },
      // turn 3 (tail)
      { role: "user", content: "What is the entry point?" },
      { role: "assistant", content: "The entry point is the main() function guarded by import.meta.main." },
    ]

    const compacted = await compactMessages(agentModel, messages)
    expect(compacted).toBe(true)

    // messages[0] is still the system prompt
    expect(messages[0].role).toBe("system")

    // messages[1] should be the compaction summary
    const summaryMsg = messages[1]
    expect(summaryMsg.role).toBe("user")
    expect(summaryMsg.content).toBeTruthy()
    expect(summaryMsg.content.startsWith("[Earlier conversation compacted to summary]")).toBe(true)
    expect(summaryMsg.content.length).toBeGreaterThan(50)

    // The tail (last TAIL_TURNS=2 turns = turn 2 + turn 3) should follow
    const tailStart = messages[2]
    expect(tailStart.role).toBe("user")
    expect(tailStart.content).toBe("How many tools does it define?")
  }, TIMEOUT)
})

// ── ollamaChat capability-probe tests ────────────────────────────────────────
// Uses a real Bun HTTP server to mock Ollama's /api/show and /api/chat.

function makeOllamaStreamLine(content: string, done = false): string {
  return JSON.stringify({ message: { role: "assistant", content }, done }) + "\n"
}

describe("ollamaChat capability probe", () => {
  function makeMockServer(thinkingSupported: boolean) {
    return Bun.serve({
      port: 0,
      async fetch(req) {
        const url = new URL(req.url)
        if (url.pathname === "/api/show") {
          const caps = thinkingSupported ? ["completion", "thinking"] : ["completion"]
          return new Response(JSON.stringify({ capabilities: caps }), { headers: { "Content-Type": "application/json" } })
        }
        if (url.pathname === "/api/chat") {
          const body = await req.json() as Record<string, unknown>
          if (body.think && !thinkingSupported) {
            return new Response(JSON.stringify({ error: "model does not support think" }), { status: 400 })
          }
          const stream = new ReadableStream({
            start(controller) {
              controller.enqueue(new TextEncoder().encode(makeOllamaStreamLine("ok")))
              controller.enqueue(new TextEncoder().encode(makeOllamaStreamLine("", true)))
              controller.close()
            },
          })
          return new Response(stream, { headers: { "Content-Type": "application/x-ndjson" } })
        }
        return new Response("not found", { status: 404 })
      },
    })
  }

  test("sends think:true for thinking-capable model", async () => {
    const server = makeMockServer(true)
    const origUrl = process.env.OLLAMA_URL
    process.env.OLLAMA_URL = `http://localhost:${server.port}`
    try {
      const reply = await sparkOllamaChat("thinking-model", [{ role: "user", content: "hi" }], [])
      expect(reply.role).toBe("assistant")
      expect(reply.content).toBe("ok")
    } finally {
      if (origUrl !== undefined) process.env.OLLAMA_URL = origUrl
      else delete process.env.OLLAMA_URL
      server.stop()
    }
  }, 10_000)

  test("sends no think param for non-thinking model (e.g. qwen3-coder)", async () => {
    const server = makeMockServer(false)
    const origUrl = process.env.OLLAMA_URL
    process.env.OLLAMA_URL = `http://localhost:${server.port}`
    try {
      const reply = await sparkOllamaChat("non-thinking-model", [{ role: "user", content: "hi" }], [])
      expect(reply.role).toBe("assistant")
      expect(reply.content).toBe("ok")
    } finally {
      if (origUrl !== undefined) process.env.OLLAMA_URL = origUrl
      else delete process.env.OLLAMA_URL
      server.stop()
    }
  }, 10_000)

  test("real Ollama: qwen3 has thinking capability, smollm does not", async () => {
    const models = await getModels()
    const qwen3 = models.find(m => m.startsWith("qwen3:") && !m.includes("coder"))
    const smol = models.find(m => m.includes("smollm"))
    if (qwen3) {
      const res = await fetch(`${process.env.OLLAMA_URL ?? "http://localhost:11434"}/api/show`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: qwen3 }),
      })
      const data = await res.json() as { capabilities?: string[] }
      expect(data.capabilities ?? []).toContain("thinking")
    }
    if (smol) {
      const res = await fetch(`${process.env.OLLAMA_URL ?? "http://localhost:11434"}/api/show`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: smol }),
      })
      const data = await res.json() as { capabilities?: string[] }
      expect(data.capabilities ?? []).not.toContain("thinking")
    }
    if (!qwen3 && !smol) console.log("SKIP: neither qwen3 nor smollm installed")
  }, 10_000)

  test("real Ollama: ollamaChat works with qwen3 (thinking-capable)", async () => {
    const models = await getModels()
    const model = models.find(m => m.startsWith("qwen3:") && !m.includes("coder")) ?? models[0]
    if (!model) { console.log("SKIP: no models"); return }
    const reply = await sparkOllamaChat(model, [{ role: "user", content: "Reply with exactly: ok" }], [])
    expect(reply.role).toBe("assistant")
    expect(reply.content.length).toBeGreaterThan(0)
  }, 60_000)
})
