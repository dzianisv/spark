#!/usr/bin/env bun
/**
 * eval.ts — goal supervision evaluation harness for spark.ts
 *
 * PURPOSE
 * -------
 * Measures whether changes to the goal/autopilot supervisor improve or
 * regress the agent's ability to correctly detect task completion.
 *
 * WHAT IT DOES (3 phases)
 * -----------------------
 * 1. Unit test suite — runs spark.test.ts "goal supervisor" + "deriveGoal"
 *    describe blocks via `bun test`. Counts pass/fail.
 *
 * 2. Before/after fixture comparison — runs each FIXTURE through two impls:
 *    - "before": checkGoalBefore() — original impl inlined verbatim
 *      (last 6 msgs × 300 chars, "Recent transcript:" framing).
 *    - "after":  checkGoal() from spark.ts — current impl
 *      (last user msg + last AI msg only, tighter context).
 *    Correctness = (judged reached) === (expectedReached).
 *
 * 3. Judge sub-agent — isolated LLM call with system "You are a precise
 *    code quality judge". Sees before/after scores + test output tail.
 *    Returns { score: 1–10, feedback: string }. Never sees spark.ts source.
 *
 * OUTPUT: eval.csv
 * ----------------
 * Appends one row per run:
 *   commit_id, timestamp, score, pass, fail, test_suite, feedback
 *
 *   commit_id  — git rev-parse --short HEAD
 *   timestamp  — ISO-8601 UTC
 *   score      — judge sub-agent score (1–10)
 *   pass       — unit test passes + correct fixture verdicts
 *   fail       — unit test failures + wrong fixture verdicts
 *   test_suite — "goal_supervisor_before_after"
 *   feedback   — judge 2–3 sentence assessment
 *
 * EXIT CODE
 * ---------
 *   0 — all unit tests pass AND all fixtures correct
 *   1 — any unit test failure OR any fixture wrong
 *
 * FIXTURES (FIXTURES array below)
 * --------------------------------
 * Synthetic conversation snapshots. Each has: messages[], goal, expectedReached.
 * Designed to cover specific judge failure modes:
 *
 *   f1  reached=true  — clear done with test output as evidence
 *   f2  reached=false — explicit failure output in AI message
 *   f3  reached=true  — [supervisor] nudge is last user msg; AI confirms done
 *   f4  reached=false — first AI says done, LAST AI says still failing
 *                       (tests that judge uses LAST message, not first)
 *   f5  reached=false — agent only planning, no actual work done
 *   f6  reached=true  — long failure history; last exchange shows success
 *                       (tests that noisy history doesn't confuse the judge)
 *
 * JUDGE SUB-AGENT DESIGN
 * ----------------------
 * judgeSubagent() is intentionally isolated:
 *   - separate system prompt (not the agent's)
 *   - sees only scores, fixture table, test output tail
 *   - does NOT see spark.ts source or full conversation history
 * This mirrors the Claude Code stop-hook pattern: judge is seeded with
 * task + latest reply only, never the live conversation context.
 *
 * MODEL SELECTION
 * ---------------
 * Override: JUDGE_MODEL or AGENT_MODEL env var.
 * Auto-pick preference order (for 8GB RAM machines):
 *   qwen3.5 (6.3GB) > qwen3:8b > qwen2.5 > qwen3:0.6b (too small for judging)
 *   qwen3:14b (8.8GB) is skipped — exceeds 8GB, causes swap thrash.
 *
 * IMPROVEMENT CYCLE
 * -----------------
 *   1. Edit spark.ts (goal/autopilot/deriveGoal/checkGoal)
 *   2. Run: bun eval.ts
 *   3. Check eval.csv for score trend and feedback
 *   4. Fix regressions, iterate
 *
 * USAGE
 * -----
 *   bun eval.ts
 *   JUDGE_MODEL=qwen3.5:latest bun eval.ts
 */

import { appendFile, readFile, writeFile } from "node:fs/promises"
import { exec } from "node:child_process"
import { promisify } from "node:util"
import { checkGoal, deriveGoal } from "./spark.ts"
import type { Message } from "./spark.ts"

const execAsync = promisify(exec)

const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://localhost:11434"
const MODEL     = process.env.JUDGE_MODEL ?? process.env.AGENT_MODEL ?? ""
const CSV_FILE  = "eval.csv"
const CSV_HEADER = "commit_id,timestamp,score,pass,fail,test_suite,feedback\n"

// ── Helpers ───────────────────────────────────────────────────────────────────

async function getModels(): Promise<string[]> {
  const res = await fetch(`${OLLAMA_URL}/api/tags`)
  if (!res.ok) return []
  const d = await res.json() as { models: { name: string }[] }
  return d.models.map(m => m.name)
}

function pickModel(models: string[]): string {
  const score = (n: string) => {
    const l = n.toLowerCase()
    if (l.includes("qwen3.5")) return 3000     // fits 8GB, best reasoning
    if (l.includes("qwen3") && !l.includes("14b") && !l.includes("0.6")) return 2000
    if (l.includes("qwen2.5") || l.includes("qwen:7b")) return 1500
    if (l.includes("llama3") || l.includes("mistral")) return 1000
    if (l.includes("qwen3:0.6")) return 200    // too small for judging
    return 400
  }
  return [...models].sort((a, b) => score(b) - score(a))[0] ?? ""
}

async function ollamaChat(model: string, messages: { role: string; content: string }[]): Promise<string> {
  const ac = new AbortController()
  const timer = setTimeout(() => ac.abort(), 120_000) // 2-min timeout per judge call
  try {
    const res = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages, stream: false }),
      signal: ac.signal,
    })
    if (!res.ok) throw new Error(`Ollama ${res.status}: ${await res.text()}`)
    const d = await res.json() as { message: { content: string } }
    return (d.message.content ?? "").trim()
  } finally {
    clearTimeout(timer)
  }
}

async function getCommitId(): Promise<string> {
  try { return (await execAsync("git rev-parse --short HEAD")).stdout.trim() }
  catch { return "unknown" }
}

async function ensureCsvHeader(): Promise<void> {
  try {
    const content = await readFile(CSV_FILE, "utf-8")
    if (!content.startsWith("commit_id")) {
      await writeFile(CSV_FILE, CSV_HEADER + content)
    }
  } catch {
    await writeFile(CSV_FILE, CSV_HEADER)
  }
}

function csvEscape(s: string): string {
  return `"${s.replace(/"/g, '""').replace(/\n/g, " ")}"`
}

// ── Old checkGoal (before our changes) ───────────────────────────────────────
// Inlined verbatim from the pre-change implementation so we can compare it
// against the current one without touching git history.
//
// What it did:
//   - sliced last 6 messages, truncated each to 300 chars
//   - concatenated as "role: content" pairs under "Recent transcript:" heading
//   - included lastAssistant separately under "The agent's final message:"
//
// Problem: with noisy history (many supervisor nudges, failed attempts) the
// judge sees mostly injected [supervisor] messages, not the real signal.
// Also, 300-char truncation often cuts off the test output that proves done.

interface GoalVerdict { reached: boolean; feedback: string }

function parseVerdict(text: string): GoalVerdict {
  try {
    const parsed = JSON.parse(text)
    return { reached: Boolean(parsed.reached), feedback: String(parsed.feedback ?? "") }
  } catch {
    return { reached: false, feedback: "" }
  }
}

async function checkGoalBefore(model: string, goal: string, messages: Message[]): Promise<GoalVerdict> {
  const lastAssistant = [...messages].reverse().find(m => m.role === "assistant")?.content ?? ""
  const tail = messages.slice(-6).map(m => {
    const c = typeof m.content === "string" ? m.content.slice(0, 300) : ""
    return `${m.role}: ${c}`
  }).join("\n")

  const judgeSystem = "You are a strict goal supervisor for a coding agent. You judge whether a GOAL is fully accomplished based on the conversation. Be skeptical of unverified claims."
  const judgeUser = `GOAL: ${goal}\n\nThe agent's final message:\n${lastAssistant}\n\nRecent transcript:\n${tail}\n\nIs the GOAL fully reached? Respond with ONLY a JSON object: {"reached": true|false, "feedback": "<if not reached, one concrete next action to push the agent forward; empty string if reached>"}`

  const reply = await ollamaChat(model, [
    { role: "system", content: judgeSystem },
    { role: "user", content: judgeUser },
  ])
  return parseVerdict(reply)
}

// ── Test fixtures ─────────────────────────────────────────────────────────────
// Static conversation snapshots used to probe judge correctness.
//
// Design rules:
//   - Each fixture must have an unambiguous expectedReached verdict
//   - The evidence in messages[] must be clear enough for any small LLM to judge
//   - At least one "reached=true" fixture must include explicit test output
//     (bare claim "I'm done" is insufficient — mirrors CC's evidence rule)
//   - f4 and f6 specifically test that the judge uses the LAST AI message,
//     not the first — this is the key behavioral difference between old and new
//
// When adding new fixtures: keep them independent (no shared state), make the
// ground truth obvious from message content alone, and document what failure
// mode they probe in the description field.

interface Fixture {
  id: string
  description: string
  goal: string
  messages: Message[]
  expectedReached: boolean
}

const FIXTURES: Fixture[] = [
  {
    id: "f1-clear-done",
    description: "Last AI clearly done with verification output shown",
    goal: "Write a sum function with passing tests",
    expectedReached: true,
    messages: [
      { role: "system", content: "You are spark." },
      { role: "user", content: "Write a sum function" },
      { role: "assistant", content: "I tried but failed initially." },
      { role: "user", content: "[supervisor] Goal not yet reached. Keep working." },
      { role: "assistant", content: `Done. sum() written to math.ts. Test run output:\n\`\`\`\n✓ sum(1, 2) equals 3\n✓ sum(-1, 1) equals 0\n✓ sum(0, 0) equals 0\n3 pass  0 fail\n\`\`\`\nAll tests pass. Task complete.` },
    ],
  },
  {
    id: "f2-clearly-broken",
    description: "Test output explicitly shows failures",
    goal: "Fix the failing CI build",
    expectedReached: false,
    messages: [
      { role: "system", content: "You are spark." },
      { role: "user", content: "Fix the failing CI build" },
      { role: "assistant", content: "Test output: 5 FAILED, 0 passed. Errors: Cannot find module './auth'. I have not started fixing yet." },
    ],
  },
  {
    id: "f3-supervisor-nudge-success",
    description: "Last user is supervisor nudge; last AI confirms done",
    goal: "Create a README.md file",
    expectedReached: true,
    messages: [
      { role: "system", content: "You are spark." },
      { role: "user", content: "Create a README.md file" },
      { role: "assistant", content: "Created README.md." },
      { role: "user", content: "[supervisor] Goal not yet reached. Verify the file actually exists on disk." },
      { role: "assistant", content: "Verified: README.md exists on disk, 42 lines, content correct. Task fully complete." },
    ],
  },
  {
    id: "f4-multi-turn-last-matters",
    description: "First assistant says done but last says still failing — judge must use LAST",
    goal: "Make all tests pass",
    expectedReached: false,
    messages: [
      { role: "system", content: "You are spark." },
      { role: "user", content: "Make all tests pass" },
      { role: "assistant", content: "I have completed the task and all tests pass." }, // first — wrong
      { role: "user", content: "[supervisor] Goal not yet reached. Run the tests again." },
      { role: "assistant", content: "Re-ran: 3 FAILED. The sort is broken on empty arrays. Not done yet." }, // last — correct
    ],
  },
  {
    id: "f5-planning-not-done",
    description: "Agent is planning, hasn't done anything yet",
    goal: "Write tests for the parser module",
    expectedReached: false,
    messages: [
      { role: "system", content: "You are spark." },
      { role: "user", content: "Write tests for the parser module" },
      { role: "assistant", content: "I'm still analyzing the test failures. I can see 3 errors. Let me start fixing them one by one." },
    ],
  },
  {
    id: "f6-deep-history-noise",
    description: "Long history of failures, last 2 msgs show success — new impl should see it; old may get confused",
    goal: "Refactor auth module to use JWT",
    expectedReached: true,
    messages: [
      { role: "system", content: "You are spark." },
      { role: "user", content: "Refactor auth module to use JWT" },
      { role: "assistant", content: "Attempt 1 failed: cannot find jsonwebtoken." },
      { role: "user", content: "[supervisor] Keep working." },
      { role: "assistant", content: "Attempt 2 failed: type mismatch on token payload." },
      { role: "user", content: "[supervisor] Keep working." },
      { role: "assistant", content: "Attempt 3 failed: tests still broken." },
      { role: "user", content: "[supervisor] Keep working." },
      { role: "assistant", content: "JWT refactor complete. All 8 auth tests pass. Token signing and verification confirmed. Done." },
    ],
  },
]

// ── Run before/after comparison ───────────────────────────────────────────────
// Runs each fixture through both impls in parallel (Promise.all) to save time.
// Prints a live result line per fixture so you can watch progress.
// Returns FixtureResult[] — one entry per fixture with correctness booleans.

interface FixtureResult {
  id: string
  description: string
  expectedReached: boolean
  beforeReached: boolean
  afterReached: boolean
  beforeCorrect: boolean
  afterCorrect: boolean
}

async function runComparison(model: string): Promise<FixtureResult[]> {
  const results: FixtureResult[] = []
  for (const f of FIXTURES) {
    process.stdout.write(`  [${f.id}] ${f.description}… `)
    const [before, after] = await Promise.all([
      checkGoalBefore(model, f.goal, f.messages),
      checkGoal(model, f.goal, f.messages),
    ])
    const r: FixtureResult = {
      id: f.id,
      description: f.description,
      expectedReached: f.expectedReached,
      beforeReached: before.reached,
      afterReached: after.reached,
      beforeCorrect: before.reached === f.expectedReached,
      afterCorrect: after.reached === f.expectedReached,
    }
    results.push(r)
    const bMark = r.beforeCorrect ? "✓" : "✗"
    const aMark = r.afterCorrect  ? "✓" : "✗"
    console.log(`before=${bMark}(${before.reached}) after=${aMark}(${after.reached}) expected=${f.expectedReached}`)
  }
  return results
}

// ── Judge sub-agent ───────────────────────────────────────────────────────────
// An isolated LLM call that evaluates the overall quality of the changes.
//
// "Sub-agent" means: fresh system prompt, no shared state with the eval run.
// It sees the fixture score table + test output tail, not spark.ts source.
// This mirrors the Claude Code stop-hook judge pattern (blog post:
// claude-code-stop-hook-reflection-judge.md): judge is seeded with task +
// latest reply only, seeded fresh, context never accumulates.
//
// Scoring rubric given to the judge:
//   1–4: regression or no meaningful change
//   5–6: marginal improvement, notable risks
//   7–8: clear improvement, minor risks remain
//   9–10: strong improvement, well tested, risks addressed
//
// The judge's feedback field is stored verbatim in eval.csv so you can read
// the trend across commits without re-running the judge.

async function judgeSubagent(
  model: string,
  results: FixtureResult[],
  bpassCount: number,
  apassCount: number,
  testRunOutput: string,
): Promise<{ score: number; feedback: string }> {
  const table = results.map(r =>
    `${r.id}: expected=${r.expectedReached} before=${r.beforeReached}(${r.beforeCorrect ? "✓" : "✗"}) after=${r.afterReached}(${r.afterCorrect ? "✓" : "✗"})`
  ).join("\n")

  const prompt = `You are a code quality judge (sub-agent) evaluating recent changes to spark.ts goal supervision.

## Changes evaluated
1. \`checkGoal()\`: narrowed context from "last 6 msgs × 300 chars" → "last user msg + last AI msg only"
2. Removed MAX_GOAL_CHECKS=10 cap (supervisor runs until done or interrupted)
3. Added \`deriveGoal()\`: LLM derives/refines goal when /autopilot starts
4. /autopilot prints: "copilot: ● Started autopilot objective #N: <preview>"

## Before/after fixture comparison (${FIXTURES.length} cases)
Before score: ${bpassCount}/${FIXTURES.length} correct
After score:  ${apassCount}/${FIXTURES.length} correct

${table}

## Automated test suite (goal supervisor + deriveGoal tests)
${testRunOutput.slice(-2000)}

## Score 1-10
Consider: correctness improvement, design soundness, risk of removing the cap, test coverage quality.

Respond with ONLY valid JSON:
{"score": <1-10>, "feedback": "<2-3 sentences of concrete feedback on what improved, what risks remain, what to test next>"}`

  const reply = await ollamaChat(model, [
    { role: "system", content: "You are a precise code quality judge. Respond with valid JSON only." },
    { role: "user", content: prompt },
  ])

  const jsonMatch = reply.match(/\{[\s\S]*\}/)
  if (!jsonMatch) return { score: 0, feedback: "judge parse error: " + reply.slice(0, 100) }
  try {
    const p = JSON.parse(jsonMatch[0])
    return {
      score: Math.min(10, Math.max(0, Number(p.score) || 0)),
      feedback: String(p.feedback ?? "").replace(/\n/g, " "),
    }
  } catch {
    return { score: 0, feedback: "json parse error" }
  }
}

// ── Run bun test suite ────────────────────────────────────────────────────────
// Runs only the "goal supervisor" and "deriveGoal" describe blocks — the ones
// directly testing the code changed by the goal/autopilot refactor.
// Full suite (~90 tests) is too slow for the eval loop; these 23 tests run in
// ~90s with qwen3:0.6b and cover all the behavioral changes.
//
// Output is captured and passed to judgeSubagent() (tail of last 2000 chars)
// so the judge can see which specific tests passed/failed.
// Exit code from bun test is intentionally not propagated here — we aggregate
// failures at the main() level alongside fixture results.

async function runTestSuite(): Promise<{ output: string; pass: number; fail: number }> {
  try {
    const result = await execAsync(
      `bun test spark.test.ts --test-name-pattern "goal supervisor|deriveGoal|buildGoalBlock|buildSupervisorFeedback" 2>&1`,
      { cwd: process.cwd(), timeout: 300_000 },
    )
    const output = result.stdout + result.stderr
    const pass = parseInt(output.match(/(\d+) pass/)?.[1] ?? "0", 10)
    const fail = parseInt(output.match(/(\d+) fail/)?.[1] ?? "0", 10)
    return { output, pass, fail }
  } catch (err: unknown) {
    const output = err instanceof Error && "stdout" in err
      ? String((err as Record<string, unknown>).stdout)
      : String(err)
    const pass = parseInt(output.match(/(\d+) pass/)?.[1] ?? "0", 10)
    const fail = parseInt(output.match(/(\d+) fail/)?.[1] ?? "0", 10)
    return { output, pass, fail }
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log("spark eval — goal supervision before/after assessment\n")

  const commitId = await getCommitId()
  const models   = await getModels()
  const model    = MODEL || pickModel(models)
  if (!model) { console.error("No Ollama models found."); process.exit(1) }

  console.log(`commit:      ${commitId}`)
  console.log(`judge model: ${model}`)
  console.log(`fixtures:    ${FIXTURES.length}`)

  // 1. Run automated test suite
  console.log("\n⏳ running bun test suite…")
  const { output: testOutput, pass: testPass, fail: testFail } = await runTestSuite()
  console.log(`   ${testPass} pass  ${testFail} fail`)

  // 2. Run before/after fixture comparison
  console.log("\n⚖️  before/after fixture comparison:")
  const results = await runComparison(model)
  const beforePass = results.filter(r => r.beforeCorrect).length
  const afterPass  = results.filter(r => r.afterCorrect).length
  console.log(`\n   before: ${beforePass}/${FIXTURES.length}  after: ${afterPass}/${FIXTURES.length}`)

  // 3. Judge sub-agent
  console.log("\n🤖 judge sub-agent evaluating…")
  const { score, feedback } = await judgeSubagent(model, results, beforePass, afterPass, testOutput)
  console.log(`   score:    ${score}/10`)
  console.log(`   feedback: ${feedback}`)

  // 4. Append to eval.csv
  await ensureCsvHeader()
  const timestamp = new Date().toISOString()
  const suiteName = `goal_supervisor_before_after`
  const totalPass = testPass + afterPass
  const totalFail = testFail + (FIXTURES.length - afterPass)
  const row = [commitId, timestamp, score, totalPass, totalFail, csvEscape(suiteName), csvEscape(feedback)].join(",") + "\n"
  await appendFile(CSV_FILE, row)

  // 5. Print current CSV
  console.log(`\n📊 eval.csv:`)
  console.log(await readFile(CSV_FILE, "utf-8"))

  process.exit(testFail > 0 || afterPass < FIXTURES.length ? 1 : 0)
}

main().catch(err => {
  console.error("eval error:", err)
  process.exit(1)
})
