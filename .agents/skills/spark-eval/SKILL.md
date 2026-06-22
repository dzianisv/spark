# spark-eval — Evaluation skill for spark.ts goal/autopilot system

Evaluate whether the spark.ts goal supervision system works correctly by running
real coding tasks through the agent with `/goal` enabled, then scoring each result
with an isolated judge subagent. Append results to `eval.csv`. Commit and push.

---

## When to invoke

- After any change to `checkGoal()`, `deriveGoal()`, or the `/autopilot` handler in spark.ts
- Before merging goal/autopilot changes
- As part of the improvement cycle: run → score → fix → repeat until score ≥ 8/10

---

## Eval test set

Five standard tasks. Each has a clear, verifiable done condition. Run ALL five
every eval cycle to get a stable score. Do not skip tasks even if earlier ones fail.

```
TASK_1: "Write a hello world program in TypeScript. Save it to /tmp/spark-eval/hello.ts. Run it with bun and show the output."
DONE_1: File /tmp/spark-eval/hello.ts exists AND `bun /tmp/spark-eval/hello.ts` outputs "Hello, World!" or similar greeting.

TASK_2: "Write a FizzBuzz function in /tmp/spark-eval/fizzbuzz.ts. It should print numbers 1-15, replacing multiples of 3 with Fizz, 5 with Buzz, 15 with FizzBuzz. Run it."
DONE_2: Running the file produces correct FizzBuzz output (1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz).

TASK_3: "Create /tmp/spark-eval/sum.ts with an exported sum(a, b) function. Write a test file sum.test.ts alongside it. Run the tests."
DONE_3: sum.test.ts exists, runs via bun test, and all tests pass.

TASK_4: "Read the first 5 lines of /Users/engineer/workspace/spark/spark.ts and write them to /tmp/spark-eval/header.txt"
DONE_4: /tmp/spark-eval/header.txt exists and contains the first 5 lines of spark.ts.

TASK_5: "Create /tmp/spark-eval/sort.ts with a function that sorts an array of numbers in ascending order. Add at least 3 test cases and run them."
DONE_5: sort.ts and sort.test.ts exist, tests run and pass including edge case for empty array.
```

---

## Steps

### 1. Prepare

```bash
mkdir -p /tmp/spark-eval
git -C /Users/engineer/workspace/spark rev-parse --short HEAD  # capture commit id
```

### 2. Run each task through spark.ts with /goal enabled

For each TASK_N, launch spark.ts in non-interactive mode using stdin pipe.
The `/goal` command sets the done condition before the task message:

```bash
cd /Users/engineer/workspace/spark

# Example for TASK_1:
printf "/goal %s\n%s\nautopilot_exit\n" "DONE_1_CONDITION" "TASK_1_TEXT" | \
  timeout 120 bun spark.ts 2>&1 | tee /tmp/spark-eval/task1.log

# Repeat for tasks 2-5, saving each to task{N}.log
```

**Important:** spark.ts is interactive. Use the `Task` sub-agent approach instead:
spawn a `general-purpose` subagent per task, give it the full spark.ts path,
have it run the agent non-interactively using `bun -e` or direct API calls
to Ollama with the spark system prompt and tools.

### 3. Spawn one judge subagent per task (in parallel)

After all 5 tasks complete, spawn 5 `general-purpose` subagents in parallel.
Each receives:

```
You are an honest, isolated evaluation judge. You have NO prior context.

TASK: <task description>
DONE CONDITION: <done condition>
AGENT LOG:
<contents of task{N}.log>

Score this agent run on 3 dimensions (1-5 each):
- goal_reached: Did the agent achieve the done condition? (5=yes with evidence, 1=no)
- tool_quality: Did it use the right tools effectively? (5=excellent, 1=wrong tools)
- efficiency: Did it complete in reasonable steps? (5=direct, 1=looped/stuck)

Respond ONLY with valid JSON:
{
  "task_id": "TASK_N",
  "goal_reached": <1-5>,
  "tool_quality": <1-5>,
  "efficiency": <1-5>,
  "total": <sum>,
  "reached_boolean": true|false,
  "feedback": "<1-2 sentences: what worked, what failed, concrete next improvement>"
}
```

### 4. Write results to eval.csv

Append one row per task:

```
commit_id, timestamp, task_id, goal_reached, tool_quality, efficiency, total, reached_boolean, feedback
```

CSV header (create file if missing):
```
commit_id,timestamp,task_id,goal_reached,tool_quality,efficiency,total,reached_boolean,feedback
```

Use `eval.csv` in `/Users/engineer/workspace/spark/`.

### 5. Print summary

```
spark-eval results — commit <sha>
────────────────────────────────────
TASK_1  goal=5 tools=4 eff=4 → 13/15  ✓ reached
TASK_2  goal=3 tools=3 eff=2 →  8/15  ✗ not reached
...
────────────────────────────────────
Avg score: 10.4/15   Reached: 4/5
```

### 6. Commit and push

```bash
cd /Users/engineer/workspace/spark
git add eval.csv
git commit -m "eval: spark-eval run $(git rev-parse --short HEAD) — score <avg>/15

Tasks: 5  Reached: N/5  Avg: X.X/15

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push
```

### 7. Improvement cycle

If avg score < 11/15 OR reached < 4/5:

1. Read the feedback column in eval.csv for the failing tasks
2. Identify the root cause — is it:
   - Judge says bare claim "I'm done" → improve goal injection (mandatory evidence rule)
   - Agent loops without progressing → improve escalating feedback  
   - Agent doesn't call tools → check system prompt tool instructions
   - Autopilot exits too early → check `autopilot_exit` tool description
3. Apply the fix to spark.ts
4. Re-run this skill from Step 2
5. Repeat until avg ≥ 11/15 AND reached ≥ 4/5

---

## Known improvement areas (from docs/model.md)

Priority order for fixes when score is low:

1. **Goal injection framing** (highest impact)
   - Current: `Current goal: <text>` appended to system prompt
   - Better: inject as MANDATORY block with evidence rule (see docs/model.md)
   
2. **Escalating feedback**
   - Current: same flat supervisor nudge every check
   - Better: escalate tone at check 3, 5, 10 — add planning-loop detection

3. **Judge schema enrichment**
   - Current: `{"reached": bool, "feedback": string}`
   - Better: `{"status": ..., "stuck": bool, "missing": [], "next_actions": []}`

4. **Planning loop detection**
   - Current: doom loop detection (3× identical tool call)
   - Better: detect read-only tool pattern when writes are expected

---

## Files

| File | Role |
|---|---|
| `eval.ts` | Automated before/after harness (checkGoal fixture comparison + judge) |
| `eval.csv` | Accumulated results across all eval runs |
| `spark.test.ts` | Unit tests (goal supervisor + deriveGoal describe blocks) |
| `docs/model.md` | Reference: CC vs spark.ts design comparison |
| `.agents/skills/spark-eval/SKILL.md` | This file |
