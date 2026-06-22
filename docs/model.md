# Autopilot & Goal Supervision Model

How spark.ts implements autonomous agent persistence — and how that compares to the reference implementation in Claude Code's agents-supervisor plugin.

---

## Mental Model

Three distinct modes drive agent behavior after each turn:

| Mode | Trigger | Loop driver |
|---|---|---|
| `reflection` | default | one pass per user message |
| `goal` | `/goal <condition>` | loops until judge says condition met |
| `autopilot` | `/autopilot <task>` | loops until `autopilot_exit` tool OR goal met |

**autopilot = goal + extended tool budget + exit tool.**  
The only difference is how the goal is set (user-explicit vs LLM-derived) and that autopilot gives the agent 200 tool rounds instead of 30.

---

## Goal Lifecycle

```
/goal set  ──► goal saved to .agents/spark/goal
              ▼
         injected into system prompt on startup ("Current goal: …")
              ▼
         each turn: agent runs → checkGoal() judge call
              ▼
         reached=true  ──► break supervise loop
         reached=false ──► inject [supervisor] feedback → re-enter loop
```

### `/autopilot` adds:

```
/autopilot <task>
   ▼
deriveGoal(model, messages, existingGoal)   ← LLM synthesizes goal
   ▼
goal saved + counter incremented
   ▼
print "copilot: ● Started autopilot objective #N: <preview>"
   ▼
kick-off message pushed: "[autopilot] Objective #N: <refined goal>"
   ▼
agent loop with autopilot_exit tool available
   ▼
after each tool-round batch: checkGoal() judge call
   ▼
verdict.reached OR autopilot_exit called  ──► stop
```

---

## Reference: Claude Code / agents-supervisor

Source: `~/workspace/agents-supervisor/core/goal.mjs` and `opencode/supervisor-impl.ts`.

### Key differences from spark.ts

| Concern | CC / agents-supervisor | spark.ts |
|---|---|---|
| Goal persistence | `.supervisor/goals/<sessionId>.json` (mode 0600) | `.agents/spark/goal` (plain text) |
| Max attempts | default 16, configurable via `maxAttempts:` in supervisor.yaml | removed cap — runs until done or interrupted |
| Deadline | 30 min hard deadline per goal | none — relies on Esc/Ctrl+C |
| Mode state | 3 modes: `reflection`, `goal`, `autopilot` | 2 states: `autopilot` bool + `goal` string |
| Goal injection | `buildGoalRequirementSection()` — structured "MANDATORY" block with evidence rule | appended to system prompt as plain text |
| Judge prompt | self-assessment JSON (status, stuck, missing, next_actions) + escalating feedback | simple `{"reached": bool, "feedback": string}` |
| Planning loop detection | yes — detects read-only tool calls when writes expected | no |
| Action loop detection | yes — detects repeated identical tool calls | yes (doom loop: 3× identical → nudge) |
| Feedback escalation | severity-aware, escalates tone per attempt | flat — same supervisor nudge every time |

### CC's goal requirement injection (key insight)

CC injects a **mandatory evidence rule** alongside the goal condition:

```
## GOAL (mandatory completion requirement)

MANDATORY: The following user-set goal condition MUST be demonstrably met
before the task can be considered complete:

  "<condition>"

Evidence rule: a claim that this goal condition is met must be backed by
evidence already surfaced in the transcript — commands run and their output,
tests actually executed, artefacts produced. A bare assertion that the
condition is satisfied does NOT count as evidence.
```

spark.ts currently injects goal as `Current goal: <text>` — no mandatory framing, no evidence rule. This is why the judge sometimes accepts bare claims.

### CC's judge verdict schema

```json
{
  "status": "complete | in_progress | waiting_for_user",
  "stuck": false,
  "feedback": "...",
  "missing": ["..."],
  "next_actions": ["..."],
  "needs_user_action": ["..."]
}
```

spark.ts uses a simpler schema:
```json
{ "reached": true, "feedback": "..." }
```

### CC's self-assessment rubric (antipatterns mined from 227 real stops)

The CC supervisor catches these premature-stop patterns (78% of real stops were premature):

1. **PERMISSION-SEEKING** (~40%): agent asks "Want me to…?" instead of just doing it
2. **STOPPED-WITH-TODOS** (~30%): agent lists remaining work then stops without doing it  
3. **VERIFICATION-DEFERRAL**: agent wrote code but didn't run tests
4. **FALSE-COMPLETE**: claims done but core action never happened, no evidence

spark.ts has no equivalent antipattern detection. The AUTOPILOT_NUDGE message covers some of this but not systematically.

### CC's escalating feedback

```
attempt 1-N: "Task Incomplete (severity)\n<feedback>\n### Missing\n...\n### Next Actions\n..."
planning loop: "STOP: Planning Loop Detected — you've only read files, start writing"
action loop:   "STOP: Action Loop Detected — repeating same commands, change approach"
```

spark.ts uses a flat nudge. No escalation.

---

## What spark.ts Does Well

- **Simpler surface area** — 2 states vs 3 modes, one JSON file vs per-session store
- **LLM-derived goal** (`deriveGoal`) — CC requires user to write the goal; spark synthesizes it from conversation
- **Objective counter + preview** — `copilot: ● Started autopilot objective #N: <preview>` is more ergonomic than CC's slash commands
- **No deadline** — for local Ollama this is correct; a 30-min deadline would abort long tasks

## What should be improved

- [ ] Inject goal as MANDATORY block with evidence rule (not plain text)
- [ ] Add escalating feedback per supervisor check count
- [ ] Detect planning loops (read-only tools when writes expected)
- [ ] Richer judge schema: `status`, `stuck`, `missing[]`, `next_actions[]`
- [ ] Soft attempt warning at N checks (e.g. print warning at 5, 10, 20) instead of hard cap

---

## Files

| File | Role |
|---|---|
| `spark.ts` L1108 | `GOAL_FILE` — persistence path `.agents/spark/goal` |
| `spark.ts` L1125 | `AUTOPILOT_COUNT_FILE` — objective counter |
| `spark.ts` L1175 | `deriveGoal()` — LLM synthesizes/refines goal |
| `spark.ts` L1215 | `checkGoal()` — judge call (last user + last AI msg) |
| `spark.ts` L1166 | `parseVerdict()` — JSON parser with fail-safe |
| `spark.ts` L1453 | `/autopilot` handler — derive, save, print, kick off |
| `spark.ts` L1633 | `supervise:` loop — drives goal checks after each turn |
