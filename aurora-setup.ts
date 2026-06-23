#!/usr/bin/env bun
/**
 * Aurora setup script — starts Aurora (ChatGPT web proxy) via Docker,
 * collects the user's ChatGPT access token, and verifies the service is live.
 *
 * Usage: bun aurora-setup.ts
 */

import { existsSync, mkdirSync, appendFileSync } from "fs";
import { homedir, platform } from "os";
import * as readline from "readline";
import * as path from "path";

// ── helpers ──────────────────────────────────────────────────────────────────

function banner(msg: string) {
  console.log(`\n\x1b[1;36m▶ ${msg}\x1b[0m`);
}

function ok(msg: string) {
  console.log(`\x1b[1;32m✔ ${msg}\x1b[0m`);
}

function warn(msg: string) {
  console.log(`\x1b[1;33m⚠ ${msg}\x1b[0m`);
}

function die(msg: string): never {
  console.error(`\x1b[1;31m✖ ${msg}\x1b[0m`);
  process.exit(1);
}

async function run(
  cmd: string[],
  opts: { silent?: boolean; allowFail?: boolean } = {}
): Promise<{ code: number; stdout: string; stderr: string }> {
  const proc = Bun.spawn(cmd, {
    stdout: "pipe",
    stderr: "pipe",
  });
  const [out, err, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  if (!opts.silent && out.trim()) process.stdout.write(out);
  if (!opts.silent && err.trim()) process.stderr.write(err);
  if (!opts.allowFail && code !== 0) {
    die(`Command failed (exit ${code}): ${cmd.join(" ")}`);
  }
  return { code, stdout: out.trim(), stderr: err.trim() };
}

async function prompt(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
  });
  return new Promise((resolve) => {
    process.stdout.write(question);
    rl.once("line", (line) => {
      rl.close();
      resolve(line.trim());
    });
  });
}

// ── 1. Check Docker ───────────────────────────────────────────────────────────

banner("Checking Docker...");
const docker = await run(["docker", "info"], { silent: true, allowFail: true });
if (docker.code !== 0) {
  die("Docker is not running. Start Docker Desktop (or the daemon) and retry.");
}
ok("Docker is running.");

// ── 2. Pull / start Aurora container ─────────────────────────────────────────

banner("Starting Aurora container...");

const IMAGE = "ghcr.io/aurora-develop/aurora:latest";
const CONTAINER = "aurora";
const CONFIG_DIR = path.join(homedir(), ".aurora");

if (!existsSync(CONFIG_DIR)) {
  mkdirSync(CONFIG_DIR, { recursive: true });
  ok(`Created ${CONFIG_DIR}`);
}

// Check if container already exists
const inspect = await run(
  ["docker", "inspect", "--format", "{{.State.Status}}", CONTAINER],
  { silent: true, allowFail: true }
);

if (inspect.code === 0) {
  // Container exists — start it (idempotent if already running)
  warn(`Container '${CONTAINER}' already exists (${inspect.stdout}). Starting it...`);
  await run(["docker", "start", CONTAINER], { silent: true, allowFail: true });
  ok(`Container '${CONTAINER}' started.`);
} else {
  // Pull and create fresh
  console.log(`Pulling ${IMAGE} ...`);
  await run(["docker", "pull", IMAGE]);
  await run([
    "docker", "run", "-d",
    "--name", CONTAINER,
    "-p", "8080:8080",
    "-v", `${CONFIG_DIR}:/app/config`,
    IMAGE,
  ]);
  ok(`Container '${CONTAINER}' created and started.`);
}

// ── 3. Open ChatGPT session URL ───────────────────────────────────────────────

banner("Getting your ChatGPT access token...");

const SESSION_URL = "https://chatgpt.com/api/auth/session";
const os = platform();

console.log(`
┌─────────────────────────────────────────────────────────────┐
│  To get your access token:                                  │
│  1. Make sure you are logged in to ChatGPT in your browser  │
│  2. The following URL will open — it returns JSON           │
│  3. Copy the value of the "accessToken" field               │
│                                                             │
│  ${SESSION_URL}          │
└─────────────────────────────────────────────────────────────┘
`);

if (os === "darwin") {
  await run(["open", SESSION_URL], { silent: true, allowFail: true });
} else if (os === "linux") {
  await run(["xdg-open", SESSION_URL], { silent: true, allowFail: true });
} else {
  warn("Open this URL manually in your browser:");
  console.log(`  ${SESSION_URL}`);
}

// ── 4. Read token from stdin ──────────────────────────────────────────────────

const token = await prompt("\nPaste your accessToken here and press Enter: ");
if (!token) {
  die("No token provided. Aborting.");
}

// ── 5. Write token to ~/.aurora/access_tokens.txt ────────────────────────────

banner("Saving token...");
const TOKEN_FILE = path.join(CONFIG_DIR, "access_tokens.txt");
appendFileSync(TOKEN_FILE, token + "\n", "utf8");
ok(`Token appended to ${TOKEN_FILE}`);

// ── 6. Restart Aurora ────────────────────────────────────────────────────────

banner("Restarting Aurora to pick up the new token...");
await run(["docker", "restart", CONTAINER]);
ok("Aurora restarted.");

// ── 7. Wait for Aurora to respond ────────────────────────────────────────────

banner("Waiting for Aurora to become ready (max 15 s)...");

const HEALTH_URL = "http://localhost:8080/v1/models";
const deadline = Date.now() + 15_000;
let ready = false;

while (Date.now() < deadline) {
  try {
    const res = await fetch(HEALTH_URL, {
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok || res.status === 401) {
      // 401 means Aurora is alive but wants auth — still "running"
      ready = true;
      break;
    }
  } catch {
    // not ready yet
  }
  await Bun.sleep(1000);
  process.stdout.write(".");
}

if (!ready) {
  die("Aurora did not respond within 15 s. Check `docker logs aurora` for errors.");
}
process.stdout.write("\n");
ok("Aurora is responding.");

// ── 8. Final instructions ─────────────────────────────────────────────────────

console.log(`
\x1b[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m
\x1b[1m Aurora running at http://localhost:8080\x1b[0m

 Connect Spark (once SPARK_BACKEND=openai is implemented):

   \x1b[1;33mSPARK_BACKEND=openai \\
   SPARK_API_URL=http://localhost:8080 \\
   SPARK_API_KEY=${token.slice(0, 12)}... \\
   bun spark.ts\x1b[0m

 Tip: your token expires in ~7 days.  Re-run this script to refresh.
\x1b[1;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m
`);
