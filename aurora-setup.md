# Aurora Setup

## What is Aurora?

[Aurora](https://github.com/aurora-develop/aurora) is an open-source Go reverse-proxy that exposes an **OpenAI-compatible REST API** (`/v1/chat/completions`, `/v1/models`, etc.) backed by your existing ChatGPT web session. Instead of paying for an OpenAI API key you supply your browser's `accessToken` — the same credential ChatGPT's own web frontend uses — and Aurora forwards requests to ChatGPT's internal API, translating responses into the standard OpenAI format any client can consume.

---

## Quick start

```bash
bun aurora-setup.ts
```

The script will:
1. Verify Docker is running.
2. Pull and start the Aurora container (port 8080).
3. Open the ChatGPT session URL in your browser.
4. Prompt you to paste the `accessToken`.
5. Save it to `~/.aurora/access_tokens.txt` and restart Aurora.
6. Poll `http://localhost:8080/v1/models` until Aurora responds.

---

## How to get your ChatGPT access token

1. **Log in to ChatGPT** in your browser at <https://chat.openai.com> (or <https://chatgpt.com>).  
   Make sure you are fully authenticated — if you see the chat UI you're good.

2. **Navigate to the session endpoint.**  
   The script will open this URL automatically; or paste it yourself:
   ```
   https://chatgpt.com/api/auth/session
   ```
   The page returns a JSON blob like:
   ```json
   {
     "user": { "name": "...", "email": "..." },
     "expires": "2026-06-29T...",
     "accessToken": "eyJhbGci..."
   }
   ```
   > **Screenshot hint:** The raw JSON is displayed directly in the browser tab.  
   > In Chrome/Edge you can use *Ctrl+A → Ctrl+C* to copy everything, then paste into a JSON formatter to find `accessToken`.  
   > In Safari the JSON is pretty-printed and clickable.

3. **Copy the `accessToken` value** — the long string that starts with `eyJ`.  
   Do **not** copy the surrounding quotes.

4. **Paste it** when the setup script prompts:
   ```
   Paste your accessToken here and press Enter:
   ```

---

## Connecting Spark to Aurora

> Aurora exposes a standard OpenAI-compatible API, so any code that already talks  
> to OpenAI just needs three environment variables changed.

```bash
SPARK_API_URL=http://localhost:8080 \
SPARK_API_KEY=<your-accessToken> \
bun spark.ts
```

Or read the token from the saved file:

```bash
SPARK_API_URL=http://localhost:8080 \
SPARK_API_KEY=$(head -1 ~/.aurora/access_tokens.txt) \
bun spark.ts
```

| Variable | Value |
|---|---|
| `SPARK_API_URL` | `http://localhost:8080` |
| `SPARK_API_KEY` | your ChatGPT `accessToken` |

Aurora maps the OpenAI model names (e.g. `gpt-4o`) to ChatGPT web models, so  
you can pass any model name the ChatGPT web UI supports.

---

## Token refresh

ChatGPT session tokens expire after **approximately 7 days**. When Aurora starts  
returning `401` errors, re-run the setup script:

```bash
bun aurora-setup.ts
```

The script appends new tokens to `~/.aurora/access_tokens.txt`; Aurora uses the  
most recently added valid token.

To see all stored tokens:

```bash
cat ~/.aurora/access_tokens.txt
```

To wipe and start fresh:

```bash
rm ~/.aurora/access_tokens.txt
bun aurora-setup.ts
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Docker is not running` | Start Docker Desktop, then retry |
| Aurora returns `401` | Token expired — re-run the script |
| Port 8080 in use | `docker stop aurora && docker rm aurora`, change `-p 8082:8080` in the script |
| `docker logs aurora` shows errors | Check Aurora's [GitHub issues](https://github.com/aurora-develop/aurora/issues) |
| Browser doesn't open automatically | Open `https://chatgpt.com/api/auth/session` manually |

---

## Managing the container

```bash
# Stop Aurora
docker stop aurora

# Start Aurora again (no re-setup needed)
docker start aurora

# View logs
docker logs -f aurora

# Remove container entirely (re-run setup to recreate)
docker rm -f aurora
```
