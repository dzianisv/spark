#!/bin/sh
set -e

if [ -n "$CHATGPT_ACCESS_TOKEN" ] && [ -z "$SPARK_API_URL" ]; then
  echo "[spark-sandbox] Starting Aurora (ChatGPT proxy)..."
  mkdir -p "$HOME/.aurora"
  echo "$CHATGPT_ACCESS_TOKEN" > "$HOME/.aurora/access_tokens.txt"

  # Start Aurora in background
  aurora &

  # Wait up to 15s for Aurora to be ready (30 x 0.5s)
  i=0
  until curl -sf http://localhost:8080/v1/models > /dev/null 2>&1; do
    i=$((i+1))
    if [ $i -ge 30 ]; then
      echo "[spark-sandbox] Aurora failed to start within 15s" >&2
      break
    fi
    sleep 0.5
  done

  if curl -sf http://localhost:8080/v1/models > /dev/null 2>&1; then
    echo "[spark-sandbox] Aurora ready at http://localhost:8080"
  fi

  export SPARK_API_URL=http://localhost:8080
  export SPARK_API_KEY=$CHATGPT_ACCESS_TOKEN
fi

exec "$@"
