#!/bin/sh
set -e

if [ -n "$CHATGPT_ACCESS_TOKEN" ]; then
  mkdir -p "$HOME/.aurora"
  echo "$CHATGPT_ACCESS_TOKEN" > "$HOME/.aurora/access_tokens.txt"
  aurora &
  i=0
  until curl -sf http://localhost:8080/v1/models > /dev/null 2>&1; do
    i=$((i+1))
    [ $i -ge 30 ] && echo "[sandbox] Aurora failed to start" >&2 && break
    sleep 0.5
  done
  echo "[sandbox] Aurora ready"
  export SPARK_API_URL=http://localhost:8080
  export SPARK_API_KEY=$CHATGPT_ACCESS_TOKEN
fi

exec "$@"
