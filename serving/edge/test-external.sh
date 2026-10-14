#!/usr/bin/env bash
# Run on Arch while its Internet connection is mobile tethering, outside home LAN.
set -euo pipefail
read -r -s -p 'API key: ' QWEN36_CLIENT_KEY
printf '\n'
trap 'unset QWEN36_CLIENT_KEY' EXIT
base=https://llm.example.com
printf 'header = "Authorization: Bearer %s"\n' "$QWEN36_CLIENT_KEY" |
  curl --fail-with-body --silent --show-error --max-time 30 --config - "$base/v1/models"
printf '\n'
printf 'header = "Authorization: Bearer %s"\n' "$QWEN36_CLIENT_KEY" |
  curl --fail-with-body --silent --show-error --max-time 180 --config - \
    -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    --data '{"model":"qwen36-35b","messages":[{"role":"user","content":"Ответь по-русски: запрос пришёл через мобильный интернет."}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
printf '\n'
printf 'header = "Authorization: Bearer %s"\n' "$QWEN36_CLIENT_KEY" |
  curl --fail-with-body --silent --show-error --no-buffer --max-time 180 --config - \
    -H 'Content-Type: application/json' "$base/v1/chat/completions" \
    --data '{"model":"qwen36-35b","messages":[{"role":"user","content":"Кратко объясни по-русски потоковую выдачу токенов."}],"stream":true,"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
printf '\n'
