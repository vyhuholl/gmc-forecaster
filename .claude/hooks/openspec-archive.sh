#!/usr/bin/env bash
# Claude Code PreToolUse (Bash). Перед `git commit` архивирует завершённые
# OpenSpec changes через core-скрипт. Если `openspec archive` упал —
# БЛОКИРУЕТ коммит и просит Claude вызвать скилл opsx:archive, затем повторить
# коммит (архив входит в тот же коммит — core делает git add -A openspec).
set -uo pipefail

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null || true)

# Реагируем только на git commit; иначе — молча пропускаем.
printf '%s' "$cmd" | grep -Eq 'git[[:space:]].*commit' || exit 0

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
core="$root/scripts/archive-complete-changes.sh"
[ -x "$core" ] || exit 0

result=$("$core" 2>&1)
status=$?

# Успех (или архивировать нечего) → разрешаем коммит; архив уже застейджен.
[ "$status" -eq 0 ] && exit 0

# Провал openspec archive → блокируем и эскалируем в скилл.
failed=$(printf '%s\n' "$result" | sed -n 's/^FAILED://p' | paste -sd', ' -)
reason="OpenSpec: change(s) [$failed] завершены, но 'openspec archive' упал:

$result

Заархивируй через скилл Claude Code: вызови opsx:archive для каждого — $failed — \
затем повтори git commit. Архивация должна войти в этот же коммит \
(git add -A openspec)."

jq -n --arg r "$reason" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: $r
  }
}'
exit 0
