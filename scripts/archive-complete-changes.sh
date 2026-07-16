#!/usr/bin/env bash
# Архивирует завершённые OpenSpec changes (все tasks = [x], ни одной [ ]).
# Успешно заархивированный change стейджится (git add -A openspec) → входит в
# текущий коммит. Используется и Claude Code hook'ом, и pre-commit'ом.
#
# stdout — контракт для вызывающего: строки "OK:<name>" / "FAILED:<name>".
# stderr — вывод openspec (для сообщения об ошибке).
# exit: 0 — провалов нет; 1 — хотя бы один `openspec archive` упал.
set -uo pipefail

root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
cd "$root" || exit 0

command -v openspec >/dev/null 2>&1 || {
  echo "openspec не найден в PATH — пропуск архивации" >&2
  exit 0
}

changes_dir="openspec/changes"
[ -d "$changes_dir" ] || exit 0

failed=0
for dir in "$changes_dir"/*/; do
  name=$(basename "$dir")
  [ "$name" = "archive" ] && continue
  tasks="$dir/tasks.md"
  [ -f "$tasks" ] || continue

  unchecked=$(grep -cE '^[[:space:]]*- \[ \]' "$tasks" || true)
  checked=$(grep -cE '^[[:space:]]*- \[[xX]\]' "$tasks" || true)
  [ "$((unchecked + checked))" -gt 0 ] || continue  # нет чекбоксов — не тот change
  [ "$unchecked" -eq 0 ] || continue                # есть незавершённые — пропуск

  # Архивация с валидацией (без --no-validate); -y — без интерактивных промптов.
  if openspec archive "$name" -y >&2; then
    git add -A openspec >/dev/null 2>&1 || true
    echo "OK:$name"
  else
    echo "FAILED:$name"
    failed=1
  fi
done

exit "$failed"
