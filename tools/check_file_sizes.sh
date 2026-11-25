#!/usr/bin/env bash
# Enforce source-size limits: Rust files ≤ 400 lines (warn above 300),
# Python files ≤ 300 lines. Function length (≤ 60 lines) is enforced by
# clippy::too_many_lines (see clippy.toml).
set -euo pipefail
cd "$(dirname "$0")/.."
status=0
while IFS= read -r f; do
  n=$(wc -l < "$f")
  if [ "$n" -gt 400 ]; then echo "ERROR: $f has $n lines (> 400)"; status=1
  elif [ "$n" -gt 300 ]; then echo "warning: $f has $n lines (> 300)"; fi
done < <(git ls-files '*.rs')
while IFS= read -r f; do
  n=$(wc -l < "$f")
  if [ "$n" -gt 300 ]; then echo "ERROR: $f has $n lines (> 300)"; status=1; fi
done < <(git ls-files '*.py')
[ "$status" -eq 0 ] && echo "file sizes OK"
exit "$status"
