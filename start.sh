#!/usr/bin/env bash
# Launch wordsearch-bot (creates the venv on first run).
set -e
cd "$(dirname "$0")"
exec uv run wsbot "$@"
