#!/usr/bin/env bash
# Example GSM8K evaluation run using the local verifiers repo.

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required but not found in PATH." >&2
  echo "Install uv by following the instructions in README.md." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-gpt-4.1-mini}"
NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
REPEATS="${REPEATS:-1}"

echo "[run-gsm8k] Installing gsm8k environment from local repo (idempotent)..."
uv run vf-install gsm8k --from-repo

echo "[run-gsm8k] Evaluating gsm8k with model=${MODEL_NAME} examples=${NUM_EXAMPLES} repeats=${REPEATS}"
uv run vf-eval gsm8k \
  -m "${MODEL_NAME}" \
  -n "${NUM_EXAMPLES}" \
  -r "${REPEATS}" \
  "$@"
