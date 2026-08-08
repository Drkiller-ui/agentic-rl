#!/usr/bin/env bash
# 冻结 200 题的 Rubric（只跑一次，三个 Actor 共用）。
# 三步串联：
#   ① export_evaluation_task_facts.py            → shared/task_facts.jsonl         (.venv-shopsim)
#   ② build_trajectory_evaluation_artifacts.py rubric-candidates → rubric_candidates.jsonl (.venv)
#   ③ run_trajectory_evaluation_models.py    curate-rubrics     → rubrics.jsonl         (.venv + OpenCode)
set -euo pipefail

# 非交互 bash 不读 ~/.bashrc；显式加载 OpenCode env（login shell 由 /etc/profile.d 自动加载）
if [[ -f /etc/profile.d/opencode_llm.sh ]]; then
  source /etc/profile.d/opencode_llm.sh
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOPSIM_ROOT="${SHOPSIM_ROOT:-$ROOT/environments/ShopSimulator/shop_env}"
TASKS_FILE="${TASKS_FILE:-$ROOT/data/evaluation/tasks.jsonl}"
SHARED_DIR="${SHARED_DIR:-$ROOT/shared}"
TASK_FACTS="${TASK_FACTS:-$SHARED_DIR/task_facts.jsonl}"
CANDIDATES="${CANDIDATES:-$SHARED_DIR/rubric_candidates.jsonl}"
RUBRICS="${RUBRICS:-$SHARED_DIR/rubrics.jsonl}"
SHOPSIM_VENV="${SHOPSIM_VENV:-$ROOT/environments/ShopSimulator/.venv-shopsim}"
MAIN_VENV="${MAIN_VENV:-$ROOT/.venv}"
USE_FORCE="${USE_FORCE:---force}"

if [[ ! -x "$SHOPSIM_VENV/bin/python" ]]; then
  echo "fatal: $SHOPSIM_VENV/bin/python not found; run scripts/setup.sh first" >&2
  exit 1
fi
if [[ ! -x "$MAIN_VENV/bin/python" ]]; then
  echo "fatal: $MAIN_VENV/bin/python not found; run scripts/setup.sh first" >&2
  exit 1
fi
if [[ -z "${OPENAI_BASE_URL:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
  echo "warning: OPENAI_BASE_URL / OPENAI_API_KEY not set; curate-rubrics will fail" >&2
fi

mkdir -p "$SHARED_DIR"
cd "$ROOT"

echo "==> ① export TaskFacts (.venv-shopsim)"
"$SHOPSIM_VENV/bin/python" scripts/export_evaluation_task_facts.py \
  --shopsim-root "$SHOPSIM_ROOT" \
  --tasks        "$TASKS_FILE" \
  --output       "$TASK_FACTS" \
  $USE_FORCE

echo "==> ② rubric-candidates (.venv, no LLM)"
"$MAIN_VENV/bin/python" scripts/build_trajectory_evaluation_artifacts.py rubric-candidates \
  --task-facts "$TASK_FACTS" \
  --output     "$CANDIDATES" \
  --allow-blind-final \
  $USE_FORCE

echo "==> ③ curate-rubrics (V4 Flash via OpenCode)"
"$MAIN_VENV/bin/python" scripts/run_trajectory_evaluation_models.py curate-rubrics \
  --task-facts "$TASK_FACTS" \
  --candidates "$CANDIDATES" \
  --output     "$RUBRICS" \
  --allow-blind-final
