#!/usr/bin/env bash
# 对单个 Actor 跑 Pro Judge + 四面板聚合。
# 依赖：① 已跑过 freeze_rubrics.sh；② 已跑过 evaluate.sh <LABEL>。
#
# 四步串联：
#   ① build_trajectory_evaluation_artifacts.py preprocess  → shared/preprocessed/<label>.jsonl
#   ② build_trajectory_evaluation_artifacts.py judge-inputs → shared/judge_requests/<label>.jsonl
#   ③ run_trajectory_evaluation_models.py            judge  → shared/judgments/<label>.jsonl
#   ④ build_trajectory_evaluation_artifacts.py assemble   → outputs/evaluation/<label>/{evaluations.jsonl,evaluation_summary.json}
set -euo pipefail

LABEL="${1:?usage: $0 <label>   (label = baseline | sft | grpo)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_DIR="${SHARED_DIR:-$ROOT/shared}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"
RUBRICS="${RUBRICS:-$SHARED_DIR/rubrics.jsonl}"
TRAJECTORIES="$OUTPUT_DIR/trajectories.jsonl"
PREPROCESSED="${PREPROCESSED:-$SHARED_DIR/preprocessed/$LABEL.jsonl}"
JUDGE_REQUESTS="${JUDGE_REQUESTS:-$SHARED_DIR/judge_requests/$LABEL.jsonl}"
JUDGMENTS="${JUDGMENTS:-$SHARED_DIR/judgments/$LABEL.jsonl}"
ACTOR_META="${ACTOR_META:-$OUTPUT_DIR/summary.json}"
EVALUATIONS="$OUTPUT_DIR/evaluations.jsonl"
EVAL_SUMMARY="$OUTPUT_DIR/evaluation_summary.json"
MAIN_VENV="${MAIN_VENV:-$ROOT/.venv}"
USE_FORCE="${USE_FORCE:---force}"

if [[ ! -x "$MAIN_VENV/bin/python" ]]; then
  echo "fatal: $MAIN_VENV/bin/python not found; run scripts/setup.sh first" >&2
  exit 1
fi
if [[ ! -f "$RUBRICS" ]]; then
  echo "fatal: $RUBRICS not found; run scripts/freeze_rubrics.sh first" >&2
  exit 1
fi
if [[ ! -f "$TRAJECTORIES" ]]; then
  echo "fatal: $TRAJECTORIES not found; run scripts/evaluate.sh $LABEL first" >&2
  exit 1
fi
if [[ -z "${OPENAI_BASE_URL:-}" || -z "${OPENAI_API_KEY:-}" ]]; then
  echo "warning: OPENAI_BASE_URL / OPENAI_API_KEY not set; judge will fail" >&2
fi

mkdir -p "$SHARED_DIR/preprocessed" "$SHARED_DIR/judge_requests" "$SHARED_DIR/judgments"
cd "$ROOT"

echo "==> ① preprocess raw trajectories"
exec "$MAIN_VENV/bin/python" scripts/build_trajectory_evaluation_artifacts.py preprocess \
  --raw    "$TRAJECTORIES" \
  --output "$PREPROCESSED" \
  $USE_FORCE

echo "==> ② build judge-inputs (rubric + actor_visible trajectory)"
exec "$MAIN_VENV/bin/python" scripts/build_trajectory_evaluation_artifacts.py judge-inputs \
  --preprocessed "$PREPROCESSED" \
  --rubrics      "$RUBRICS" \
  --output       "$JUDGE_REQUESTS" \
  $USE_FORCE

echo "==> ③ judge (V4 Pro via OpenCode)"
exec "$MAIN_VENV/bin/python" scripts/run_trajectory_evaluation_models.py judge \
  --requests "$JUDGE_REQUESTS" \
  --output   "$JUDGMENTS"

echo "==> ④ assemble four-panel summary"
exec "$MAIN_VENV/bin/python" scripts/build_trajectory_evaluation_artifacts.py assemble \
  --preprocessed    "$PREPROCESSED" \
  --rubrics         "$RUBRICS" \
  --judges          "$JUDGMENTS" \
  --expected-tasks  "$ROOT/data/evaluation/tasks.jsonl" \
  --actor           "$ACTOR_META" \
  --output          "$EVALUATIONS" \
  --summary         "$EVAL_SUMMARY" \
  $USE_FORCE
