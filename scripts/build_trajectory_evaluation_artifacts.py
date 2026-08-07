#!/usr/bin/env python3
"""Build offline trajectory-evaluation artifacts without calling models or envs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    ArtifactError,
    atomic_jsonl_writer,
    index_jsonl,
    iter_jsonl,
    load_json,
    write_json_atomic,
)
from shopping_grpo.evaluation.blind_guard import guard_blind_final
from shopping_grpo.evaluation.comparison import compare_evaluation_runs
from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.model_client import DEFAULT_PRO_MODEL
from shopping_grpo.evaluation.prompts import (
    TRAJECTORY_JUDGE_PROMPT_VERSION,
    build_trajectory_judge_messages,
)
from shopping_grpo.evaluation.results import (
    assemble_task_evaluation,
    build_not_judged_result,
    summarize_evaluations,
)
from shopping_grpo.evaluation.rubric import (
    extract_rubric_candidates,
    stable_hash,
)
from shopping_grpo.evaluation.trajectory import normalize_trajectory


PREPROCESSED_SCHEMA = "shopping-preprocessed-trajectory-v1"
JUDGE_REQUEST_SCHEMA = "shopping-judge-request-v2"


def _guard_blind_final(paths: Iterable[Path], *, allowed: bool) -> None:
    guard_blind_final(paths, allowed=allowed)


def _task_ids(path: Path) -> list[int]:
    result = []
    seen = set()
    for row_number, row in enumerate(iter_jsonl(path), start=1):
        if "task_id" not in row:
            raise ArtifactError(f"{path}:{row_number}: missing task_id")
        task_id = int(row["task_id"])
        if task_id in seen:
            raise ArtifactError(
                f"{path}:{row_number}: duplicate task_id={task_id}"
            )
        seen.add(task_id)
        result.append(task_id)
    return result


def preprocess(args: argparse.Namespace) -> None:
    _guard_blind_final(
        [args.raw],
        allowed=args.allow_blind_final,
    )
    written = 0
    seen_trajectories = set()
    with atomic_jsonl_writer(args.output, force=args.force) as write:
        for raw in iter_jsonl(args.raw):
            if args.limit is not None and written >= args.limit:
                break
            normalized = normalize_trajectory(raw)
            trajectory_id = normalized["trajectory_id"]
            if not trajectory_id:
                raise ArtifactError("raw trajectory is missing trajectory_id")
            if trajectory_id in seen_trajectories:
                raise ArtifactError(
                    f"duplicate trajectory_id={trajectory_id!r}"
                )
            seen_trajectories.add(trajectory_id)
            metrics = compute_deterministic_metrics(normalized)
            write(
                {
                    "schema_version": PREPROCESSED_SCHEMA,
                    "task_id": normalized["task_id"],
                    "trajectory_id": trajectory_id,
                    "normalized_trajectory": normalized,
                    "deterministic_metrics": metrics,
                }
            )
            written += 1
    print(
        json.dumps(
            {"command": "preprocess", "written": written, "output": str(args.output)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def rubric_candidates(args: argparse.Namespace) -> None:
    _guard_blind_final(
        [args.task_facts],
        allowed=args.allow_blind_final,
    )
    written = 0
    with atomic_jsonl_writer(args.output, force=args.force) as write:
        for task_facts in iter_jsonl(args.task_facts):
            write(extract_rubric_candidates(task_facts))
            written += 1
    print(
        json.dumps(
            {
                "command": "rubric-candidates",
                "written": written,
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def judge_inputs(args: argparse.Namespace) -> None:
    _guard_blind_final(
        [args.preprocessed, args.rubrics],
        allowed=args.allow_blind_final,
    )
    rubrics = index_jsonl(args.rubrics, key="task_id")
    written = 0
    with atomic_jsonl_writer(args.output, force=args.force) as write:
        for row in iter_jsonl(args.preprocessed):
            if row.get("schema_version") != PREPROCESSED_SCHEMA:
                raise ArtifactError("unsupported preprocessed schema")
            task_id = int(row["task_id"])
            rubric = rubrics.get(task_id)
            if rubric is None:
                raise ArtifactError(f"missing Rubric for task_id={task_id}")
            normalized = row["normalized_trajectory"]
            metrics = row["deterministic_metrics"]
            validity = metrics.get("validity") or {}
            if validity.get("infrastructure_invalid"):
                judge = build_not_judged_result(
                    task_id=task_id,
                    trajectory_id=row["trajectory_id"],
                    reason="infrastructure_invalid",
                )
                request = {
                    "schema_version": JUDGE_REQUEST_SCHEMA,
                    "task_id": task_id,
                    "trajectory_id": row["trajectory_id"],
                    "judge_required": False,
                    "prompt_version": TRAJECTORY_JUDGE_PROMPT_VERSION,
                    "judge_model": args.judge_model,
                    "not_judged_result": judge,
                }
            else:
                messages = build_trajectory_judge_messages(
                    normalized=normalized,
                    rubric_bundle=rubric,
                    deterministic_metrics=metrics,
                )
                request = {
                    "schema_version": JUDGE_REQUEST_SCHEMA,
                    "task_id": task_id,
                    "trajectory_id": row["trajectory_id"],
                    "judge_required": True,
                    "prompt_version": TRAJECTORY_JUDGE_PROMPT_VERSION,
                    "judge_model": args.judge_model,
                    "rubric_ids": [
                        item["rubric_id"] for item in rubric["rubrics"]
                    ],
                    "allowed_event_ids": [
                        event["event_id"]
                        for event in normalized.get("events") or []
                        if isinstance(event, dict) and event.get("event_id")
                    ],
                    "messages": messages,
                }
            request["judge_request_hash"] = stable_hash(request)
            write(request)
            written += 1
    print(
        json.dumps(
            {
                "command": "judge-inputs",
                "written": written,
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def assemble(args: argparse.Namespace) -> None:
    _guard_blind_final(
        [
            args.preprocessed,
            args.rubrics,
            args.judges,
            args.expected_tasks,
        ],
        allowed=args.allow_blind_final,
    )
    expected_task_ids = _task_ids(args.expected_tasks)
    expected_set = set(expected_task_ids)
    rubrics = index_jsonl(
        args.rubrics,
        key="task_id",
        allowed_keys=expected_set,
    )
    judges = index_jsonl(args.judges, key="trajectory_id")
    actor = load_json(args.actor)
    evaluations = []
    seen_tasks = set()
    for row in iter_jsonl(args.preprocessed):
        if row.get("schema_version") != PREPROCESSED_SCHEMA:
            raise ArtifactError("unsupported preprocessed schema")
        task_id = int(row["task_id"])
        if task_id not in expected_set:
            raise ArtifactError(f"unexpected preprocessed task_id={task_id}")
        if task_id in seen_tasks:
            raise ArtifactError(
                f"formal assembly requires one trajectory per task; duplicate {task_id}"
            )
        seen_tasks.add(task_id)
        rubric = rubrics.get(task_id)
        if rubric is None:
            raise ArtifactError(f"missing Rubric for task_id={task_id}")
        judge = judges.get(row["trajectory_id"])
        if judge is None:
            validity = row["deterministic_metrics"].get("validity") or {}
            if validity.get("infrastructure_invalid"):
                judge = build_not_judged_result(
                    task_id=task_id,
                    trajectory_id=row["trajectory_id"],
                    reason="infrastructure_invalid",
                )
            else:
                raise ArtifactError(
                    f"missing Judge result for trajectory_id={row['trajectory_id']}"
                )
        evaluations.append(
            assemble_task_evaluation(
                actor=actor,
                normalized_trajectory=row["normalized_trajectory"],
                deterministic_metrics=row["deterministic_metrics"],
                rubric_bundle=rubric,
                judge_result=judge,
            )
        )
    with atomic_jsonl_writer(args.output, force=args.force) as write:
        for evaluation in evaluations:
            write(evaluation)
    summary = summarize_evaluations(
        expected_task_ids=expected_task_ids,
        evaluations=evaluations,
    )
    write_json_atomic(args.summary, summary, force=args.force)
    print(
        json.dumps(
            {
                "command": "assemble",
                "evaluations": len(evaluations),
                "expected_tasks": len(expected_task_ids),
                "output": str(args.output),
                "summary": str(args.summary),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def compare(args: argparse.Namespace) -> None:
    paths = [args.expected_tasks]
    runs = {}
    for specification in args.model_evaluations:
        if "=" not in specification:
            raise ArtifactError(
                "--model-evaluations must use LABEL=PATH"
            )
        label, raw_path = specification.split("=", 1)
        label = label.strip()
        if not label or label in runs:
            raise ArtifactError(
                f"invalid or duplicate model label {label!r}"
            )
        path = Path(raw_path)
        paths.append(path)
        runs[label] = list(iter_jsonl(path))
    _guard_blind_final(paths, allowed=args.allow_blind_final)
    expected_task_ids = _task_ids(args.expected_tasks)
    comparison = compare_evaluation_runs(
        expected_task_ids=expected_task_ids,
        runs=runs,
    )
    write_json_atomic(args.output, comparison, force=args.force)
    print(
        json.dumps(
            {
                "command": "compare",
                "models": list(runs),
                "expected_tasks": len(expected_task_ids),
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _common_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-blind-final",
        action="store_true",
        help="显式允许处理已冻结 final blind test；日常开发不得使用",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 Shopping Agent 纯离线轨迹评测 artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser(
        "preprocess",
        help="将现有 raw Rollout 规范化并计算确定性指标",
    )
    command.add_argument("--raw", type=Path, required=True)
    command.add_argument("--limit", type=int)
    _common_output(command)
    command.set_defaults(handler=preprocess)

    command = subparsers.add_parser(
        "rubric-candidates",
        help="从私有 TaskFacts 生成代码约束的 Rubric 候选",
    )
    command.add_argument("--task-facts", type=Path, required=True)
    _common_output(command)
    command.set_defaults(handler=rubric_candidates)

    command = subparsers.add_parser(
        "judge-inputs",
        help="从已冻结 Rubric 构建 Actor-visible Judge 请求",
    )
    command.add_argument("--preprocessed", type=Path, required=True)
    command.add_argument("--rubrics", type=Path, required=True)
    command.add_argument(
        "--judge-model",
        default=DEFAULT_PRO_MODEL,
        help="写入请求身份；运行 Judge 时模型必须完全一致",
    )
    _common_output(command)
    command.set_defaults(handler=judge_inputs)

    command = subparsers.add_parser(
        "assemble",
        help="拼装已缓存 Judge 结果并生成四部分汇总",
    )
    command.add_argument("--preprocessed", type=Path, required=True)
    command.add_argument("--rubrics", type=Path, required=True)
    command.add_argument("--judges", type=Path, required=True)
    command.add_argument("--expected-tasks", type=Path, required=True)
    command.add_argument("--actor", type=Path, required=True)
    command.add_argument("--summary", type=Path, required=True)
    _common_output(command)
    command.set_defaults(handler=assemble)

    command = subparsers.add_parser(
        "compare",
        help="按 task_id 对 Base/SFT/GRPO 结果做分栏配对比较",
    )
    command.add_argument("--expected-tasks", type=Path, required=True)
    command.add_argument(
        "--model-evaluations",
        action="append",
        required=True,
        help="重复传入 LABEL=/path/to/evaluations.jsonl",
    )
    _common_output(command)
    command.set_defaults(handler=compare)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
