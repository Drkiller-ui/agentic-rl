#!/usr/bin/env python3
"""Call frozen Flash/Pro prompts; never imported by the training runtime."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from shopping_grpo.evaluation.artifacts import (
    ArtifactError,
    append_jsonl_fsync,
    index_jsonl,
    iter_jsonl,
)
from shopping_grpo.evaluation.blind_guard import guard_blind_final
from shopping_grpo.evaluation.contracts import (
    ContractValidationError,
    validate_judge_result,
    validate_rubric_bundle,
)
from shopping_grpo.evaluation.model_client import (
    DEFAULT_FLASH_MODEL,
    DEFAULT_PRO_MODEL,
    client_from_environment,
)
from shopping_grpo.evaluation.prompts import (
    RUBRIC_CURATOR_PROMPT_VERSION,
    TRAJECTORY_JUDGE_PROMPT_VERSION,
    build_rubric_curator_messages,
)
from shopping_grpo.evaluation.rubric import (
    materialize_rubric_bundle,
    stable_hash,
)


def _guard_blind_final(paths: list[Path], *, allowed: bool) -> None:
    guard_blind_final(paths, allowed=allowed)


def _client(args: argparse.Namespace):
    return client_from_environment(
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        response_format_json=args.response_format_json,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
    )


def _repair_messages(
    original_messages: list[dict],
    invalid_result: dict,
    error: ContractValidationError,
) -> list[dict]:
    """Ask for one schema-only repair without relaxing the frozen contract."""

    return [
        *deepcopy(original_messages),
        {
            "role": "assistant",
            "content": json.dumps(
                invalid_result,
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
        {
            "role": "user",
            "content": (
                "上一个 JSON 未通过固定 schema 校验："
                f"{error}。请严格遵守原始任务和 schema，只输出修正后的完整 JSON；"
                "不要解释，不要添加原 schema 之外的字段。"
            ),
        },
    ]


def _audit_with_validation_retries(
    metadata: dict,
    failures: list[dict],
) -> dict:
    audit = deepcopy(metadata)
    audit["validation_attempts"] = len(failures) + 1
    audit["validation_failures"] = failures
    return audit


def _existing_output(
    args: argparse.Namespace,
    *,
    key: str,
) -> dict:
    if not args.output.exists():
        return {}
    if not args.resume:
        raise FileExistsError(
            f"output already exists: {args.output}; use --resume or a new path"
        )
    return index_jsonl(args.output, key=key)


def _validated_judge_request_hash(
    request: dict,
    *,
    trajectory_id: str,
) -> str:
    claimed_hash = request.get("judge_request_hash")
    if not isinstance(claimed_hash, str) or not claimed_hash:
        raise ArtifactError(
            f"Judge request hash missing for {trajectory_id}"
        )
    hash_material = {
        key: value
        for key, value in request.items()
        if key != "judge_request_hash"
    }
    actual_hash = stable_hash(hash_material)
    if claimed_hash != actual_hash:
        raise ArtifactError(
            f"Judge request content hash mismatch for {trajectory_id}"
        )
    return claimed_hash


def curate_rubrics(args: argparse.Namespace) -> None:
    _guard_blind_final(
        [args.task_facts, args.candidates],
        allowed=args.allow_blind_final,
    )
    task_facts = index_jsonl(args.task_facts, key="task_id")
    candidates = index_jsonl(args.candidates, key="task_id")
    if set(task_facts) != set(candidates):
        raise ArtifactError(
            "task facts and candidate task IDs must match exactly"
        )
    existing = _existing_output(args, key="task_id")
    for task_id, bundle in existing.items():
        validated = validate_rubric_bundle(
            bundle,
            expected_task_id=int(task_id),
        )
        generation = validated["generation"]
        audit = validated.get("curator_audit") or {}
        if (
            generation["curator_model"] != args.model
            or generation["curator_prompt_version"]
            != RUBRIC_CURATOR_PROMPT_VERSION
            or validated["rubric_version"] != args.rubric_version
            or generation["task_data_hash"]
            != str(task_facts[task_id]["task_data_hash"])
            or generation["query_hash"]
            != str(task_facts[task_id]["query_hash"])
            or generation["extractor_version"]
            != str(candidates[task_id]["extractor_version"])
            or audit.get("requested_thinking") is not args.thinking
            or audit.get("requested_reasoning_effort")
            != (args.reasoning_effort if args.thinking else None)
        ):
            raise ArtifactError(
                f"cached Rubric version mismatch for task_id={task_id}"
            )
    client = None
    written = 0
    skipped = 0
    for task_id in sorted(task_facts):
        if task_id in existing:
            skipped += 1
            continue
        if args.limit is not None and written >= args.limit:
            break
        facts = task_facts[task_id]
        candidate_bundle = candidates[task_id]
        messages = build_rubric_curator_messages(
            task_id=int(task_id),
            query=str(facts["query"]),
            candidates=candidate_bundle["candidates"],
        )
        if client is None:
            client = _client(args)
        request_messages = messages
        validation_failures = []
        for validation_attempt in range(args.validation_retries + 1):
            completion = client.complete_json(request_messages)
            try:
                bundle = materialize_rubric_bundle(
                    task_facts=facts,
                    candidates=candidate_bundle,
                    curator_response=completion["result"],
                    curator_model=args.model,
                    curator_prompt_version=RUBRIC_CURATOR_PROMPT_VERSION,
                    rubric_version=args.rubric_version,
                )
                break
            except ContractValidationError as exc:
                validation_failures.append(
                    {
                        "attempt": validation_attempt + 1,
                        "error": str(exc),
                        "response_hash": stable_hash(
                            completion["result"]
                        ),
                        "provider_request_id": completion[
                            "metadata"
                        ].get("provider_request_id"),
                    }
                )
                if validation_attempt >= args.validation_retries:
                    raise
                request_messages = _repair_messages(
                    messages,
                    completion["result"],
                    exc,
                )
        bundle["curator_audit"] = _audit_with_validation_retries(
            completion["metadata"],
            validation_failures,
        )
        bundle["curator_raw_response"] = deepcopy(completion["result"])
        append_jsonl_fsync(args.output, bundle)
        written += 1
    print(
        json.dumps(
            {
                "command": "curate-rubrics",
                "model": args.model,
                "skipped_cached": skipped,
                "written": written,
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def run_judge(args: argparse.Namespace) -> None:
    _guard_blind_final(
        [args.requests],
        allowed=args.allow_blind_final,
    )
    existing = _existing_output(args, key="trajectory_id")
    for trajectory_id, result in existing.items():
        audit = result.get("judge_audit") or {}
        prompt_version = result.get("judge_prompt_version")
        if not result.get("judge_request_hash"):
            raise ArtifactError(
                f"cached Judge request hash missing for {trajectory_id}"
            )
        if result.get("judge_status") != "not_judged" and (
            audit.get("requested_model") != args.model
            or prompt_version != TRAJECTORY_JUDGE_PROMPT_VERSION
            or audit.get("requested_thinking") is not args.thinking
            or audit.get("requested_reasoning_effort")
            != (args.reasoning_effort if args.thinking else None)
        ):
            raise ArtifactError(
                f"cached Judge version mismatch for {trajectory_id}"
            )
    client = None
    written = 0
    skipped = 0
    for request in iter_jsonl(args.requests):
        task_id = int(request["task_id"])
        trajectory_id = str(request["trajectory_id"])
        request_hash = _validated_judge_request_hash(
            request,
            trajectory_id=trajectory_id,
        )
        if request.get("judge_model") != args.model:
            raise ArtifactError(
                f"Judge request model mismatch for {trajectory_id}"
            )
        if trajectory_id in existing:
            cached = existing[trajectory_id]
            if cached.get("judge_request_hash") != request_hash:
                raise ArtifactError(
                    f"cached Judge request hash mismatch for {trajectory_id}"
                )
            if request.get("judge_required") is False:
                if cached.get("judge_status") != "not_judged":
                    raise ArtifactError(
                        f"cached Judge status mismatch for {trajectory_id}"
                    )
            else:
                validate_judge_result(
                    cached,
                    rubric_ids=request.get("rubric_ids") or [],
                    expected_task_id=task_id,
                    expected_trajectory_id=trajectory_id,
                    allowed_event_ids=request.get("allowed_event_ids") or [],
                )
            skipped += 1
            continue
        if args.limit is not None and written >= args.limit:
            break
        if request.get("judge_required") is False:
            result = request.get("not_judged_result")
            if not isinstance(result, dict):
                raise ArtifactError(
                    f"missing not_judged_result for {trajectory_id}"
                )
            result = deepcopy(result)
            result["judge_prompt_version"] = (
                TRAJECTORY_JUDGE_PROMPT_VERSION
            )
            result["judge_model"] = args.model
            result["judge_request_hash"] = request_hash
            append_jsonl_fsync(args.output, result)
            written += 1
            continue
        if request.get("prompt_version") != TRAJECTORY_JUDGE_PROMPT_VERSION:
            raise ArtifactError(
                f"unsupported Judge prompt version for {trajectory_id}"
            )
        messages = request.get("messages")
        if not isinstance(messages, list):
            raise ArtifactError(
                f"Judge request messages missing for {trajectory_id}"
            )
        rubric_ids = request.get("rubric_ids")
        event_ids = request.get("allowed_event_ids")
        if not isinstance(rubric_ids, list) or not isinstance(event_ids, list):
            raise ArtifactError(
                f"Judge request validation IDs missing for {trajectory_id}"
            )
        if client is None:
            client = _client(args)
        request_messages = messages
        validation_failures = []
        for validation_attempt in range(args.validation_retries + 1):
            completion = client.complete_json(request_messages)
            try:
                result = validate_judge_result(
                    completion["result"],
                    rubric_ids=rubric_ids,
                    expected_task_id=task_id,
                    expected_trajectory_id=trajectory_id,
                    allowed_event_ids=event_ids,
                )
                break
            except ContractValidationError as exc:
                validation_failures.append(
                    {
                        "attempt": validation_attempt + 1,
                        "error": str(exc),
                        "response_hash": stable_hash(
                            completion["result"]
                        ),
                        "provider_request_id": completion[
                            "metadata"
                        ].get("provider_request_id"),
                    }
                )
                if validation_attempt >= args.validation_retries:
                    raise
                request_messages = _repair_messages(
                    messages,
                    completion["result"],
                    exc,
                )
        result["judge_prompt_version"] = TRAJECTORY_JUDGE_PROMPT_VERSION
        result["judge_model"] = args.model
        result["judge_request_hash"] = request_hash
        result["judge_audit"] = _audit_with_validation_retries(
            completion["metadata"],
            validation_failures,
        )
        result["judge_raw_response"] = deepcopy(completion["result"])
        append_jsonl_fsync(args.output, result)
        written += 1
    print(
        json.dumps(
            {
                "command": "judge",
                "model": args.model,
                "skipped_cached": skipped,
                "written": written,
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _model_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_max_tokens: int,
) -> None:
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--max-tokens", type=int, default=default_max_tokens)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--validation-retries",
        type=int,
        default=2,
        help="JSON 已解析但未通过固定 schema 时的有限修复次数",
    )
    parser.add_argument("--response-format-json", action="store_true")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="为 DeepSeek V4 显式启用 thinking",
    )
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="校验已有缓存版本并只追加缺失 task/trajectory",
    )
    parser.add_argument("--allow-blind-final", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 Shopping 需求 Rubric 和轨迹 Judge 模型"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("curate-rubrics")
    command.add_argument("--task-facts", type=Path, required=True)
    command.add_argument("--candidates", type=Path, required=True)
    command.add_argument("--rubric-version", default="task-rubric-v1")
    _model_arguments(
        command,
        default_model=DEFAULT_FLASH_MODEL,
        default_max_tokens=2048,
    )
    command.set_defaults(handler=curate_rubrics)

    command = commands.add_parser("judge")
    command.add_argument("--requests", type=Path, required=True)
    _model_arguments(
        command,
        default_model=DEFAULT_PRO_MODEL,
        default_max_tokens=4096,
    )
    command.set_defaults(handler=run_judge)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    if args.validation_retries < 0:
        raise SystemExit("--validation-retries cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
