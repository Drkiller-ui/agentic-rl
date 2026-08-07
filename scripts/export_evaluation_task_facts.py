#!/usr/bin/env python3
"""Export private TaskFacts with the exact ShopSimulator v2.1 goal ordering."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from shopping_grpo.evaluation.artifacts import iter_jsonl, write_jsonl_atomic
from shopping_grpo.evaluation.task_facts import task_facts_from_environment


ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"
FINAL_BLIND_ASSET = "shop_benchmark_reward_v3_final_200"


def _task_ids(path: Path) -> list[int]:
    result = []
    seen = set()
    for row_number, row in enumerate(iter_jsonl(path), start=1):
        if "task_id" not in row:
            raise ValueError(f"{path}:{row_number}: missing task_id")
        task_id = int(row["task_id"])
        if task_id in seen:
            raise ValueError(
                f"{path}:{row_number}: duplicate task_id={task_id}"
            )
        seen.add(task_id)
        result.append(task_id)
    return result


def load_environment_data(shopsim_root: Path):
    """Load the same products and deterministic goals as the API process."""

    root = shopsim_root.resolve()
    if not (root / "web_agent_site").is_dir():
        raise ValueError(
            "--shopsim-root must point to ShopSimulator/shop_env"
        )
    os.environ["SHOP_ENVIRONMENT_VERSION"] = ENVIRONMENT_VERSION
    sys.path.insert(0, str(root))
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals
    from web_agent_site.utils import DEFAULT_FILE_PATH

    products, product_item_dict, prices, _ = load_products(
        DEFAULT_FILE_PATH,
        num_products=None,
        human_goals=None,
    )
    goals = get_goals(products, prices)
    return goals, product_item_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "导出评测私有 TaskFacts；该文件不得进入 Actor 或 Judge 的隐藏输入"
        )
    )
    parser.add_argument("--shopsim-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-blind-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        FINAL_BLIND_ASSET in str(args.tasks)
        and not args.allow_blind_final
    ):
        raise SystemExit(
            "refusing to export final blind TaskFacts without "
            "--allow-blind-final"
        )
    task_ids = _task_ids(args.tasks)
    goals, product_item_dict = load_environment_data(args.shopsim_root)
    rows = task_facts_from_environment(
        task_ids=task_ids,
        goals=goals,
        product_item_dict=product_item_dict,
    )
    write_jsonl_atomic(args.output, rows, force=args.force)
    print(
        json.dumps(
            {
                "environment_version": ENVIRONMENT_VERSION,
                "tasks": len(rows),
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
