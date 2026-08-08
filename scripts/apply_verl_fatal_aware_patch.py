#!/usr/bin/env python3
"""Apply or restore the pinned veRL 0.8 fatal-aware advantage patch.

The patch inserts a one-sided advantage/returns clamp for fatal trajectories
into `verl.trainer.ppo.ray_trainer.compute_advantage` (GRPO branch).  Fatal
trajectories (consecutive guard rejections / repeat loop / consecutive repeated
actions, tagged by `ShoppingToolAgentLoop` into ``extra_fields["is_fatal"]``)
keep their full reward in group statistics but receive only non-negative
gradient; worst case degenerates to a hard mask.

This patch stacks on top of the dynamic-sampling patch; it is applied with a
plain Python anchor insert (idempotent) instead of `git patch` because the
installed file is no longer the pristine upstream version.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import py_compile
import shutil
import sys
from pathlib import Path


EXPECTED_VERL_VERSION = "0.8.0"
PATCH_MARKER = "SHOPPING_GRPO_FATAL_AWARE_PATCH_V1"
BACKUP_SUFFIX = ".shopping-grpo-fatal-aware.orig"

# The GRPO branch ends with this exact two-line sequence; the following
# ``else:`` line is unique to the GRPO branch (GAE and generic branches are
# followed by different statements).  Insert the clamp right after it.
ANCHOR = (
    '        data.batch["returns"] = returns\n'
    "    else:\n"
    "        # handle all other adv estimator type other than GAE and GRPO\n"
)
INSERT = (
    '        data.batch["returns"] = returns\n'
    "        # SHOPPING_GRPO_FATAL_AWARE_PATCH_V1: one-sided advantage clamp\n"
    "        # for fatal trajectories.  Fatal trajectories keep full reward in\n"
    "        # the group statistics computed above, but their tokens receive\n"
    "        # only non-negative gradient; worst case degenerates to a hard mask.\n"
    '        if "is_fatal" in data.non_tensor_batch:\n'
    '            _fatal_mask = data.non_tensor_batch["is_fatal"]\n'
    '            if hasattr(_fatal_mask, "astype"):\n'
    "                _fatal_mask = _fatal_mask.astype(bool)\n"
    "            if np.any(_fatal_mask):\n"
    "                for _fi in np.where(_fatal_mask)[0]:\n"
    '                    data.batch["advantages"][_fi].clamp_(min=0.0)\n'
    '                    data.batch["returns"][_fi].clamp_(min=0.0)\n'
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_installed_ray_trainer() -> Path:
    installed_version = importlib.metadata.version("verl")
    if installed_version != EXPECTED_VERL_VERSION:
        raise RuntimeError(
            f"expected verl=={EXPECTED_VERL_VERSION}, got verl=={installed_version}"
        )

    import verl

    verl_source = Path(verl.__file__).resolve()
    expected_environment = (PROJECT_ROOT / ".venv").resolve()
    if not verl_source.is_relative_to(expected_environment):
        raise RuntimeError(f"verl.__file__ is not from the project environment: {verl_source}")

    target = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    if not target.is_file():
        raise RuntimeError(f"installed ray_trainer.py does not exist: {target}")
    return target.resolve()


def validate_runtime_and_target(target_override: Path | None) -> Path:
    installed_target = resolve_installed_ray_trainer()
    if target_override is None:
        return installed_target
    target = target_override.resolve()
    if not target.is_file():
        raise RuntimeError(f"target ray_trainer.py does not exist: {target}")
    return target


def verify_patched(target: Path) -> None:
    text = target.read_text(encoding="utf-8")
    if PATCH_MARKER not in text:
        raise RuntimeError(f"patched ray_trainer.py is missing marker {PATCH_MARKER}")
    if ANCHOR in text:
        raise RuntimeError("patched ray_trainer.py still contains the unpatched anchor")
    py_compile.compile(str(target), doraise=True)


def apply_patch(target: Path) -> None:
    text = target.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        verify_patched(target)
        print(f"veRL fatal-aware patch already applied: {target}")
        return
    if ANCHOR not in text:
        raise RuntimeError(
            "refusing to patch unknown ray_trainer.py: expected GRPO branch anchor not found"
        )
    count = text.count(ANCHOR)
    if count != 1:
        raise RuntimeError(f"expected exactly one GRPO branch anchor, found {count}")

    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(target, backup)

    rollback_source = backup
    try:
        target.write_text(text.replace(ANCHOR, INSERT), encoding="utf-8")
        verify_patched(target)
    except Exception:
        shutil.copy2(rollback_source, target)
        raise

    print(f"applied veRL fatal-aware patch: {target}")
    print(f"backup: {backup}")


def restore_patch(target: Path) -> None:
    backup = Path(str(target) + BACKUP_SUFFIX)
    text = target.read_text(encoding="utf-8")
    if PATCH_MARKER not in text:
        print(f"veRL ray_trainer.py has no fatal-aware patch: {target}")
        return
    if not backup.is_file():
        raise RuntimeError(f"cannot restore without backup: {backup}")

    restore_temp = target.with_name(target.name + ".shopping-grpo-fatal-restore.tmp")
    shutil.copy2(backup, restore_temp)
    restore_temp.replace(target)
    if PATCH_MARKER in target.read_text(encoding="utf-8"):
        raise RuntimeError(f"restore verification failed: {target}")
    py_compile.compile(str(target), doraise=True)
    print(f"restored pre-fatal-aware veRL ray_trainer.py: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore",
        action="store_true",
        help="restore the pre-patch file from the automatic backup",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the target is already patched without modifying it",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="override ray_trainer.py target for isolated patch-script tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum((args.restore, args.check)) > 1:
        raise SystemExit("--restore and --check are mutually exclusive")
    try:
        target = validate_runtime_and_target(args.target)
        if args.restore:
            restore_patch(target)
        elif args.check:
            verify_patched(target)
            print(f"verified veRL fatal-aware patch: {target}")
        else:
            apply_patch(target)
    except (OSError, RuntimeError, py_compile.PyCompileError) as exc:
        raise SystemExit(f"veRL fatal-aware patch error: {exc}") from exc


if __name__ == "__main__":
    main()
