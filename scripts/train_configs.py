#!/usr/bin/env python3
"""Run generated EfficientVit training configs with the local Python env."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def config_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def manifest_configs(
    budget: int,
    run_filters: set[str],
    targets: set[str],
    source_methods: set[str],
    target_methods: set[str],
    run_ids: set[str],
) -> list[Path]:
    manifest = ROOT / "results" / f"experiment_manifest_b{budget}.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest. Run build_experiment_configs.py first: {manifest}")
    rows = json.loads(manifest.read_text(encoding="utf-8"))["rows"]
    configs = []
    matched = set()
    for row in rows:
        tokens = {row["run_id"], row["target"], row["source_method"], row["target_method"]}
        if targets and row["target"] not in targets:
            continue
        if source_methods and row["source_method"] not in source_methods:
            continue
        if target_methods and row["target_method"] not in target_methods:
            continue
        if run_ids and row["run_id"] not in run_ids:
            continue
        if run_filters and not run_filters.issubset(tokens):
            continue
        matched |= run_filters if run_filters else set()
        configs.append(config_path(row["config_path"]))
    missing_filters = sorted(run_filters - matched)
    if missing_filters:
        raise ValueError(f"Filters did not match any runnable config: {missing_filters}")
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--filter", nargs="*", default=[])
    parser.add_argument("--target", nargs="*", default=[])
    parser.add_argument("--source-method", nargs="*", default=[])
    parser.add_argument("--target-method", nargs="*", default=[])
    parser.add_argument("--run-id", nargs="*", default=[])
    parser.add_argument("--config", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional argument passed to train_seg.py; repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [config_path(path) for path in args.config]
    if not configs:
        if args.budget is None:
            raise ValueError("--budget is required unless explicit --config paths are provided")
        configs = manifest_configs(
            args.budget,
            set(args.filter),
            set(args.target),
            set(args.source_method),
            set(args.target_method),
            set(args.run_id),
        )
    if not configs:
        raise ValueError("No runnable configs selected")
    env = os.environ.copy()
    env.setdefault("TAC_ROOT", str(ROOT))
    for config in configs:
        cmd = [sys.executable, str(ROOT / "scripts/train_seg_amp_compat.py"), "--config", str(config), *args.extra_arg]
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        efficientvit_root = Path(env.get("EFFICIENTVIT_ROOT", str(ROOT / "external/EfficientVit"))).expanduser()
        efficientvit_root = efficientvit_root.resolve() if efficientvit_root.is_absolute() else (ROOT / efficientvit_root).resolve()
        env["EFFICIENTVIT_ROOT"] = str(efficientvit_root)
        subprocess.run(cmd, check=True, cwd=str(efficientvit_root), env=env)


if __name__ == "__main__":
    main()
