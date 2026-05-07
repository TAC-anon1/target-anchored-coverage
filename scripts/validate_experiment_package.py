#!/usr/bin/env python3
"""Validate generated method-grid configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def count_ids(path: Path) -> int:
    return len(read_ids(path))


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicates = len(values) - len(set(values))
    if duplicates:
        raise ValueError(f"{path}: contains {duplicates} duplicate subject IDs")
    return values


def pair_allowed(source_method: str, target_method: str, budget: int) -> bool:
    if source_method == "tac":
        return target_method == "tac"
    if source_method == "orient" and target_method == "tac":
        return False
    if source_method == "target_only":
        return target_method != "none"
    if source_method == "full_source":
        return True
    if target_method == "none":
        return False
    return True


def expected_combo_count(budget: int) -> int:
    grid = json.loads((ROOT / "configs/method_grid.json").read_text(encoding="utf-8"))
    count = 0
    for _target in grid["targets"]:
        for source_method in grid["source_methods"]:
            for target_method in ["none", *grid["target_methods"]]:
                count += int(pair_allowed(source_method, target_method, budget))
    return count


def resolve_config_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_manifest(path: Path, output_root: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    budget = int(manifest["budget"])
    expected_count = expected_combo_count(budget)
    actual_count = len(manifest["rows"])
    if actual_count != expected_count:
        raise ValueError(f"{path}: rows={actual_count}, expected runnable rows {expected_count}")
    row_ids = [row["run_id"] for row in manifest["rows"]]
    duplicate_ids = len(row_ids) - len(set(row_ids))
    if duplicate_ids:
        raise ValueError(f"{path}: contains {duplicate_ids} duplicate run IDs")
    for row in manifest["rows"]:
        config_path = resolve_config_path(row["config_path"])
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for domain in config["data"]["domains"]:
            split_dir = repo_path(domain["split_txt"])
            train_file = split_dir / "train_subjects.txt"
            if not train_file.exists():
                raise FileNotFoundError(train_file)
            train_count = count_ids(train_file)
            if domain["name"] == "BraTS21_Source":
                if row["source_method"] == "full_source":
                    cache_ids = read_ids(ROOT / "cache" / row["target"] / "source_ids.txt")
                    if train_count != len(cache_ids):
                        raise ValueError(f"{row['run_id']}: full source count {train_count} != cache count {len(cache_ids)}")
                elif train_count != int(row["budget"]):
                    raise ValueError(f"{row['run_id']}: source count {train_count} != budget {row['budget']}")
            if domain["name"] == "BraTS21_T_train" and train_count != 10:
                raise ValueError(f"{row['run_id']}: target train count {train_count} != 10")
        for key in ("val", "test"):
            split_dir = repo_path(config["data"][key]["split_txt"])
            required = split_dir / f"{key}_subjects.txt"
            if not required.exists():
                raise FileNotFoundError(required)
        save_dir = repo_path(config["training"]["save_dir"])
        expected_root = output_root.resolve()
        if not is_relative_to(save_dir, expected_root):
            raise ValueError(f"{row['run_id']}: unexpected save_dir {save_dir}")
    forbidden_keys = {"skip" + "ped", "skip" + "ped_count"}
    if forbidden_keys & set(manifest):
        raise ValueError(f"{path}: reviewer manifest must contain runnable rows only")
    print(f"manifest ok: {path} runnable_rows={len(manifest['rows'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 150, 200])
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser()
    output_root = output_root.resolve() if output_root.is_absolute() else (ROOT / output_root).resolve()
    for budget in args.budgets:
        validate_manifest(ROOT / "results" / f"experiment_manifest_b{budget}.json", output_root)
    print("experiment package validation complete")


if __name__ == "__main__":
    main()
