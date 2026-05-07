#!/usr/bin/env python3
"""Validate TAC package inputs, generated source splits, and training configs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CACHE_FILES = {
    "source_ids.txt",
    "source_embeddings_l2.npy",
    "reliability_P.npy",
    "reliability_G.npy",
    "reliability_M.npy",
    "reliability_spec.json",
    "influence.npy",
    "late_influence.npy",
    "manifest.json",
}


def load_selector_module():
    path = ROOT / "scripts/select_tac_sources.py"
    spec = importlib.util.spec_from_file_location("select_tac_sources", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["select_tac_sources"] = module
    spec.loader.exec_module(module)
    return module


def count_ids(path: Path) -> int:
    return len(read_ids(path))


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicates = len(values) - len(set(values))
    if duplicates:
        raise ValueError(f"{path}: contains {duplicates} duplicate subject IDs")
    return values


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_cache(targets: list[str], budget: int) -> None:
    selector = load_selector_module()
    for target in targets:
        cache_dir = ROOT / "cache" / target
        missing = sorted(REQUIRED_CACHE_FILES - {path.name for path in cache_dir.iterdir() if path.is_file()})
        if missing:
            raise FileNotFoundError(f"{target}: missing cache files: {', '.join(missing)}")
        source_ids = read_ids(cache_dir / "source_ids.txt")
        source_count = len(source_ids)
        selector.compute_reliability(cache_dir, source_count)
        if source_count < budget:
            raise ValueError(f"{target}: source cache too small for budget {budget}: {source_count}")
        print(f"cache ok: {target} source_count={source_count}")


def validate_splits(targets: list[str], budget: int, require_generated: bool) -> None:
    for target in targets:
        split_file = ROOT / "data/source_splits" / target / f"tac_{budget}" / "train_subjects.txt"
        if not split_file.exists():
            if require_generated:
                raise FileNotFoundError(f"{target}: missing generated split {split_file}")
            print(f"split pending: {target} {split_file}")
            continue
        split_count = count_ids(split_file)
        if split_count != budget:
            raise ValueError(f"{target}: expected {budget} ids in {split_file}, found {split_count}")
        print(f"split ok: {target} count={split_count}")


def validate_configs(targets: list[str], budget: int, require_generated: bool) -> None:
    target_specs = json.loads((ROOT / "configs/targets.json").read_text(encoding="utf-8"))
    for target in targets:
        if target not in target_specs:
            raise KeyError(f"{target}: missing from configs/targets.json")
        support_tag = target_specs[target]["target_support_tag"]
        config_path = ROOT / "configs/generated" / f"b{budget}" / f"train_config_BraTS2021_{target}_S_TAC_{budget}_T_{support_tag}.yaml"
        if not config_path.exists():
            if require_generated:
                raise FileNotFoundError(f"{target}: missing training config {config_path}")
            print(f"config pending: {target} {config_path}")
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        domains = config["data"]["domains"]
        if len(domains) != 2:
            raise ValueError(f"{target}: expected 2 training domains, found {len(domains)}")
        if domains[0]["name"] != "BraTS21_Source" or domains[1]["name"] != "BraTS21_T_train":
            raise ValueError(f"{target}: unexpected domain order/names in {config_path}")
        source_split_dir = repo_path(config["data"]["domains"][0]["split_txt"])
        target_split_dir = repo_path(config["data"]["domains"][1]["split_txt"])
        save_dir = repo_path(config["training"]["save_dir"])
        expected_source_dir = ROOT / "data/source_splits" / target / f"tac_{budget}"
        expected_target_dir = ROOT / "data/target_splits" / target / "tac_10"
        if source_split_dir.resolve() != expected_source_dir.resolve():
            raise ValueError(f"{target}: source split points to {source_split_dir}, expected {expected_source_dir}")
        if target_split_dir.resolve() != expected_target_dir.resolve():
            raise ValueError(f"{target}: target split points to {target_split_dir}, expected {expected_target_dir}")
        if count_ids(source_split_dir / "train_subjects.txt") != budget:
            raise ValueError(f"{target}: source split in config does not contain {budget} ids")
        if count_ids(target_split_dir / "train_subjects.txt") != 10:
            raise ValueError(f"{target}: target split in config does not contain 10 ids")
        for required in [
            target_split_dir / "train_subjects.txt",
            repo_path(config["data"]["val"]["split_txt"]) / "val_subjects.txt",
            repo_path(config["data"]["test"]["split_txt"]) / "test_subjects.txt",
        ]:
            if not required.exists():
                raise FileNotFoundError(f"{target}: configured split file missing: {required}")
        print(f"config ok: {target} save_dir={save_dir}")


def validate_training_manifest(targets: list[str], budget: int, require_generated: bool) -> None:
    manifest_path = ROOT / "results" / f"training_config_manifest_b{budget}.json"
    if not manifest_path.exists():
        if require_generated:
            raise FileNotFoundError(f"missing TAC training manifest {manifest_path}")
        print(f"manifest pending: {manifest_path}")
        return
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(rows) != len(targets):
        raise ValueError(f"{manifest_path}: expected {len(targets)} rows, found {len(rows)}")
    seen = {row["target"] for row in rows}
    if seen != set(targets):
        raise ValueError(f"{manifest_path}: target set {sorted(seen)} != expected {sorted(targets)}")
    for row in rows:
        config_path = Path(row["config_path"])
        config_path = config_path if config_path.is_absolute() else ROOT / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"{manifest_path}: config_path missing: {config_path}")
        if int(row["budget"]) != budget:
            raise ValueError(f"{manifest_path}: row budget {row['budget']} != {budget}")
    print(f"manifest ok: {manifest_path} rows={len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--require-generated", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_specs = json.loads((ROOT / "configs/targets.json").read_text(encoding="utf-8"))
    targets = sorted(target_specs)
    missing_cache = [target for target in targets if not (ROOT / "cache" / target).is_dir()]
    if missing_cache:
        raise FileNotFoundError(f"missing cache directories: {', '.join(missing_cache)}")
    validate_cache(targets, args.budget)
    validate_splits(targets, args.budget, args.require_generated)
    validate_configs(targets, args.budget, args.require_generated)
    validate_training_manifest(targets, args.budget, args.require_generated)
    print("TAC package validation complete")


if __name__ == "__main__":
    main()
