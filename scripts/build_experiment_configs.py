#!/usr/bin/env python3
"""Build training configs for generated non-reference methods."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRATS_DATA_ROOT = Path(
    "external/BraTS2021_preprocessed"
)
TARGET_EVAL_SPLITS = {
    "TCGA_LGG": "data/target_splits/TCGA_LGG/eval",
    "C4": "data/target_splits/C4/eval",
    "C5": "data/target_splits/C5/eval",
    "TCGA_GBM": "data/target_splits/TCGA_GBM/eval",
}
TARGET_METHOD_SPLITS = {
    "random1": "random1_10",
    "random2": "random2_10",
    "random3": "random3_10",
    "coreset": "coreset_10",
    "lada": "lada_10",
    "aada": "aada_10",
    "ada_clue": "clue_10",
    "tac": "tac_10",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_ids(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def arg_path(value: Path | None, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def env_path(name: str, default: Path) -> Path:
    value = Path(os.environ.get(name, str(default))).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (ROOT / value).resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def config_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def split_exists(path: Path, count: int | None = None) -> bool:
    split_file = path / "train_subjects.txt"
    if not split_file.exists():
        return False
    return count is None or count_ids(split_file) == count


def source_split(target: str, source_method: str, target_method: str, budget: int) -> Path | None:
    base = ROOT / "data/source_splits" / target
    if source_method == "target_only":
        return None
    if source_method == "full_source":
        return base / "full_source"
    if source_method in {"random1", "random2", "random3"}:
        return base / f"{source_method}_{budget}"
    if source_method == "coreset":
        return base / f"coreset_{budget}"
    if source_method == "orient":
        return base / f"orient_{target_method}_{budget}"
    if source_method == "tac":
        return base / f"tac_{budget}"
    raise KeyError(source_method)


def target_split(target: str, target_method: str) -> Path | None:
    if target_method == "none":
        return None
    return ROOT / "data/target_splits" / target / TARGET_METHOD_SPLITS[target_method]


def default_pair_allowed(source_method: str, target_method: str, budget: int) -> bool:
    if target_method in {"aada", "ada_clue"} and budget != 50:
        return False
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


def run_id(target: str, source_method: str, target_method: str, budget: int) -> str:
    return f"B{budget}_{target}_S{source_method}_T{target_method}"


def config_for(
    target: str,
    source_method: str,
    target_method: str,
    budget: int,
    brats_data_root: Path,
    output_root: Path,
) -> dict[str, Any] | None:
    source_dir = source_split(target, source_method, target_method, budget)
    target_dir = target_split(target, target_method)
    eval_dir = repo_path(TARGET_EVAL_SPLITS[target])

    domains: list[dict[str, Any]] = []
    if source_dir is not None:
        expected = None if source_method == "full_source" else budget
        if not split_exists(source_dir, expected):
            return None
        domains.append({
            "name": "BraTS21_Source",
            "path": str(brats_data_root),
            "split": "train",
            "split_txt": config_path(source_dir),
        })
    if target_dir is not None:
        if not split_exists(target_dir, 10):
            return None
        domains.append({
            "name": "BraTS21_T_train",
            "path": str(brats_data_root),
            "split": "train",
            "split_txt": config_path(target_dir),
        })
    if not domains:
        return None
    for required in (eval_dir / "val_subjects.txt", eval_dir / "test_subjects.txt"):
        if not required.exists():
            return None

    save_dir = output_root / f"b{budget}" / run_id(target, source_method, target_method, budget)
    return {
        "model": {
            "name": "efficientvit_l1",
            "in_channels": 4,
            "num_classes": 4,
            "pretrained": True,
        },
        "data": {
            "skip_empty_train": True,
            "skip_empty_val": False,
            "domains": domains,
            "val": {
                "path": str(brats_data_root),
                "split": "val",
                "split_txt": config_path(eval_dir),
            },
            "test": {
                "path": str(brats_data_root),
                "split": "test",
                "split_txt": config_path(eval_dir),
            },
            "img_size": 512,
            "batch_size": 4,
            "num_workers": 4,
        },
        "optimizer": {
            "lr": 0.0001,
            "weight_decay": 1.0e-05,
        },
        "scheduler": {
            "T_max": 20,
            "eta_min": 1.0e-06,
        },
        "training": {
            "epochs": 20,
            "save_dir": config_path(save_dir),
            "feature_mode": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--grid", type=Path, default=ROOT / "configs/method_grid.json")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--source-methods", nargs="*", default=None)
    parser.add_argument("--target-methods", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/experiments")
    parser.add_argument("--brats-data-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_path = arg_path(args.grid)
    if grid_path is None:
        raise ValueError("grid path is required")
    grid = read_json(grid_path)
    targets = args.targets or grid["targets"]
    source_methods = args.source_methods or grid["source_methods"]
    target_methods = args.target_methods or grid["target_methods"]
    brats_data_root = arg_path(args.brats_data_root, env_path("BRATS_DATA_ROOT", DEFAULT_BRATS_DATA_ROOT))
    if brats_data_root is None:
        raise ValueError("brats_data_root is required")
    out_dir = arg_path(args.out_dir, ROOT / "configs/experiments" / f"b{args.budget}")
    manifest_path = arg_path(args.manifest, ROOT / "results" / f"experiment_manifest_b{args.budget}.json")
    output_root = args.output_root.expanduser().resolve() if args.output_root.is_absolute() else (ROOT / args.output_root).resolve()
    if out_dir is None or manifest_path is None:
        raise ValueError("out_dir and manifest_path are required")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for target in targets:
        methods_for_target = ["none", *target_methods] if "full_source" in source_methods else target_methods
        for source_method in source_methods:
            for target_method in methods_for_target:
                if not default_pair_allowed(source_method, target_method, args.budget):
                    continue
                config = config_for(target, source_method, target_method, args.budget, brats_data_root, output_root)
                row_id = run_id(target, source_method, target_method, args.budget)
                if config is None:
                    raise FileNotFoundError(f"{row_id}: required generated split is missing or has the wrong count")
                out_path = out_dir / f"{row_id}.yaml"
                with out_path.open("w", encoding="utf-8") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
                rows.append({
                    "run_id": row_id,
                    "budget": args.budget,
                    "target": target,
                    "source_method": source_method,
                    "target_method": target_method,
                    "config_path": str(out_path.relative_to(ROOT)),
                    "save_dir": config["training"]["save_dir"],
                    "domains": [domain["split_txt"] for domain in config["data"]["domains"]],
                })

    manifest = {
        "budget": args.budget,
        "generated_count": len(rows),
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"generated={len(rows)}")
    print(manifest_path)


if __name__ == "__main__":
    main()
