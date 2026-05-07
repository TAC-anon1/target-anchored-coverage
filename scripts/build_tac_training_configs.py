#!/usr/bin/env python3
"""Build EfficientVit training configs for TAC source-target curation outputs."""

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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_ids(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def env_path(name: str, default: Path) -> Path:
    value = Path(os.environ.get(name, str(default))).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (ROOT / value).resolve()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def arg_path(value: Path | None, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


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


def build_config(
    target: str,
    target_spec: dict[str, Any],
    budget: int,
    brats_data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    source_split_dir = (ROOT / "data/source_splits" / target / f"tac_{budget}").resolve()
    source_split_file = source_split_dir / "train_subjects.txt"
    if not source_split_file.exists():
        raise FileNotFoundError(f"{target}: missing TAC source split {source_split_file}")
    source_count = count_ids(source_split_file)
    if source_count != budget:
        raise ValueError(f"{target}: expected {budget} source ids, found {source_count} in {source_split_file}")

    target_train_split = repo_path(target_spec["target_train_split"])
    target_eval_split = repo_path(target_spec["target_eval_split"])
    for required in [
        target_train_split / "train_subjects.txt",
        target_eval_split / "val_subjects.txt",
        target_eval_split / "test_subjects.txt",
    ]:
        if not required.exists():
            raise FileNotFoundError(f"{target}: missing required split file {required}")

    support_tag = target_spec["target_support_tag"]
    save_dir = output_root / f"tac_b{budget}" / f"{target}_S_TAC{budget}_T_{support_tag}_10"
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
            "domains": [
                {
                    "name": "BraTS21_Source",
                    "path": str(brats_data_root),
                    "split": "train",
                    "split_txt": config_path(source_split_dir),
                },
                {
                    "name": "BraTS21_T_train",
                    "path": str(brats_data_root),
                    "split": "train",
                    "split_txt": config_path(target_train_split),
                },
            ],
            "val": {
                "path": str(brats_data_root),
                "split": "val",
                "split_txt": config_path(target_eval_split),
            },
            "test": {
                "path": str(brats_data_root),
                "split": "test",
                "split_txt": config_path(target_eval_split),
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
    parser.add_argument("--target", default="all", help="Target name or all.")
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--targets-config", type=Path, default=ROOT / "configs/targets.json")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--brats-data-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets_config = arg_path(args.targets_config)
    if targets_config is None:
        raise ValueError("targets config path is required")
    targets = read_json(targets_config)
    if args.target != "all":
        if args.target not in targets:
            raise KeyError(f"Unknown target {args.target}; choices: {', '.join(sorted(targets))}")
        targets = {args.target: targets[args.target]}

    brats_data_root = arg_path(args.brats_data_root, env_path("BRATS_DATA_ROOT", DEFAULT_BRATS_DATA_ROOT))
    if brats_data_root is None:
        raise ValueError("brats_data_root is required")
    output_root = args.output_root.expanduser().resolve() if args.output_root.is_absolute() else (ROOT / args.output_root).resolve()
    out_dir = arg_path(args.out_dir, ROOT / "configs/generated" / f"b{args.budget}")
    manifest_path = arg_path(args.manifest, ROOT / "results" / f"training_config_manifest_b{args.budget}.json")
    if out_dir is None or manifest_path is None:
        raise ValueError("out_dir and manifest_path are required")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = []
    for target, target_spec in targets.items():
        config = build_config(target, target_spec, args.budget, brats_data_root, output_root)
        support_tag = target_spec["target_support_tag"]
        out_path = out_dir / f"train_config_BraTS2021_{target}_S_TAC_{args.budget}_T_{support_tag}.yaml"
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
        row = {
            "target": target,
            "budget": args.budget,
            "config_path": str(out_path.relative_to(ROOT)),
            "source_split": config["data"]["domains"][0]["split_txt"],
            "target_train_split": config["data"]["domains"][1]["split_txt"],
            "save_dir": config["training"]["save_dir"],
        }
        manifest.append(row)
        print(f"{target}: {out_path} -> {config['training']['save_dir']}")

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
