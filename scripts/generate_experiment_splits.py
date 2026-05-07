#!/usr/bin/env python3
"""Generate all reviewer-facing target and source splits from reproducible inputs.

This script does not import prebuilt experiment split folders. It constructs the
target-acquisition splits, baseline source splits, ORIENT-style source splits,
and TAC source-target curation splits from subject lists, embeddings, and score
arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from select_tac_sources import (
    compute_diagnostics,
    load_arrays,
    load_config,
    load_target_support,
    select_tac,
    signal_mean_table,
    write_summary,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_SOURCE_ROOT = ROOT / "external/selection_inputs"
TARGET_ACTIVE_DIR = {
    "TCGA_LGG": "split_TCGA_LGG_active",
    "C4": "split_C4_active",
    "C5": "split_C5_active",
    "TCGA_GBM": "split_TCGA_GBM_active",
}
TARGETS = tuple(TARGET_ACTIVE_DIR)
TARGET_METHOD_DIR = {
    "random1": "random1_10",
    "random2": "random2_10",
    "random3": "random3_10",
    "coreset": "coreset_10",
    "lada": "lada_10",
    "aada": "aada_10",
    "ada_clue": "clue_10",
    "tac": "tac_10",
}


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_ids(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def active_dir(split_source_root: Path, target: str) -> Path:
    path = split_source_root / TARGET_ACTIVE_DIR[target]
    if not path.exists():
        raise FileNotFoundError(f"Missing active split directory for {target}: {path}")
    return path


def load_target_state(split_source_root: Path, target: str) -> dict[str, Any]:
    root = active_dir(split_source_root, target)
    ids = read_ids(root / "target_subject_ids.txt")
    vectors = l2_normalize(np.load(root / "target_subject_vecs.npy"))
    if len(ids) != vectors.shape[0]:
        raise ValueError(f"{target}: target_subject_ids and target_subject_vecs length mismatch")

    def required_score(filename: str) -> np.ndarray:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"{target}: missing required target score file {path}")
        values = np.load(path).astype(np.float32).reshape(-1)
        if values.shape[0] != len(ids):
            raise ValueError(f"{target}: {filename} length {values.shape[0]} != {len(ids)}")
        return values

    uncertainty_raw = required_score("target_uncertainty.npy")
    fg_score_raw = required_score("target_subject_fg_score.npy")
    lada_score_raw = required_score("lada_li_scores.npy")
    consistency_raw = required_score("target_consistency.npy")
    return {
        "root": root,
        "ids": ids,
        "vectors": vectors,
        "uncertainty": normalize(uncertainty_raw),
        "fg_score": normalize(fg_score_raw),
        "lada_score": normalize(lada_score_raw),
        "consistency": normalize(consistency_raw),
        "uncertainty_fg": normalize(uncertainty_raw * fg_score_raw),
    }


def greedy_kcenter(ids: list[str], vectors: np.ndarray, budget: int, priority: np.ndarray | None = None) -> list[str]:
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if budget > len(ids):
        raise ValueError(f"budget {budget} exceeds pool size {len(ids)}")
    vectors = l2_normalize(vectors)
    priority = normalize(priority) if priority is not None else np.zeros(len(ids), dtype=np.float32)
    selected: list[int] = [int(np.argmax(priority))] if np.any(priority > 0) else [0]
    min_distance = 1.0 - np.maximum(vectors @ vectors[selected[0]], -1.0)
    while len(selected) < budget:
        score = min_distance + 0.05 * priority
        score[np.asarray(selected, dtype=np.int64)] = -np.inf
        idx = int(np.argmax(score))
        selected.append(idx)
        min_distance = np.minimum(min_distance, 1.0 - np.maximum(vectors @ vectors[idx], -1.0))
    return [ids[idx] for idx in selected]


def weighted_kmeans_select(
    ids: list[str],
    vectors: np.ndarray,
    weights: np.ndarray,
    budget: int,
    seed: int = 2025,
) -> list[str]:
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if budget > len(ids):
        raise ValueError(f"budget {budget} exceeds pool size {len(ids)}")
    vectors = l2_normalize(vectors)
    weights = np.maximum(np.asarray(weights, dtype=np.float32).reshape(-1), 1e-6)
    if weights.shape[0] != len(ids):
        raise ValueError(f"weights length {weights.shape[0]} != ids length {len(ids)}")

    kmeans = KMeans(
        n_clusters=int(budget),
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=int(seed),
    )
    kmeans.fit(vectors, sample_weight=weights)
    centers = kmeans.cluster_centers_.astype(np.float32)
    weight_term = normalize(weights)

    selected: list[int] = []
    used: set[int] = set()
    for cluster_idx in range(int(budget)):
        distances = np.linalg.norm(vectors - centers[cluster_idx], axis=1)
        distance_term = 1.0 - normalize(distances)
        priority = 0.7 * distance_term + 0.3 * weight_term
        order = np.argsort(-priority, kind="mergesort")
        for idx_value in order:
            idx = int(idx_value)
            if idx in used or int(kmeans.labels_[idx]) != cluster_idx:
                continue
            selected.append(idx)
            used.add(idx)
            break

    if len(selected) < int(budget):
        for idx_value in np.argsort(-weights, kind="mergesort"):
            idx = int(idx_value)
            if idx in used:
                continue
            selected.append(idx)
            used.add(idx)
            if len(selected) == int(budget):
                break
    return [ids[idx] for idx in selected[: int(budget)]]


def generate_target_eval(split_source_root: Path, target: str) -> None:
    root = active_dir(split_source_root, target)
    dst = ROOT / "data/target_splits" / target / "eval"
    for filename in ("val_subjects.txt", "test_subjects.txt"):
        values = read_ids(root / filename)
        write_ids(dst / filename, values)


def generate_target_split(split_source_root: Path, target: str, method: str) -> None:
    state = load_target_state(split_source_root, target)
    ids = state["ids"]
    vectors = state["vectors"]
    dst = ROOT / "data/target_splits" / target / TARGET_METHOD_DIR[method] / "train_subjects.txt"
    if method.startswith("random"):
        seed = int(method.replace("random", ""))
        rng = np.random.default_rng(20260506 + 1009 * seed + 31 * TARGETS.index(target))
        selected = rng.choice(np.asarray(ids, dtype=object), size=10, replace=False).astype(str).tolist()
    elif method == "coreset":
        selected = greedy_kcenter(ids, vectors, 10)
    elif method == "lada":
        score = state["uncertainty"] + 0.50 * state["fg_score"] + state["lada_score"]
        selected = [ids[idx] for idx in np.argsort(-score, kind="mergesort")[:10]]
    elif method == "aada":
        source_vectors = l2_normalize(np.load(ROOT / "cache" / target / "source_embeddings_l2.npy"))
        source_centroid = l2_normalize(source_vectors.mean(axis=0, keepdims=True))[0]
        target_domainness = normalize(1.0 - np.maximum(vectors @ source_centroid, -1.0))
        score = target_domainness + 0.50 * state["uncertainty"] + 0.25 * state["fg_score"]
        selected = [ids[idx] for idx in np.argsort(-score, kind="mergesort")[:10]]
    elif method == "ada_clue":
        priority = state["uncertainty"] + 0.50 * state["fg_score"]
        selected = greedy_kcenter(ids, vectors, 10, priority=priority)
    elif method == "tac":
        weights = state["uncertainty_fg"] + 0.50 * state["lada_score"] + 0.15 * state["consistency"]
        selected = weighted_kmeans_select(ids, vectors, weights, 10, seed=2025)
    else:
        raise KeyError(method)
    write_ids(dst, selected)


def source_ids(target: str) -> list[str]:
    return read_ids(ROOT / "cache" / target / "source_ids.txt")


def source_vectors(target: str) -> np.ndarray:
    return l2_normalize(np.load(ROOT / "cache" / target / "source_embeddings_l2.npy"))


def generate_baseline_source_split(target: str, method: str, budget: int) -> None:
    ids = source_ids(target)
    dst_dir = ROOT / "data/source_splits" / target
    if method == "full_source":
        write_ids(dst_dir / "full_source/train_subjects.txt", ids)
        return
    if method.startswith("random"):
        seed = int(method.replace("random", ""))
        rng = np.random.default_rng(20260506 + 1009 * seed + 17 * TARGETS.index(target) + budget)
        selected = rng.choice(np.asarray(ids, dtype=object), size=budget, replace=False).astype(str).tolist()
        write_ids(dst_dir / f"{method}_{budget}/train_subjects.txt", selected)
        return
    if method == "coreset":
        selected = greedy_kcenter(ids, source_vectors(target), budget)
        write_ids(dst_dir / f"coreset_{budget}/train_subjects.txt", selected)
        return
    raise KeyError(method)


def target_vectors_for_split(split_source_root: Path, target: str, target_method: str) -> np.ndarray:
    state = load_target_state(split_source_root, target)
    positions = {subject_id: idx for idx, subject_id in enumerate(state["ids"])}
    selected = read_ids(ROOT / "data/target_splits" / target / TARGET_METHOD_DIR[target_method] / "train_subjects.txt")
    missing = [subject_id for subject_id in selected if subject_id not in positions]
    if missing:
        raise ValueError(f"{target} {target_method}: target IDs missing from active target state: {missing[:3]}")
    return state["vectors"][np.asarray([positions[subject_id] for subject_id in selected], dtype=np.int64)]


def generate_orient_source_split(split_source_root: Path, target: str, target_method: str, budget: int) -> None:
    ids = source_ids(target)
    src_vecs = source_vectors(target)
    tgt_vecs = target_vectors_for_split(split_source_root, target, target_method)
    query_sim = np.maximum(src_vecs @ l2_normalize(tgt_vecs).T, 0.0)
    src_sim = np.maximum(src_vecs @ src_vecs.T, 0.0)
    covered = np.zeros(query_sim.shape[1], dtype=np.float32)
    max_redundancy = np.zeros(len(ids), dtype=np.float32)
    selected: list[int] = []
    selected_mask = np.zeros(len(ids), dtype=bool)
    for _ in range(budget):
        gain = np.maximum(query_sim, covered[None, :]).sum(axis=1) - covered.sum()
        value = gain - 0.08 * max_redundancy
        value[selected_mask] = -np.inf
        idx = int(np.argmax(value))
        selected.append(idx)
        selected_mask[idx] = True
        covered = np.maximum(covered, query_sim[idx])
        max_redundancy = np.maximum(max_redundancy, src_sim[:, idx])
    write_ids(
        ROOT / "data/source_splits" / target / f"orient_{target_method}_{budget}/train_subjects.txt",
        [ids[idx] for idx in selected],
    )


def generate_tac_source_split(target: str, budget: int, selector_config: Path, split_source_root: Path) -> dict[str, Any]:
    config = load_config(selector_config, budget)
    target_support = load_target_support(split_source_root, ROOT / "data/target_splits", target)
    arrays = load_arrays(ROOT / "cache", target, config=config, target_support=target_support)
    diagnostics = compute_diagnostics(arrays, config)
    selected, score, tac_diagnostics = select_tac(arrays, diagnostics, config)
    if len(selected) != budget:
        raise RuntimeError(f"{target}: TAC selected {len(selected)} subjects, expected {budget}")
    diagnostics.update(tac_diagnostics)
    selected_ids = [arrays["source_ids"][idx] for idx in selected]
    split_dir = ROOT / "data/source_splits" / target / f"tac_{budget}"
    result_dir = ROOT / "results/source_selection" / f"b{budget}" / target
    write_ids(split_dir / "train_subjects.txt", selected_ids)
    write_ids(result_dir / "selected_ids.txt", selected_ids)
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "selected_scores.npy", score[np.asarray(selected, dtype=np.int64)])
    public_diagnostics = {key: value for key, value in diagnostics.items() if key not in {"R_top", "H_top", "F_top", "H"}}
    row = {
        "target": target,
        "source_split": str(split_dir / "train_subjects.txt"),
        "results_root": str(result_dir),
        "selected_total": len(selected_ids),
        "diagnostics": public_diagnostics,
        "signal_means": signal_mean_table(arrays, selected),
    }
    (result_dir / "run_config.json").write_text(json.dumps({"method": "TAC", "config": asdict(config), **row}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=int, default=[50, 100, 150, 200])
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument(
        "--split-source-root",
        type=Path,
        default=Path(os.environ.get("SPLIT_SOURCE_ROOT", str(DEFAULT_SPLIT_SOURCE_ROOT))),
    )
    parser.add_argument("--selector-config", type=Path, default=ROOT / "configs/tac_selector.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_source_root = args.split_source_root.expanduser().resolve()
    selector_config = args.selector_config.expanduser().resolve()
    target_methods = ("random1", "random2", "random3", "coreset", "lada", "aada", "ada_clue", "tac")
    baseline_source_methods = ("full_source", "random1", "random2", "random3", "coreset")
    orient_target_methods = ("random1", "random2", "random3", "coreset", "lada", "aada", "ada_clue")
    tac_rows_by_budget: dict[int, list[dict[str, Any]]] = {budget: [] for budget in args.budgets}

    for target in args.targets:
        generate_target_eval(split_source_root, target)
        for method in target_methods:
            generate_target_split(split_source_root, target, method)
        for method in baseline_source_methods:
            if method == "full_source":
                generate_baseline_source_split(target, method, args.budgets[0])
                continue
            for budget in args.budgets:
                generate_baseline_source_split(target, method, budget)
        for budget in args.budgets:
            tac_rows_by_budget[budget].append(generate_tac_source_split(target, budget, selector_config, split_source_root))
            for target_method in orient_target_methods:
                generate_orient_source_split(split_source_root, target, target_method, budget)

    for budget, rows in tac_rows_by_budget.items():
        config = load_config(selector_config, budget)
        write_summary(rows, ROOT / "results/source_selection" / f"b{budget}", config)
    print(f"generated splits for targets={','.join(args.targets)} budgets={','.join(map(str, args.budgets))}")


if __name__ == "__main__":
    main()
