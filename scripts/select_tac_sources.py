#!/usr/bin/env python3
"""Generate TAC source selections from frozen selection-time caches.

TAC is a single target-conditioned selector. It does not branch on target name,
does not use downstream Dice, and does not use reference subset membership.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ("TCGA_LGG", "C4", "C5", "TCGA_GBM")


@dataclass(frozen=True)
class TACConfig:
    budget: int = 150
    top_utility_pool: int = 500
    top_signal_pool: int = 220
    top_facility_pool: int = 250
    gate_power: float = 4.0
    gate_temperature: float = 0.05
    sparse_rho0: float = 0.60
    sparse_tau: float = 0.08
    coverage_alignment0: float = 0.28
    coverage_alignment_tau: float = 0.04
    barrier_weight: float = 10.0
    redundancy_lambda: float = 0.08
    sparse_match_weight: float = 0.05
    sparse_low_density_weight: float = 0.02
    consensus_multi_quota: int = 45
    consensus_facility_intercept: float = 50.0
    consensus_facility_late_slope: float = 60.0
    consensus_signal_intercept: float = 15.0
    consensus_signal_late_slope: float = 30.0


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_ids(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


def top_set(values: np.ndarray, k: int) -> set[int]:
    k = min(int(k), len(values))
    return set(np.argsort(-values, kind="mergesort")[:k].tolist())


def overlap_fraction(left: set[int], right: set[int]) -> float:
    return len(left & right) / max(1, min(len(left), len(right)))


def agreement(*sets: set[int]) -> float:
    union = set().union(*sets)
    if not union:
        return 0.0
    return float(sum(int(sum(item in value for value in sets) >= 2) for item in union) / len(union))


def mask_from_indices(length: int, indices: set[int] | list[int]) -> np.ndarray:
    mask = np.zeros(length, dtype=np.float32)
    if indices:
        mask[np.asarray(sorted(indices), dtype=np.int64)] = 1.0
    return mask


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path, budget: int | None) -> TACConfig:
    values = load_json(path)
    config = TACConfig(**values)
    if budget is not None:
        config = replace(config, budget=int(budget))
    if config.budget <= 0:
        raise ValueError(f"budget must be positive, got {config.budget}")
    return config


def facility_rank_score(cache_dir: Path, source_ids: list[str]) -> np.ndarray:
    order = read_ids(cache_dir / "target_coverage_order.txt")
    positions = {subject_id: idx for idx, subject_id in enumerate(source_ids)}
    score = np.zeros(len(source_ids), dtype=np.float32)
    denom = max(1, len(order) - 1)
    for rank, subject_id in enumerate(order):
        idx = positions.get(subject_id)
        if idx is not None:
            score[idx] = 1.0 - float(rank) / float(denom)
    return score


def load_arrays(cache_root: Path, target: str) -> dict[str, Any]:
    cache_dir = cache_root / target
    source_ids = read_ids(cache_dir / "source_ids.txt")
    embeddings = l2_normalize(np.load(cache_dir / "source_embeddings_l2.npy").astype(np.float32))
    density_raw = np.load(cache_dir / "source_density.npy").astype(np.float32)
    clusters = np.load(cache_dir / "cluster_subject_labels.npy").astype(np.int64)
    query_similarity = np.maximum(np.load(cache_dir / "query_similarity.npy").astype(np.float32), 0.0)

    reliability = normalize(np.load(cache_dir / "reliability.npy").astype(np.float32))
    influence = normalize(np.load(cache_dir / "influence.npy").astype(np.float32))
    late = normalize(np.load(cache_dir / "late_influence.npy").astype(np.float32))
    match = normalize(np.load(cache_dir / "distribution_match.npy").astype(np.float32))
    density = normalize(np.load(cache_dir / "density_score.npy").astype(np.float32))
    facility = facility_rank_score(cache_dir, source_ids)

    align = normalize(np.load(cache_dir / "facility_align.npy").astype(np.float32))

    expected = len(source_ids)
    for name, values in {
        "embeddings": embeddings,
        "density_raw": density_raw,
        "clusters": clusters,
        "query_similarity": query_similarity,
        "reliability": reliability,
        "influence": influence,
        "late": late,
        "match": match,
        "density": density,
        "facility": facility,
        "align": align,
    }.items():
        if values.shape[0] != expected:
            raise ValueError(f"{target}: {name} has first dimension {values.shape[0]}, expected {expected}")

    return {
        "source_ids": source_ids,
        "embeddings": embeddings,
        "density_raw": density_raw,
        "clusters": clusters,
        "Q": query_similarity,
        "R": reliability,
        "I": influence,
        "L": late,
        "M": match,
        "D": density,
        "F": facility,
        "A": align,
    }


def compute_diagnostics(arrays: dict[str, Any], config: TACConfig) -> dict[str, Any]:
    reliability = arrays["R"]
    influence = arrays["I"]
    late = arrays["L"]
    facility = arrays["F"]
    query_similarity = arrays["Q"]

    reliability_top = top_set(reliability, config.top_signal_pool)
    influence_top = top_set(influence, config.top_signal_pool)
    late_top = top_set(late, config.top_signal_pool)
    late_distinctness = 1.0 - overlap_fraction(influence_top, late_top)
    active_influence = late if late_distinctness > 0.08 else influence
    active_top = top_set(active_influence, config.top_signal_pool)
    facility_top = top_set(facility, config.top_facility_pool)

    coverage_trust = float(np.mean(np.max(query_similarity, axis=1)))
    sparse_strength = float(sigmoid((config.sparse_rho0 - coverage_trust) / config.sparse_tau))
    facility_alignment = float(
        (overlap_fraction(reliability_top, facility_top) + overlap_fraction(active_top, facility_top)) / 2.0
    )
    coverage_strength = float(
        (1.0 - sparse_strength)
        * sigmoid((facility_alignment - config.coverage_alignment0) / config.coverage_alignment_tau)
    )
    consensus_strength = float(
        (1.0 - sparse_strength)
        * (1.0 - coverage_strength)
        * agreement(reliability_top, active_top, facility_top)
    )

    raw_strengths = np.asarray([sparse_strength, coverage_strength, consensus_strength], dtype=np.float64)
    powered = np.power(np.maximum(raw_strengths, 1e-8), config.gate_power)
    gates = powered / (powered.sum() + 1e-12)

    return {
        "R_top": reliability_top,
        "H": active_influence,
        "H_top": active_top,
        "F_top": facility_top,
        "coverage_trust": coverage_trust,
        "facility_alignment": facility_alignment,
        "late_distinctness": float(late_distinctness),
        "raw_strengths": {
            "sparse": sparse_strength,
            "coverage": coverage_strength,
            "consensus": consensus_strength,
        },
        "gates": {
            "sparse": float(gates[0]),
            "coverage": float(gates[1]),
            "consensus": float(gates[2]),
        },
    }


def gate_activation(gate_value: float, config: TACConfig) -> float:
    return float(sigmoid((float(gate_value) - 0.5) / config.gate_temperature))


def quota_seed(candidate_indices: list[int], cluster_labels: np.ndarray, score: np.ndarray, budget: int) -> list[int]:
    if int(budget) <= 0 or not candidate_indices:
        return []
    candidate_indices = list(candidate_indices)
    labels_sub = cluster_labels[np.asarray(candidate_indices, dtype=np.int64)]
    unique_labels = sorted(set(labels_sub.tolist()))
    mass: dict[int, float] = {}
    for label in unique_labels:
        label_indices = [candidate_indices[pos] for pos, value in enumerate(labels_sub.tolist()) if value == label]
        mass[label] = float(np.maximum(score[np.asarray(label_indices, dtype=np.int64)], 0.0).sum()) + 1e-8
    total_mass = sum(mass.values()) if mass else 1.0
    quota = {label: max(1, int(np.floor(int(budget) * mass[label] / total_mass))) for label in unique_labels}
    while sum(quota.values()) > int(budget):
        label = max(quota, key=lambda key: quota[key])
        if quota[label] > 1:
            quota[label] -= 1
        else:
            break
    while sum(quota.values()) < min(int(budget), len(candidate_indices)):
        fractional = {label: (int(budget) * mass[label] / total_mass) - quota[label] for label in unique_labels}
        quota[max(fractional, key=fractional.get)] += 1

    seeded: list[int] = []
    for label in unique_labels:
        label_indices = [candidate_indices[pos] for pos, value in enumerate(labels_sub.tolist()) if value == label]
        label_indices = sorted(label_indices, key=lambda idx: float(score[idx]), reverse=True)
        seeded.extend(label_indices[: quota[label]])
    return sorted(set(seeded), key=lambda idx: float(score[idx]), reverse=True)[: int(budget)]


def consensus_components(arrays: dict[str, Any], diagnostics: dict[str, Any], config: TACConfig) -> dict[str, Any]:
    reliability = arrays["R"]
    active_influence = diagnostics["H"]
    density = arrays["D"]
    align = arrays["A"]
    late_distinctness = float(diagnostics["late_distinctness"])
    reliability_top = diagnostics["R_top"]
    active_top = diagnostics["H_top"]
    facility_top = diagnostics["F_top"]
    candidate = reliability_top | active_top | facility_top
    multi = {
        idx
        for idx in candidate
        if (idx in reliability_top) + (idx in active_top) + (idx in facility_top) >= 2
    }
    facility_member = mask_from_indices(len(reliability), facility_top)
    multi_member = mask_from_indices(len(reliability), multi)
    score = (
        (0.20 + 0.05 * late_distinctness) * reliability
        + (0.15 + 0.05 * late_distinctness) * active_influence
        + 0.15 * density
        + 0.20 * align
        + (0.20 - 0.05 * late_distinctness) * facility_member
        + (0.10 - 0.05 * late_distinctness) * multi_member
    ).astype(np.float32)
    quotas = {
        "multi": int(config.consensus_multi_quota),
        "facility_only": max(
            0,
            int(round(config.consensus_facility_intercept - config.consensus_facility_late_slope * late_distinctness)),
        ),
        "reliability_only": max(
            0,
            int(round(config.consensus_signal_intercept + config.consensus_signal_late_slope * late_distinctness)),
        ),
        "influence_only": max(
            0,
            int(round(config.consensus_signal_intercept + config.consensus_signal_late_slope * late_distinctness)),
        ),
    }
    groups = {
        "multi": sorted(multi),
        "facility_only": sorted(facility_top - reliability_top - active_top),
        "reliability_only": sorted(reliability_top - facility_top - active_top),
        "influence_only": sorted(active_top - facility_top - reliability_top),
    }
    return {
        "candidate": candidate,
        "facility_member": facility_member,
        "multi_member": multi_member,
        "score": score,
        "groups": groups,
        "quotas": quotas,
    }


def select_tac(
    arrays: dict[str, Any],
    diagnostics: dict[str, Any],
    config: TACConfig,
) -> tuple[list[int], np.ndarray, dict[str, Any]]:
    reliability = arrays["R"]
    match = arrays["M"]
    embeddings = arrays["embeddings"]
    density_raw = arrays["density_raw"]
    gates = diagnostics["gates"]
    sparse_activation = gate_activation(gates["sparse"], config)
    coverage_activation = gate_activation(gates["coverage"], config)
    consensus_activation = gate_activation(gates["consensus"], config)
    components = consensus_components(arrays, diagnostics, config)

    utility_pool = set(
        np.argsort(-reliability, kind="mergesort")[: config.top_utility_pool].tolist()
    )
    utility_member = mask_from_indices(len(reliability), utility_pool)
    low_density_member = (density_raw <= np.percentile(density_raw, 10.0)).astype(np.float32)
    sparse_score = (
        reliability
        + config.sparse_match_weight * match
        + config.sparse_low_density_weight * low_density_member
    ).astype(np.float32)

    base_score = (
        sparse_activation * (sparse_score - config.barrier_weight * (1.0 - utility_member))
        + coverage_activation
        * (reliability - config.barrier_weight * (1.0 - components["facility_member"]))
        + consensus_activation * components["score"]
    ).astype(np.float32)

    seed_scale = consensus_activation
    seeded: list[int] = []
    for name, candidate_indices in components["groups"].items():
        quota = int(round(seed_scale * components["quotas"][name]))
        seeded.extend(
            quota_seed(candidate_indices, arrays["clusters"], components["score"], quota)
        )
    seeded = sorted(set(seeded), key=lambda idx: float(components["score"][idx]), reverse=True)

    candidate_pool = set(utility_pool) | set(diagnostics["R_top"]) | set(diagnostics["H_top"]) | set(diagnostics["F_top"])
    allowed = np.zeros(len(reliability), dtype=bool)
    allowed[np.asarray(sorted(candidate_pool), dtype=np.int64)] = True

    sim = embeddings @ embeddings.T
    selected: list[int] = []
    selected_mask = np.zeros(len(reliability), dtype=bool)
    max_similarity = np.zeros(len(reliability), dtype=np.float32)
    for idx in seeded[: config.budget]:
        if not selected_mask[idx]:
            selected.append(idx)
            selected_mask[idx] = True
            max_similarity = np.maximum(max_similarity, sim[:, idx])

    while len(selected) < config.budget:
        value = base_score - config.redundancy_lambda * np.maximum(max_similarity, 0.0)
        value[selected_mask | ~allowed] = -np.inf
        best = int(np.argmax(value))
        if not np.isfinite(value[best]):
            break
        selected.append(best)
        selected_mask[best] = True
        max_similarity = np.maximum(max_similarity, sim[:, best])

    selected_set = set(selected)
    tac_diagnostics = {
        "objective": "tac_continuous_quota",
        "gate_activations": {
            "sparse": sparse_activation,
            "coverage": coverage_activation,
            "consensus": consensus_activation,
        },
        "dynamic_consensus_quotas": components["quotas"],
        "consensus_seed_scale": seed_scale,
        "seeded_count": len(seeded),
        "selected_counts": {
            "reliability_top": len(selected_set & diagnostics["R_top"]),
            "influence_top": len(selected_set & diagnostics["H_top"]),
            "facility_top": len(selected_set & diagnostics["F_top"]),
            "multi_signal": int(np.asarray(components["multi_member"])[selected].sum()) if selected else 0,
        },
    }
    return selected, base_score, tac_diagnostics


def signal_mean_table(arrays: dict[str, Any], selected: list[int]) -> dict[str, float]:
    selected_idx = np.asarray(selected, dtype=np.int64)
    signals = {name: arrays[name] for name in ["R", "I", "L", "M", "D", "F", "A"]}
    return {
        name: float(np.mean(values[selected_idx])) if selected_idx.size else 0.0
        for name, values in signals.items()
    }


def summarize_target(
    target: str,
    config: TACConfig,
    cache_root: Path,
    data_root: Path,
    results_root: Path,
) -> dict[str, Any]:
    arrays = load_arrays(cache_root, target)
    if config.budget > len(arrays["source_ids"]):
        raise ValueError(
            f"{target}: budget {config.budget} exceeds source pool size {len(arrays['source_ids'])}"
        )
    diagnostics = compute_diagnostics(arrays, config)
    selected, score, tac_diagnostics = select_tac(arrays, diagnostics, config)
    if len(selected) != config.budget:
        raise RuntimeError(f"{target}: selected {len(selected)} subjects, expected {config.budget}")
    diagnostics.update(tac_diagnostics)

    selected_ids = [arrays["source_ids"][idx] for idx in selected]
    split_dir = data_root / target / f"tac_{config.budget}"
    target_results = results_root / target
    write_ids(split_dir / "train_subjects.txt", selected_ids)
    write_ids(target_results / "selected_ids.txt", selected_ids)
    target_results.mkdir(parents=True, exist_ok=True)
    np.save(target_results / "selected_scores.npy", score[np.asarray(selected, dtype=np.int64)])

    public_diagnostics = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"R_top", "H_top", "F_top", "H"}
    }
    row = {
        "target": target,
        "source_split": str(split_dir / "train_subjects.txt"),
        "results_root": str(target_results),
        "selected_total": len(selected_ids),
        "diagnostics": public_diagnostics,
        "signal_means": signal_mean_table(arrays, selected),
    }
    (target_results / "run_config.json").write_text(json.dumps({
        "method": "TAC",
        "config": asdict(config),
        **row,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def write_summary(rows: list[dict[str, Any]], results_root: Path, config: TACConfig) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# TAC Source Selection Summary",
        "",
        "TAC uses one continuous target-conditioned objective for every target.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(asdict(config), indent=2, sort_keys=True),
        "```",
        "",
        "## Selected Subsets",
        "",
        "| Target | Selected | Sparse act. | Coverage act. | Consensus act. | R-top | H-top | F-top | Multi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        diag = row["diagnostics"]
        acts = diag["gate_activations"]
        counts = diag["selected_counts"]
        lines.append(
            f"| {row['target']} | {row['selected_total']} | "
            f"{acts['sparse']:.3f} | {acts['coverage']:.3f} | {acts['consensus']:.3f} | "
            f"{counts['reliability_top']} | {counts['influence_top']} | "
            f"{counts['facility_top']} | {counts['multi_signal']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- The selector reads only frozen target-conditioned signal caches.",
        "- The selector does not read target identity-specific reference splits, validation Dice, or test Dice.",
        "- Output source subsets are written under `data/source_splits/<target>/tac_<budget>/`.",
        "",
    ])
    (results_root / "TAC_SELECTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("all", *DEFAULT_TARGETS), default="all")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--selector-config", type=Path, default=ROOT / "configs/tac_selector.json")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/source_splits")
    parser.add_argument("--results-root", type=Path, default=None)
    return parser.parse_args()


def arg_path(value: Path | None, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def main() -> None:
    args = parse_args()
    selector_config = arg_path(args.selector_config)
    cache_root = arg_path(args.cache_root)
    data_root = arg_path(args.data_root)
    if selector_config is None or cache_root is None or data_root is None:
        raise ValueError("selector_config, cache_root, and data_root are required")
    config = load_config(selector_config, args.budget)
    results_root = arg_path(args.results_root, ROOT / "results/source_selection" / f"b{config.budget}")
    if results_root is None:
        raise ValueError("results_root is required")
    target_names = sorted(DEFAULT_TARGETS) if args.target == "all" else [args.target]
    missing = [target for target in target_names if not (cache_root / target).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing cache directories: {', '.join(missing)}")
    targets = target_names
    rows = [
        summarize_target(target, config, cache_root, data_root, results_root)
        for target in targets
    ]
    write_summary(rows, results_root, config)
    for row in rows:
        print(f"{row['target']}: selected {row['selected_total']} -> {row['source_split']}")
    print(results_root / "TAC_SELECTION_SUMMARY.md")


if __name__ == "__main__":
    main()
