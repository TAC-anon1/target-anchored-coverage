#!/usr/bin/env python3
"""Generate the TAC source-curation stage from primitive selection-time inputs.

TAC means Target-Anchored Coverage for Source-Target Curation. The full TAC
pipeline first selects a target support set, then uses that support as the
anchor for source curation. This script implements the deterministic
source-curation stage. It recomputes target-conditioned source signals from the
generated target support, does not branch on target name, does not use
downstream Dice, and does not use reference subset membership.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ("TCGA_LGG", "C4", "C5", "TCGA_GBM")
TARGET_ACTIVE_DIR = {
    "TCGA_LGG": "split_TCGA_LGG_active",
    "C4": "split_C4_active",
    "C5": "split_C5_active",
    "TCGA_GBM": "split_TCGA_GBM_active",
}


@dataclass(frozen=True)
class TACConfig:
    budget: int = 150
    top_utility_pool: int = 500
    top_signal_pool: int = 220
    top_facility_pool: int = 250
    facility_backend: str = "auto"
    facility_magnification_eta: float = 1.0
    facility_stop_if_negative_gain: bool = False
    facility_show_progress: bool = False
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


def top_indices(values: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), len(values))
    return np.argsort(-values, kind="mergesort")[:k].astype(np.int64)


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


def compute_reliability(cache_dir: Path, expected: int) -> np.ndarray:
    spec = load_json(cache_dir / "reliability_spec.json")
    component_files = spec["component_files"]
    weights = spec["weights"]
    score = np.zeros(expected, dtype=np.float32)
    for component in ("P", "G", "M"):
        values = np.load(cache_dir / component_files[component]).astype(np.float32).reshape(-1)
        if values.shape[0] != expected:
            raise ValueError(
                f"{cache_dir.name}: reliability component {component} has length {values.shape[0]}, expected {expected}"
            )
        score += float(weights[component]) * values
    return normalize(score)


def target_acquisition_weights(target_root: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = read_ids(target_root / "target_subject_ids.txt")
    vectors = l2_normalize(np.load(target_root / "target_subject_vecs.npy").astype(np.float32))
    if len(ids) != vectors.shape[0]:
        raise ValueError(f"{target_root}: target_subject_ids and target_subject_vecs length mismatch")

    def required_score(filename: str) -> np.ndarray:
        path = target_root / filename
        if not path.exists():
            raise FileNotFoundError(f"missing target score file {path}")
        values = np.load(path).astype(np.float32).reshape(-1)
        if values.shape[0] != len(ids):
            raise ValueError(f"{path}: length {values.shape[0]} != target ids length {len(ids)}")
        return values

    uncertainty = required_score("target_uncertainty.npy")
    foreground = required_score("target_subject_fg_score.npy")
    local_inconsistency = required_score("lada_li_scores.npy")
    consistency = required_score("target_consistency.npy")
    weights = (
        normalize(uncertainty * foreground)
        + 0.50 * normalize(local_inconsistency)
        + 0.15 * normalize(consistency)
    ).astype(np.float32)
    return ids, vectors, weights


def load_target_support(
    split_source_root: Path,
    target_split_root: Path,
    target: str,
) -> dict[str, Any]:
    target_root = split_source_root / TARGET_ACTIVE_DIR[target]
    selected_path = target_split_root / target / "tac_10" / "train_subjects.txt"
    if not selected_path.exists():
        raise FileNotFoundError(
            f"{target}: missing TAC target support split {selected_path}. "
            "Run scripts/generate_experiment_splits.py first, or provide --target-split-root."
        )
    all_ids, all_vectors, all_weights = target_acquisition_weights(target_root)
    positions = {subject_id: idx for idx, subject_id in enumerate(all_ids)}
    selected_ids = read_ids(selected_path)
    missing = [subject_id for subject_id in selected_ids if subject_id not in positions]
    if missing:
        raise ValueError(f"{target}: selected target IDs missing from target state: {missing[:3]}")
    selected_positions = np.asarray([positions[subject_id] for subject_id in selected_ids], dtype=np.int64)
    return {
        "ids": selected_ids,
        "vectors": all_vectors[selected_positions],
        "weights": all_weights[selected_positions],
    }


def compute_source_density(embeddings: np.ndarray, neighbors: int = 20) -> np.ndarray:
    similarity = np.maximum(embeddings @ embeddings.T, 0.0).astype(np.float32)
    np.fill_diagonal(similarity, -np.inf)
    k = min(int(neighbors), max(1, similarity.shape[0] - 1))
    top = np.partition(similarity, -k, axis=1)[:, -k:]
    top[~np.isfinite(top)] = 0.0
    return top.mean(axis=1).astype(np.float32)


def compute_source_clusters(embeddings: np.ndarray, cluster_count: int = 32, seed: int = 2025) -> np.ndarray:
    k = min(int(cluster_count), embeddings.shape[0])
    if k <= 1:
        return np.zeros(embeddings.shape[0], dtype=np.int64)
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=int(seed),
    )
    return model.fit_predict(embeddings).astype(np.int64)


def numpy_target_facility_order(
    source_similarity: np.ndarray,
    query_similarity: np.ndarray,
    budget: int,
    redundancy_lambda: float,
) -> list[int]:
    covered = np.zeros(query_similarity.shape[1], dtype=np.float32)
    max_redundancy = np.zeros(source_similarity.shape[0], dtype=np.float32)
    selected: list[int] = []
    selected_mask = np.zeros(source_similarity.shape[0], dtype=bool)
    for _ in range(min(int(budget), source_similarity.shape[0])):
        gain = np.maximum(query_similarity, covered[None, :]).sum(axis=1) - covered.sum()
        value = gain - float(redundancy_lambda) * max_redundancy
        value[selected_mask] = -np.inf
        idx = int(np.argmax(value))
        if not np.isfinite(value[idx]):
            break
        selected.append(idx)
        selected_mask[idx] = True
        covered = np.maximum(covered, query_similarity[idx])
        max_redundancy = np.maximum(max_redundancy, source_similarity[:, idx])
    return selected


def submodlib_target_facility_order(
    source_similarity: np.ndarray,
    query_similarity: np.ndarray,
    budget: int,
    eta: float,
    stop_if_negative_gain: bool,
    show_progress: bool,
) -> list[int]:
    from submodlib.functions.facilityLocationMutualInformation import (  # type: ignore[import-not-found]
        FacilityLocationMutualInformationFunction,
    )

    source_similarity = np.ascontiguousarray(source_similarity, dtype=np.float32)
    query_similarity = np.ascontiguousarray(query_similarity, dtype=np.float32)
    objective = FacilityLocationMutualInformationFunction(
        n=source_similarity.shape[0],
        num_queries=query_similarity.shape[1],
        data_sijs=source_similarity,
        query_sijs=query_similarity,
        magnificationEta=float(eta),
    )
    result = objective.maximize(
        budget=min(int(budget), source_similarity.shape[0]),
        optimizer="LazyGreedy",
        stopIfNegativeGain=bool(stop_if_negative_gain),
        show_progress=bool(show_progress),
    )
    return [int(idx) for idx, _gain in result]


def target_facility_order(
    embeddings: np.ndarray,
    query_similarity: np.ndarray,
    config: TACConfig,
) -> tuple[list[int], str]:
    backend = str(config.facility_backend).lower()
    if backend not in {"auto", "numpy", "submodlib"}:
        raise ValueError(f"facility_backend must be one of auto, numpy, submodlib; got {config.facility_backend!r}")
    source_similarity = np.maximum(embeddings @ embeddings.T, 0.0).astype(np.float32)
    if backend in {"auto", "submodlib"}:
        try:
            order = submodlib_target_facility_order(
                source_similarity=source_similarity,
                query_similarity=query_similarity,
                budget=config.top_facility_pool,
                eta=config.facility_magnification_eta,
                stop_if_negative_gain=config.facility_stop_if_negative_gain,
                show_progress=config.facility_show_progress,
            )
            return order, "submodlib"
        except (ImportError, ModuleNotFoundError) as exc:
            if backend == "submodlib":
                raise ImportError(
                    "facility_backend='submodlib' requires the optional submodlib package. "
                    "Install submodlib, or set facility_backend to 'auto' or 'numpy'."
                ) from exc
    order = numpy_target_facility_order(
        source_similarity=source_similarity,
        query_similarity=query_similarity,
        budget=config.top_facility_pool,
        redundancy_lambda=config.redundancy_lambda,
    )
    return order, "numpy"


def facility_rank_score_from_indices(length: int, order: list[int]) -> np.ndarray:
    score = np.zeros(length, dtype=np.float32)
    denom = max(1, len(order) - 1)
    for rank, idx in enumerate(order):
        score[int(idx)] = 1.0 - float(rank) / float(denom)
    return score


def compute_target_conditioned_signals(
    source_ids: list[str],
    embeddings: np.ndarray,
    target_support: dict[str, Any],
    config: TACConfig,
) -> dict[str, np.ndarray]:
    target_vectors = l2_normalize(target_support["vectors"])
    target_weights = np.maximum(np.asarray(target_support["weights"], dtype=np.float32).reshape(-1), 1e-6)
    if target_vectors.shape[0] != target_weights.shape[0]:
        raise ValueError("target support vectors and weights length mismatch")
    query_similarity = np.maximum(embeddings @ target_vectors.T, 0.0).astype(np.float32)

    match_k = min(8, query_similarity.shape[1])
    top_local = np.argsort(-query_similarity, axis=1, kind="mergesort")[:, :match_k]
    row_weights = target_weights[top_local]
    row_weights = row_weights / (row_weights.sum(axis=1, keepdims=True) + 1e-12)
    match_raw = (query_similarity[np.arange(query_similarity.shape[0])[:, None], top_local] * row_weights).sum(axis=1)
    match = normalize(match_raw)

    facility_order, facility_backend = target_facility_order(
        embeddings,
        query_similarity,
        config=config,
    )
    facility = facility_rank_score_from_indices(len(source_ids), facility_order)
    if facility_order:
        centroid = embeddings[np.asarray(facility_order, dtype=np.int64)].sum(axis=0, keepdims=True)
        centroid = l2_normalize(centroid)[0]
        align = normalize(np.maximum(embeddings @ centroid, 0.0))
    else:
        align = np.zeros(len(source_ids), dtype=np.float32)

    return {
        "Q": query_similarity,
        "M": match,
        "F": facility,
        "A": align,
        "facility_backend": facility_backend,
    }


def load_config(path: Path, budget: int | None) -> TACConfig:
    values = load_json(path)
    config = TACConfig(**values)
    if budget is not None:
        config = replace(config, budget=int(budget))
    if config.budget <= 0:
        raise ValueError(f"budget must be positive, got {config.budget}")
    if str(config.facility_backend).lower() not in {"auto", "numpy", "submodlib"}:
        raise ValueError(
            f"facility_backend must be one of auto, numpy, submodlib; got {config.facility_backend!r}"
        )
    config = replace(config, facility_backend=str(config.facility_backend).lower())
    return config


def load_arrays(
    cache_root: Path,
    target: str,
    config: TACConfig,
    target_support: dict[str, Any],
) -> dict[str, Any]:
    cache_dir = cache_root / target
    source_ids = read_ids(cache_dir / "source_ids.txt")
    embeddings = l2_normalize(np.load(cache_dir / "source_embeddings_l2.npy").astype(np.float32))
    density_raw = compute_source_density(embeddings)
    clusters = compute_source_clusters(embeddings)

    reliability = compute_reliability(cache_dir, len(source_ids))
    influence = normalize(np.load(cache_dir / "influence.npy").astype(np.float32))
    late = normalize(np.load(cache_dir / "late_influence.npy").astype(np.float32))
    density = normalize(density_raw)
    target_signals = compute_target_conditioned_signals(source_ids, embeddings, target_support, config)
    query_similarity = target_signals["Q"]
    match = target_signals["M"]
    facility = target_signals["F"]
    align = target_signals["A"]
    facility_backend = target_signals["facility_backend"]

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
        "facility_backend": facility_backend,
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
        "facility_backend": arrays["facility_backend"],
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
    target_support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arrays = load_arrays(cache_root, target, config=config, target_support=target_support)
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
        "# TAC Source-Target Curation Summary",
        "",
        "TAC is Target-Anchored Coverage for Source-Target Curation.",
        "",
        "The full pipeline selects a target support set and then applies one continuous target-conditioned source-curation objective for every target.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(asdict(config), indent=2, sort_keys=True),
        "```",
        "",
        "## Selected Subsets",
        "",
        "| Target | Selected | Facility backend | Sparse act. | Coverage act. | Consensus act. | R-top | H-top | F-top | Multi |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        diag = row["diagnostics"]
        acts = diag["gate_activations"]
        counts = diag["selected_counts"]
        lines.append(
            f"| {row['target']} | {row['selected_total']} | "
            f"{diag['facility_backend']} | "
            f"{acts['sparse']:.3f} | {acts['coverage']:.3f} | {acts['consensus']:.3f} | "
            f"{counts['reliability_top']} | {counts['influence_top']} | "
            f"{counts['facility_top']} | {counts['multi_signal']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- The source-curation stage recomputes density, clustering, distribution-match, query-similarity, and target-coverage signals from source embeddings and the generated TAC target support.",
        "- If `facility_backend` is `submodlib` or available through `auto`, target coverage uses FLMI greedy selection on freshly computed K/Q similarities; otherwise `auto` uses the deterministic NumPy facility-style fallback.",
        "- The source-curation stage does not read legacy reference splits, validation Dice, or test Dice.",
        "- Output source subsets are written under `data/source_splits/<target>/tac_<budget>/`.",
        "",
    ])
    summary_path = results_root / "TAC_SOURCE_TARGET_CURATION_SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (results_root / "TAC_SELECTION_SUMMARY.md").write_text(
        "This file is retained as a compatibility pointer.\n"
        f"See `{summary_path.name}` for the TAC source-target curation summary.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("all", *DEFAULT_TARGETS), default="all")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument(
        "--facility-backend",
        choices=("auto", "numpy", "submodlib"),
        default=None,
        help="Override the TAC coverage backend. 'submodlib' requires the optional submodlib package.",
    )
    parser.add_argument("--selector-config", type=Path, default=ROOT / "configs/tac_selector.json")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/source_splits")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--split-source-root", type=Path, default=ROOT / "external/selection_inputs")
    parser.add_argument("--target-split-root", type=Path, default=ROOT / "data/target_splits")
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
    split_source_root = arg_path(args.split_source_root)
    target_split_root = arg_path(args.target_split_root)
    if selector_config is None or cache_root is None or data_root is None:
        raise ValueError("selector_config, cache_root, and data_root are required")
    config = load_config(selector_config, args.budget)
    if args.facility_backend is not None:
        config = replace(config, facility_backend=args.facility_backend)
    results_root = arg_path(args.results_root, ROOT / "results/source_selection" / f"b{config.budget}")
    if results_root is None:
        raise ValueError("results_root is required")
    target_names = sorted(DEFAULT_TARGETS) if args.target == "all" else [args.target]
    missing = [target for target in target_names if not (cache_root / target).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing cache directories: {', '.join(missing)}")
    targets = target_names
    target_supports = {
        target: load_target_support(split_source_root, target_split_root, target)
        for target in targets
    }
    rows = [
        summarize_target(target, config, cache_root, data_root, results_root, target_supports[target])
        for target in targets
    ]
    write_summary(rows, results_root, config)
    for row in rows:
        print(f"{row['target']}: selected {row['selected_total']} -> {row['source_split']}")
    print(results_root / "TAC_SOURCE_TARGET_CURATION_SUMMARY.md")


if __name__ == "__main__":
    main()
