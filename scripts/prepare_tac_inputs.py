#!/usr/bin/env python3
"""Prepare TAC reviewer inputs from raw BraTS slices and EfficientViT checkpoints.

This script is the raw-code path for quantities that the selector consumes from
`external/selection_inputs/<target>/` and `cache/<target>/`. It extracts
selection-time subject embeddings and target acquisition scores, computes
target-anchored reliability primitives P/G/M, computes head-gradient influence
from warmup checkpoints, and writes the cache files used by
`generate_experiment_splits.py`.

It does not use downstream Dice, validation/test metrics, or reference/legacy
subset membership.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience for minimal environments
    def tqdm(values: Iterable[Any], **_: Any) -> Iterable[Any]:
        return values


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("TCGA_LGG", "C4", "C5", "TCGA_GBM")
TARGET_ACTIVE_DIR = {
    "TCGA_LGG": "split_TCGA_LGG_active",
    "C4": "split_C4_active",
    "C5": "split_C5_active",
    "TCGA_GBM": "split_TCGA_GBM_active",
}


@dataclass(frozen=True)
class ImportedEfficientVit:
    torch: Any
    data_loader_cls: Any
    dataset_cls: Any
    model_cls: Any
    loss_cls: Any
    feature_extractor_cls: Any


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_ids(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    temperature = max(float(temperature), 1e-6)
    logits = values / temperature
    logits = logits - float(logits.max())
    exp_values = np.exp(logits)
    return (exp_values / (exp_values.sum() + 1e-12)).astype(np.float32)


def active_dir(split_root: Path, target: str) -> Path:
    path = split_root / TARGET_ACTIVE_DIR[target]
    if not path.exists():
        raise FileNotFoundError(f"{target}: missing target active split directory {path}")
    return path


def source_split_dir(split_root: Path, target: str) -> Path:
    path = split_root / f"splits_{target}_source"
    if not (path / "train_subjects.txt").exists():
        raise FileNotFoundError(f"{target}: missing source split {path / 'train_subjects.txt'}")
    return path


def has_brats_slice_files(root: Path) -> bool:
    return (
        (root / "imagesTr").is_dir()
        and (root / "labelsTr").is_dir()
        and any((root / "imagesTr").glob("*.npy"))
        and any((root / "labelsTr").glob("*.npy"))
    )


def import_efficientvit(efficientvit_root: Path) -> ImportedEfficientVit:
    efficientvit_root = efficientvit_root.expanduser().resolve()
    if not efficientvit_root.exists():
        raise FileNotFoundError(f"EfficientVit root not found: {efficientvit_root}")
    sys.path.insert(0, str(efficientvit_root))

    import torch
    from torch.utils.data import DataLoader

    from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
    from models.efficientvit_seg.efficientvit_seg import EfficientViT_Seg
    from models.efficientvit_seg.losses import DiceCELoss
    from models.feature_extractor import FeatureExtractor

    return ImportedEfficientVit(
        torch=torch,
        data_loader_cls=DataLoader,
        dataset_cls=BraTSSliceDataset,
        model_cls=EfficientViT_Seg,
        loss_cls=DiceCELoss,
        feature_extractor_cls=FeatureExtractor,
    )


def load_model(modules: ImportedEfficientVit, checkpoint: Path, device: Any) -> Any:
    torch = modules.torch
    model = modules.model_cls(
        backbone="efficientvit_l1",
        in_channels=4,
        num_classes=4,
        pretrained=False,
    ).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, dict):
        model_state = state.get("model_state_dict", state.get("model_state", state))
    else:
        model_state = state
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model


def build_loader(
    modules: ImportedEfficientVit,
    data_root: Path,
    split_txt_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    skip_empty: bool,
    return_meta: bool,
    shuffle: bool = False,
) -> Any:
    dataset = modules.dataset_cls(
        root_dir=str(data_root),
        split="train",
        img_size=int(img_size),
        split_txt_dir=str(split_txt_dir),
        skip_empty=bool(skip_empty),
    )
    dataset.return_meta = bool(return_meta)
    return modules.data_loader_cls(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=False,
    )


def build_training_loader(
    modules: ImportedEfficientVit,
    data_root: Path,
    split_txt_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    skip_empty: bool,
    shuffle: bool,
    drop_last: bool,
) -> Any:
    dataset = modules.dataset_cls(
        root_dir=str(data_root),
        split="train",
        img_size=int(img_size),
        split_txt_dir=str(split_txt_dir),
        skip_empty=bool(skip_empty),
    )
    dataset.return_meta = False
    return modules.data_loader_cls(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=bool(drop_last),
        pin_memory=True,
    )


def extract_backbone_features(model: Any, images: Any) -> Any:
    features = model.model.backbone(images)
    if isinstance(features, dict):
        features = list(features.values())[-1]
    return features


def decode_features(model: Any, features: Any, spatial_shape: tuple[int, int]) -> Any:
    output = model.decoder(features)
    output = model.final_head(output)
    if output.shape[2:] != spatial_shape:
        import torch.nn.functional as F

        output = F.interpolate(output, size=spatial_shape, mode="bilinear", align_corners=False)
    return output


def pixel_entropy(torch: Any, logits: Any) -> Any:
    probabilities = torch.softmax(logits, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1)
    return -(probabilities * log_probabilities).sum(dim=1).mean(dim=(1, 2))


def infinite_iter(loader: Any) -> Iterable[Any]:
    while True:
        for batch in loader:
            yield batch


def build_domain_discriminator(torch: Any, channels: int, hidden_dim: int) -> Any:
    class GradientReversalFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, values: Any, alpha: float) -> Any:
            ctx.alpha = alpha
            return values.view_as(values)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:
            return grad_output.neg() * ctx.alpha, None

    class DomainDiscriminator(torch.nn.Module):
        def __init__(self, in_channels: int, hidden_dim: int) -> None:
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, features: Any, alpha: float = 1.0) -> Any:
            values = GradientReversalFn.apply(features, alpha)
            values = self.pool(values).flatten(1)
            return self.classifier(values)

        def predict_source_prob(self, features: Any) -> Any:
            with torch.no_grad():
                values = self.pool(features).flatten(1)
                return torch.sigmoid(self.classifier(values)).squeeze(1)

    return DomainDiscriminator(int(channels), int(hidden_dim))


def compute_aada_importance_scores(
    modules: ImportedEfficientVit,
    model: Any,
    data_root: Path,
    source_split: Path,
    target_split: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
    source_skip_empty: bool,
    target_skip_empty: bool,
    device: Any,
    epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
    lambda_dann: float,
    hidden_dim: int,
    top_ratio: float,
    entropy_weight: float,
    min_slices: int,
) -> dict[str, Any]:
    torch = modules.torch
    source_loader = build_training_loader(
        modules,
        data_root,
        source_split,
        img_size,
        batch_size,
        num_workers,
        skip_empty=source_skip_empty,
        shuffle=True,
        drop_last=True,
    )
    target_loader = build_training_loader(
        modules,
        data_root,
        target_split,
        img_size,
        batch_size,
        num_workers,
        skip_empty=target_skip_empty,
        shuffle=True,
        drop_last=True,
    )
    probe_images = next(iter(source_loader))[0].to(device)
    with torch.no_grad():
        probe_features = extract_backbone_features(model, probe_images)
    discriminator = build_domain_discriminator(torch, int(probe_features.shape[1]), int(hidden_dim)).to(device)
    seg_criterion = modules.loss_cls()
    domain_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=float(learning_rate),
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1), eta_min=1e-6)
    source_iter = infinite_iter(source_loader)
    target_iter = infinite_iter(target_loader)
    for epoch in range(int(epochs)):
        model.train()
        discriminator.train()
        progress = epoch / max(int(epochs), 1)
        alpha = float(lambda_dann) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
        for _ in tqdm(range(int(steps_per_epoch)), desc=f"AADA DANN {epoch + 1}/{epochs}", ncols=100):
            source_batch = next(source_iter)
            target_batch = next(target_iter)
            source_images = source_batch[0].to(device)
            source_labels = source_batch[1].to(device)
            target_images = target_batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            source_features = extract_backbone_features(model, source_images)
            source_logits = decode_features(model, source_features, source_images.shape[2:])
            seg_loss = seg_criterion(source_logits, source_labels)
            source_domain = discriminator(source_features, alpha)
            target_features = extract_backbone_features(model, target_images)
            target_domain = discriminator(target_features, alpha)
            domain_loss = domain_criterion(source_domain, torch.ones_like(source_domain)) + domain_criterion(target_domain, torch.zeros_like(target_domain))
            loss = seg_loss + domain_loss
            loss.backward()
            optimizer.step()
        scheduler.step()

    scoring_loader = build_loader(
        modules,
        data_root,
        target_split,
        img_size,
        batch_size,
        num_workers,
        skip_empty=target_skip_empty,
        return_meta=True,
    )
    model.eval()
    discriminator.eval()
    slice_names: list[str] = []
    slice_scores: list[float] = []
    slice_diversity: list[float] = []
    slice_uncertainty: list[float] = []
    slice_fg: list[float] = []
    with torch.no_grad():
        for images, _labels, subject_ids, slice_indices in tqdm(scoring_loader, desc="AADA importance", ncols=100):
            images = images.to(device)
            features = extract_backbone_features(model, images)
            source_prob = discriminator.predict_source_prob(features)
            diversity = (1.0 - source_prob) / (source_prob + 1e-8)
            logits = decode_features(model, features, images.shape[2:])
            uncertainty = pixel_entropy(torch, logits)
            probabilities = torch.softmax(logits, dim=1)
            fg_score = probabilities[:, 1:, :, :].sum(dim=1).mean(dim=(1, 2))
            scores = diversity * uncertainty
            for idx, subject_id in enumerate(subject_ids):
                slice_names.append(f"{subject_id}_slice{int(slice_indices[idx].item())}")
                slice_scores.append(float(scores[idx].item()))
                slice_diversity.append(float(diversity[idx].item()))
                slice_uncertainty.append(float(uncertainty[idx].item()))
                slice_fg.append(float(fg_score[idx].item()))
    uncertainty_array = np.asarray(slice_uncertainty, dtype=np.float32)
    fg_array = np.asarray(slice_fg, dtype=np.float32)
    ranking = fg_array + float(entropy_weight) * uncertainty_array
    return aggregate_subject_arrays(
        slice_names=slice_names,
        arrays={
            "score": np.asarray(slice_scores, dtype=np.float32),
            "diversity": np.asarray(slice_diversity, dtype=np.float32),
            "uncertainty": uncertainty_array,
        },
        ranking_score=ranking,
        top_ratio=top_ratio,
        min_slices=min_slices,
    )


def subject_from_slice_name(name: str) -> str:
    return name.split("_slice")[0]


def aggregate_subject_arrays(
    slice_names: list[str],
    arrays: dict[str, np.ndarray],
    ranking_score: np.ndarray,
    top_ratio: float,
    min_slices: int,
) -> dict[str, Any]:
    subject_to_positions: dict[str, list[int]] = defaultdict(list)
    for position, name in enumerate(slice_names):
        subject_to_positions[subject_from_slice_name(name)].append(position)

    subject_ids = sorted(subject_to_positions)
    aggregated: dict[str, list[np.ndarray | float]] = {name: [] for name in arrays}
    subject_topk: list[int] = []
    for subject_id in subject_ids:
        positions = np.asarray(subject_to_positions[subject_id], dtype=np.int64)
        score = np.asarray(ranking_score[positions], dtype=np.float32)
        k = max(int(np.ceil(float(top_ratio) * len(positions))), int(min_slices))
        k = min(k, len(positions))
        selected_local = np.argsort(-score, kind="mergesort")[:k]
        selected_positions = positions[selected_local]
        subject_topk.append(int(k))
        for name, values in arrays.items():
            selected_values = np.asarray(values)[selected_positions]
            aggregated[name].append(selected_values.mean(axis=0))

    return {
        "subject_ids": subject_ids,
        "subject_topk": np.asarray(subject_topk, dtype=np.int64),
        "aggregated": {
            name: np.stack(values, axis=0).astype(np.float32)
            for name, values in aggregated.items()
        },
    }


def compute_local_inconsistency(features: np.ndarray, probabilities: np.ndarray, neighbors: int) -> np.ndarray:
    features = l2_normalize(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    n = features.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    similarity = (features @ features.T).astype(np.float32)
    k = min(int(neighbors), n - 1)
    output = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        row = similarity[idx].copy()
        row[idx] = -np.inf
        top = np.argsort(-row, kind="mergesort")[:k]
        weights = np.maximum(row[top], 0.0)
        weights = weights / (weights.sum() + 1e-8)
        output[idx] = 1.0 - float(np.sum(weights * (probabilities[top] @ probabilities[idx])))
    smoothed = output.copy()
    for idx in range(n):
        row = similarity[idx].copy()
        row[idx] = -np.inf
        top = np.argsort(-row, kind="mergesort")[:k]
        weights = np.maximum(row[top], 0.0)
        weights = weights / (weights.sum() + 1e-8)
        smoothed[idx] += float(np.sum(weights * output[top]))
    return smoothed.astype(np.float32)


def extract_subject_state(
    modules: ImportedEfficientVit,
    model: Any,
    loader: Any,
    device: Any,
    consistency_views: int,
    consistency_jitter: float,
    consistency_noise: float,
    top_ratio: float,
    entropy_weight: float,
    min_slices: int,
) -> dict[str, Any]:
    torch = modules.torch
    feature_model = modules.feature_extractor_cls(model).to(device)
    feature_model.eval()
    model.eval()

    slice_embeddings: list[np.ndarray] = []
    slice_entropy: list[float] = []
    slice_fg: list[float] = []
    slice_mean_probs: list[np.ndarray] = []
    slice_consistency: list[float] = []
    slice_names: list[str] = []

    def consistency_view(images: Any, view_idx: int) -> Any:
        if view_idx == 0:
            return images * (1.0 - float(consistency_jitter))
        if view_idx == 1:
            return images * (1.0 + float(consistency_jitter))
        if view_idx == 2:
            return images + float(consistency_noise)
        if view_idx == 3:
            return images - float(consistency_noise)
        scale = 1.0 + (0.5 * float(consistency_jitter) if view_idx % 2 == 0 else -0.5 * float(consistency_jitter))
        return images * scale

    with torch.no_grad():
        for images, _labels, subject_ids, slice_indices in tqdm(loader, desc="extract selection state", ncols=100):
            images = images.to(device)
            features = feature_model(images)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(images.shape[0], -1)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1).mean(dim=(1, 2))
            fg_score = probabilities[:, 1:, :, :].sum(dim=1).mean(dim=(1, 2))
            mean_probs = probabilities.mean(dim=(2, 3))

            base_pred = probabilities.argmax(dim=1)
            consistency = torch.zeros(images.shape[0], device=device, dtype=torch.float32)
            for view_idx in range(int(consistency_views)):
                aug_probs = torch.softmax(model(consistency_view(images, view_idx)), dim=1)
                consistency += (aug_probs.argmax(dim=1) == base_pred).float().mean(dim=(1, 2))
            consistency /= max(int(consistency_views), 1)

            slice_embeddings.append(features.cpu().numpy().astype(np.float32))
            slice_entropy.extend(entropy.cpu().numpy().astype(np.float32).tolist())
            slice_fg.extend(fg_score.cpu().numpy().astype(np.float32).tolist())
            slice_mean_probs.append(mean_probs.cpu().numpy().astype(np.float32))
            slice_consistency.extend(consistency.cpu().numpy().astype(np.float32).tolist())
            for subject_id, slice_idx in zip(subject_ids, slice_indices):
                slice_names.append(f"{subject_id}_slice{int(slice_idx.item())}")

    embeddings = np.concatenate(slice_embeddings, axis=0).astype(np.float32)
    entropy_array = np.asarray(slice_entropy, dtype=np.float32)
    fg_array = np.asarray(slice_fg, dtype=np.float32)
    mean_probs_array = np.concatenate(slice_mean_probs, axis=0).astype(np.float32)
    consistency_array = np.asarray(slice_consistency, dtype=np.float32)
    ranking = fg_array + float(entropy_weight) * entropy_array
    subject_state = aggregate_subject_arrays(
        slice_names=slice_names,
        arrays={
            "embeddings": embeddings,
            "entropy": entropy_array,
            "fg": fg_array,
            "mean_probs": mean_probs_array,
            "consistency": consistency_array,
        },
        ranking_score=ranking,
        top_ratio=top_ratio,
        min_slices=min_slices,
    )
    subject_vectors = l2_normalize(subject_state["aggregated"]["embeddings"])
    subject_li = compute_local_inconsistency(
        subject_vectors,
        subject_state["aggregated"]["mean_probs"],
        neighbors=10,
    )
    return {
        "subject_ids": subject_state["subject_ids"],
        "subject_embeddings_l2": subject_vectors,
        "subject_entropy": subject_state["aggregated"]["entropy"].reshape(-1).astype(np.float32),
        "subject_fg": subject_state["aggregated"]["fg"].reshape(-1).astype(np.float32),
        "subject_consistency": subject_state["aggregated"]["consistency"].reshape(-1).astype(np.float32),
        "subject_li": subject_li,
        "subject_topk": subject_state["subject_topk"],
    }


def write_selection_inputs(target_dir: Path, target_state: dict[str, Any], active_split_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    write_ids(target_dir / "target_subject_ids.txt", target_state["subject_ids"])
    np.save(target_dir / "target_subject_vecs.npy", target_state["subject_embeddings_l2"])
    np.save(target_dir / "target_uncertainty.npy", target_state["subject_entropy"])
    np.save(target_dir / "target_subject_fg_score.npy", target_state["subject_fg"])
    np.save(target_dir / "lada_li_scores.npy", target_state["subject_li"])
    np.save(target_dir / "target_consistency.npy", target_state["subject_consistency"])
    for name in ("val_subjects.txt", "test_subjects.txt"):
        source = active_split_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        write_ids(target_dir / name, read_ids(source))


def reorder_subject_state(state: dict[str, Any], desired_ids: list[str], label: str) -> dict[str, Any]:
    positions = {subject_id: idx for idx, subject_id in enumerate(state["subject_ids"])}
    missing = [subject_id for subject_id in desired_ids if subject_id not in positions]
    if missing:
        raise ValueError(f"{label}: missing {len(missing)} requested subjects; first few: {missing[:5]}")
    order = np.asarray([positions[subject_id] for subject_id in desired_ids], dtype=np.int64)
    output: dict[str, Any] = {}
    for key, value in state.items():
        if key == "subject_ids":
            output[key] = list(desired_ids)
        elif isinstance(value, np.ndarray) and value.shape[:1] == (len(order),):
            output[key] = value[order]
        else:
            output[key] = value
    return output


def target_acquisition_weights(target_state: dict[str, Any]) -> np.ndarray:
    return (
        normalize(target_state["subject_entropy"] * target_state["subject_fg"])
        + 0.50 * normalize(target_state["subject_li"])
        + 0.15 * normalize(target_state["subject_consistency"])
    ).astype(np.float32)


def weighted_kmeans_support(
    ids: list[str],
    vectors: np.ndarray,
    weights: np.ndarray,
    budget: int,
    seed: int,
) -> list[str]:
    from sklearn.cluster import KMeans

    vectors = l2_normalize(vectors)
    weights = np.maximum(np.asarray(weights, dtype=np.float32).reshape(-1), 1e-6)
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
        priority = 0.7 * (1.0 - normalize(distances)) + 0.3 * weight_term
        for idx_value in np.argsort(-priority, kind="mergesort"):
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


def build_subject_index(dataset: Any) -> dict[str, list[int]]:
    subject_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        subject_to_indices[sample[2]].append(idx)
    return subject_to_indices


def write_temp_split(subject_ids: list[str]) -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix="tac_subject_split_")
    write_ids(Path(temp_dir.name) / "train_subjects.txt", subject_ids)
    return temp_dir


def compute_subject_head_gradients(
    modules: ImportedEfficientVit,
    model: Any,
    dataset: Any,
    subject_ids: list[str],
    subject_to_indices: dict[str, list[int]],
    device: Any,
    batch_size: int,
) -> np.ndarray:
    torch = modules.torch
    criterion = modules.loss_cls()
    params = [parameter for parameter in model.final_head.parameters() if parameter.requires_grad]
    gradients: list[np.ndarray] = []
    for subject_id in tqdm(subject_ids, desc="subject head gradients", ncols=100):
        indices = subject_to_indices.get(subject_id, [])
        if not indices:
            raise KeyError(f"subject {subject_id} is absent from gradient dataset")
        model.train()
        model.zero_grad(set_to_none=True)
        total_slices = 0
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start : start + int(batch_size)]
            images = []
            labels = []
            for dataset_idx in batch_indices:
                image, label = dataset[dataset_idx]
                images.append(image)
                labels.append(label)
            image_tensor = torch.stack(images, dim=0).to(device)
            label_tensor = torch.stack(labels, dim=0).to(device)
            logits = model(image_tensor)
            loss = criterion(logits, label_tensor) * image_tensor.shape[0]
            loss.backward()
            total_slices += int(image_tensor.shape[0])
        pieces = [parameter.grad.detach().reshape(-1) for parameter in params if parameter.grad is not None]
        if not pieces:
            raise RuntimeError("no final-head gradients were produced")
        gradient = torch.cat(pieces).detach().cpu().numpy().astype(np.float32)
        gradient = gradient / max(total_slices, 1)
        gradient = gradient / (np.linalg.norm(gradient) + 1e-8)
        gradients.append(gradient.astype(np.float32))
    return np.stack(gradients, axis=0).astype(np.float32)


def build_gradient_scores(
    modules: ImportedEfficientVit,
    checkpoint_paths: list[Path],
    data_root: Path,
    source_split: Path,
    target_support_ids: list[str],
    img_size: int,
    batch_size: int,
    num_workers: int,
    source_skip_empty: bool,
    target_skip_empty: bool,
    device: Any,
    output_gradient_dir: Path,
) -> dict[str, np.ndarray | list[str]]:
    if not checkpoint_paths:
        raise ValueError("at least one gradient checkpoint is required")
    source_loader = build_loader(
        modules,
        data_root,
        source_split,
        img_size,
        batch_size,
        num_workers,
        skip_empty=source_skip_empty,
        return_meta=False,
    )
    source_dataset = source_loader.dataset
    source_ids = read_ids(source_split / "train_subjects.txt")
    source_index = build_subject_index(source_dataset)

    with write_temp_split(target_support_ids) as temp_split_name:
        target_loader = build_loader(
            modules,
            data_root,
            Path(temp_split_name),
            img_size,
            batch_size,
            num_workers,
            skip_empty=target_skip_empty,
            return_meta=False,
        )
        target_dataset = target_loader.dataset
        target_index = build_subject_index(target_dataset)

        per_checkpoint_scores: list[np.ndarray] = []
        checkpoint_names: list[str] = []
        for checkpoint in checkpoint_paths:
            checkpoint = checkpoint.expanduser().resolve()
            checkpoint_names.append(checkpoint.name)
            model = load_model(modules, checkpoint, device)
            checkpoint_dir = output_gradient_dir / checkpoint.stem
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            source_grad_path = checkpoint_dir / "source_subject_head_grads.npy"
            target_grad_path = checkpoint_dir / "target_subject_head_grads.npy"
            source_ids_path = checkpoint_dir / "source_subject_ids.txt"
            target_ids_path = checkpoint_dir / "target_subject_ids.txt"
            if source_grad_path.exists() and source_ids_path.exists() and read_ids(source_ids_path) == source_ids:
                source_gradients = np.load(source_grad_path).astype(np.float32)
            else:
                source_gradients = compute_subject_head_gradients(
                    modules,
                    model,
                    source_dataset,
                    source_ids,
                    source_index,
                    device,
                    batch_size,
                )
                np.save(source_grad_path, source_gradients)
                write_ids(source_ids_path, source_ids)
            if target_grad_path.exists() and target_ids_path.exists() and read_ids(target_ids_path) == target_support_ids:
                target_gradients = np.load(target_grad_path).astype(np.float32)
            else:
                target_gradients = compute_subject_head_gradients(
                    modules,
                    model,
                    target_dataset,
                    target_support_ids,
                    target_index,
                    device,
                    batch_size,
                )
                np.save(target_grad_path, target_gradients)
                write_ids(target_ids_path, target_support_ids)
            source_normalized = l2_normalize(source_gradients)
            query_gradient = l2_normalize(target_gradients.mean(axis=0, keepdims=True))[0]
            per_checkpoint_scores.append(np.maximum(source_normalized @ query_gradient, 0.0).astype(np.float32))

    score_stack = np.stack(per_checkpoint_scores, axis=0).astype(np.float32)
    influence = normalize(score_stack.mean(axis=0))
    late_count = min(3, score_stack.shape[0])
    late_influence = normalize(score_stack[-late_count:].mean(axis=0))
    return {
        "source_ids": source_ids,
        "checkpoint_names": checkpoint_names,
        "score_stack": score_stack,
        "influence": influence,
        "late_influence": late_influence,
    }


def build_label_index(label_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(label_dir.glob("*.npy")):
        subject_id = path.stem.split("_slice")[0]
        index[subject_id].append(path)
    return index


def subject_morphology(label_index: dict[str, list[Path]], subject_ids: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for subject_id in tqdm(subject_ids, desc="subject morphology", ncols=100):
        paths = label_index.get(subject_id, [])
        if not paths:
            raise FileNotFoundError(f"missing label slices for subject {subject_id}")
        total = 0
        wt = 0
        tc = 0
        et = 0
        for path in paths:
            label = np.load(path)
            total += int(label.size)
            wt += int((label > 0).sum())
            tc += int(np.logical_or(label == 1, label == 3).sum())
            et += int((label == 3).sum())
        stats[subject_id] = {
            "wt_ratio": float(wt / max(total, 1)),
            "tc_ratio": float(tc / max(total, 1)),
            "et_ratio": float(et / max(total, 1)),
            "has_et": float(et > 0),
        }
    return stats


def morphology_matrix(stats: dict[str, dict[str, float]], subject_ids: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [
                stats[subject_id]["wt_ratio"],
                stats[subject_id]["tc_ratio"],
                stats[subject_id]["et_ratio"],
                stats[subject_id]["has_et"],
            ]
            for subject_id in subject_ids
        ],
        dtype=np.float32,
    )


def morphology_score(features: np.ndarray, feature_min: np.ndarray, feature_max: np.ndarray) -> np.ndarray:
    normalized = np.clip((features - feature_min) / (feature_max - feature_min + 1e-8), 0.0, 1.0)
    return (
        normalized[:, 0]
        + normalized[:, 1]
        + 1.5 * normalized[:, 2]
        + 0.5 * normalized[:, 3]
    ).astype(np.float32)


def prototype_reliability_score(
    source_ids: list[str],
    source_vectors: np.ndarray,
    target_ids: list[str],
    target_vectors: np.ndarray,
    label_dir: Path,
) -> np.ndarray:
    label_index = build_label_index(label_dir)
    source_stats = subject_morphology(label_index, source_ids)
    target_stats = subject_morphology(label_index, target_ids)
    target_features = morphology_matrix(target_stats, target_ids)
    source_features = morphology_matrix(source_stats, source_ids)
    feature_min = target_features.min(axis=0)
    feature_max = target_features.max(axis=0)
    target_scores = morphology_score(target_features, feature_min, feature_max)
    source_scores = morphology_score(source_features, feature_min, feature_max)
    thresholds = np.quantile(target_scores, [1.0 / 3.0, 2.0 / 3.0]).astype(np.float32)
    target_labels = np.digitize(target_scores, thresholds, right=False).astype(np.int64)
    source_labels = np.digitize(source_scores, thresholds, right=False).astype(np.int64)

    target_vectors = l2_normalize(target_vectors)
    source_vectors = l2_normalize(source_vectors)
    prototype_rows: dict[int, list[np.ndarray]] = defaultdict(list)
    for label, vector in zip(target_labels.tolist(), target_vectors):
        prototype_rows[int(label)].append(vector)
    prototypes = {
        label: l2_normalize(np.stack(rows, axis=0).mean(axis=0, keepdims=True))[0]
        for label, rows in prototype_rows.items()
    }
    fallback = l2_normalize(target_vectors.mean(axis=0, keepdims=True))[0]
    classes = sorted(prototypes)
    prototype_matrix = np.stack([prototypes[label] for label in classes], axis=0) if classes else fallback.reshape(1, -1)
    class_to_col = {label: idx for idx, label in enumerate(classes)}
    similarity = np.maximum(source_vectors @ prototype_matrix.T, 0.0).astype(np.float32)
    values = np.zeros(len(source_ids), dtype=np.float32)
    for idx, label in enumerate(source_labels.tolist()):
        if int(label) in class_to_col:
            pos_col = class_to_col[int(label)]
            positive = float(similarity[idx, pos_col])
            if similarity.shape[1] > 1:
                mask = np.ones(similarity.shape[1], dtype=bool)
                mask[pos_col] = False
                negative = float(similarity[idx, mask].max())
            else:
                negative = 0.0
            values[idx] = positive - negative
        else:
            values[idx] = float(np.maximum(source_vectors[idx] @ fallback, 0.0))
    return normalize(values)


def distribution_match_score(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
    target_weights: np.ndarray,
    topk: int,
) -> np.ndarray:
    source_vectors = l2_normalize(source_vectors)
    target_vectors = l2_normalize(target_vectors)
    target_weights = np.maximum(np.asarray(target_weights, dtype=np.float32).reshape(-1), 1e-6)
    similarity = np.maximum(source_vectors @ target_vectors.T, 0.0).astype(np.float32)
    k = min(int(topk), similarity.shape[1])
    top = np.argsort(-similarity, axis=1, kind="mergesort")[:, :k]
    row_weights = target_weights[top]
    row_weights = row_weights / (row_weights.sum(axis=1, keepdims=True) + 1e-12)
    raw = (similarity[np.arange(similarity.shape[0])[:, None], top] * row_weights).sum(axis=1)
    return normalize(raw)


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size or a.size == 0:
        return 0.0
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def top_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(int(k), len(a), len(b))
    if k <= 0:
        return 0.0
    left = set(np.argsort(-a, kind="mergesort")[:k].tolist())
    right = set(np.argsort(-b, kind="mergesort")[:k].tolist())
    return float(len(left & right) / k)


def stability_against_variants(reference: np.ndarray, variants: list[np.ndarray], topk: int) -> float:
    if not variants:
        return 1.0
    scores = []
    for variant in variants:
        scores.append(0.5 * (safe_corrcoef(reference, variant) + 1.0))
        scores.append(top_overlap(reference, variant, topk))
    return float(np.mean(scores))


def adaptive_reliability_weights(
    p_score: np.ndarray,
    g_stack: np.ndarray,
    m_score: np.ndarray,
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
    target_weights: np.ndarray,
    temperature: float,
    topk: int,
) -> dict[str, float]:
    m_variants: list[np.ndarray] = []
    if target_vectors.shape[0] > 2:
        rng = np.random.default_rng(2025)
        keep = max(2, int(round(0.8 * target_vectors.shape[0])))
        for _ in range(8):
            subset = np.sort(rng.choice(target_vectors.shape[0], size=keep, replace=False))
            m_variants.append(distribution_match_score(source_vectors, target_vectors[subset], target_weights[subset], topk=8))
    p_rel = 0.5 + 0.5 * float(np.std(p_score))
    g_variants = [normalize(g_stack[idx]) for idx in range(g_stack.shape[0])] if g_stack.ndim == 2 else []
    g_base = normalize(g_stack.mean(axis=0)) if g_stack.ndim == 2 else normalize(g_stack)
    reliabilities = np.asarray(
        [
            p_rel,
            stability_against_variants(g_base, g_variants, topk=topk),
            stability_against_variants(m_score, m_variants, topk=topk),
        ],
        dtype=np.float32,
    )
    weights = softmax(reliabilities, temperature=temperature)
    return {"P": float(weights[0]), "G": float(weights[1]), "M": float(weights[2])}


def prepare_one_target(args: argparse.Namespace, modules: ImportedEfficientVit, device: Any, target: str) -> None:
    data_root = args.data_root.expanduser().resolve()
    if not has_brats_slice_files(data_root):
        raise FileNotFoundError(f"BraTS slice root must contain imagesTr/ and labelsTr/: {data_root}")

    split_root = args.split_root.expanduser().resolve()
    active_split = active_dir(split_root, target)
    source_split = source_split_dir(split_root, target)
    target_selection_dir = args.selection_input_root.expanduser().resolve() / TARGET_ACTIVE_DIR[target]
    target_cache_dir = args.cache_root.expanduser().resolve() / target
    target_cache_dir.mkdir(parents=True, exist_ok=True)

    torch = modules.torch
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    warmup_model = load_model(modules, args.warmup_checkpoint.expanduser().resolve(), device)
    source_loader = build_loader(
        modules,
        data_root,
        source_split,
        args.img_size,
        args.batch_size,
        args.num_workers,
        skip_empty=args.source_skip_empty,
        return_meta=True,
    )
    target_loader = build_loader(
        modules,
        data_root,
        active_split,
        args.img_size,
        args.batch_size,
        args.num_workers,
        skip_empty=args.target_skip_empty,
        return_meta=True,
    )

    print(f"[{target}] extracting source selection state")
    source_state = extract_subject_state(
        modules,
        warmup_model,
        source_loader,
        device,
        args.consistency_views,
        args.consistency_jitter,
        args.consistency_noise,
        args.subject_top_ratio,
        args.subject_entropy_weight,
        args.subject_min_slices,
    )
    source_state = reorder_subject_state(
        source_state,
        read_ids(source_split / "train_subjects.txt"),
        label=f"{target} source state",
    )
    print(f"[{target}] extracting target selection state")
    target_state = extract_subject_state(
        modules,
        warmup_model,
        target_loader,
        device,
        args.consistency_views,
        args.consistency_jitter,
        args.consistency_noise,
        args.subject_top_ratio,
        args.subject_entropy_weight,
        args.subject_min_slices,
    )

    write_selection_inputs(target_selection_dir, target_state, active_split)
    write_ids(target_cache_dir / "source_ids.txt", source_state["subject_ids"])
    np.save(target_cache_dir / "source_embeddings_l2.npy", source_state["subject_embeddings_l2"])

    if not args.skip_aada_score:
        print(f"[{target}] computing AADA DANN target scores")
        aada_model = load_model(modules, args.warmup_checkpoint.expanduser().resolve(), device)
        aada_state = compute_aada_importance_scores(
            modules,
            aada_model,
            data_root,
            source_split,
            active_split,
            args.img_size,
            args.batch_size,
            args.num_workers,
            args.source_skip_empty,
            args.target_skip_empty,
            device,
            args.aada_epochs,
            args.aada_steps_per_epoch,
            args.aada_lr,
            args.aada_lambda_dann,
            args.aada_hidden_dim,
            args.subject_top_ratio,
            args.subject_entropy_weight,
            args.subject_min_slices,
        )
        aada_positions = {subject_id: idx for idx, subject_id in enumerate(aada_state["subject_ids"])}
        missing_aada = [subject_id for subject_id in target_state["subject_ids"] if subject_id not in aada_positions]
        if missing_aada:
            raise ValueError(f"{target}: AADA scores missing target subjects: {missing_aada[:5]}")
        aada_order = np.asarray([aada_positions[subject_id] for subject_id in target_state["subject_ids"]], dtype=np.int64)
        np.save(target_selection_dir / "aada_importance_scores.npy", aada_state["aggregated"]["score"][aada_order].astype(np.float32))
        np.save(target_selection_dir / "aada_diversity.npy", aada_state["aggregated"]["diversity"][aada_order].astype(np.float32))

    target_weights_all = target_acquisition_weights(target_state)
    target_support_ids = weighted_kmeans_support(
        target_state["subject_ids"],
        target_state["subject_embeddings_l2"],
        target_weights_all,
        budget=args.target_budget,
        seed=args.seed,
    )
    target_pos = {subject_id: idx for idx, subject_id in enumerate(target_state["subject_ids"])}
    target_indices = np.asarray([target_pos[subject_id] for subject_id in target_support_ids], dtype=np.int64)
    target_support_vectors = target_state["subject_embeddings_l2"][target_indices]
    target_support_weights = target_weights_all[target_indices]

    print(f"[{target}] computing gradient influence")
    gradient_bundle = build_gradient_scores(
        modules,
        [path.expanduser().resolve() for path in args.gradient_checkpoints],
        data_root,
        source_split,
        target_support_ids,
        args.img_size,
        args.gradient_batch_size,
        args.num_workers,
        args.source_skip_empty,
        args.target_skip_empty,
        device,
        target_cache_dir / "gradient_primitives",
    )
    if gradient_bundle["source_ids"] != source_state["subject_ids"]:
        raise ValueError(f"{target}: source IDs differ between embedding and gradient extraction")

    print(f"[{target}] computing reliability primitives")
    p_score = prototype_reliability_score(
        source_state["subject_ids"],
        source_state["subject_embeddings_l2"],
        target_support_ids,
        target_support_vectors,
        data_root / "labelsTr",
    )
    m_score = distribution_match_score(
        source_state["subject_embeddings_l2"],
        target_support_vectors,
        target_support_weights,
        topk=args.match_topk,
    )
    g_score = np.asarray(gradient_bundle["late_influence"], dtype=np.float32)
    weights = adaptive_reliability_weights(
        p_score,
        np.asarray(gradient_bundle["score_stack"], dtype=np.float32),
        m_score,
        source_state["subject_embeddings_l2"],
        target_support_vectors,
        target_support_weights,
        temperature=args.reliability_temperature,
        topk=min(args.reliability_topk, len(source_state["subject_ids"])),
    )

    np.save(target_cache_dir / "reliability_P.npy", p_score)
    np.save(target_cache_dir / "reliability_G.npy", g_score)
    np.save(target_cache_dir / "reliability_M.npy", m_score)
    np.save(target_cache_dir / "influence.npy", np.asarray(gradient_bundle["influence"], dtype=np.float32))
    np.save(target_cache_dir / "late_influence.npy", np.asarray(gradient_bundle["late_influence"], dtype=np.float32))
    (target_cache_dir / "reliability_spec.json").write_text(
        json.dumps(
            {
                "mode": "adaptive_target_anchor_reliability",
                "rule": "R = minmax(lambda_P * P + lambda_G * G + lambda_M * M)",
                "component_files": {
                    "P": "reliability_P.npy",
                    "G": "reliability_G.npy",
                    "M": "reliability_M.npy",
                },
                "weights": weights,
                "normalization": "minmax after weighted sum",
                "target_support_rule": "weighted k-means over target acquisition weights",
                "target_support_ids": target_support_ids,
                "gradient_checkpoints": gradient_bundle["checkpoint_names"],
                "note": "Generated from raw BraTS labels/slices and EfficientViT checkpoints; no downstream Dice or subset membership is used.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (target_cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "target": target,
                "source_subjects": len(source_state["subject_ids"]),
                "target_candidates": len(target_state["subject_ids"]),
                "target_budget": int(args.target_budget),
                "warmup_checkpoint": str(args.warmup_checkpoint),
                "gradient_checkpoints": [str(path) for path in args.gradient_checkpoints],
                "aada_score_file": None if args.skip_aada_score else str(target_selection_dir / "aada_importance_scores.npy"),
                "selection_input_dir": str(target_selection_dir),
                "cache_dir": str(target_cache_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[{target}] wrote TAC inputs to {target_selection_dir} and {target_cache_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=["all"], help="Targets to prepare, or 'all'.")
    parser.add_argument(
        "--efficientvit-root",
        type=Path,
        default=Path(os.environ.get("EFFICIENTVIT_ROOT", ROOT / "external/EfficientVit")),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("BRATS_DATA_ROOT", ROOT / "external/BraTS2021_preprocessed")),
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path(os.environ.get("BRATS_SPLIT_ROOT", ROOT / "external/EfficientVit/data")),
        help="Directory containing split_<target>_active and splits_<target>_source.",
    )
    parser.add_argument("--selection-input-root", type=Path, default=ROOT / "external/selection_inputs")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--warmup-checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--target-budget", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--source-skip-empty", action="store_true", default=True)
    parser.add_argument("--target-skip-empty", action="store_true", default=False)
    parser.add_argument("--subject-top-ratio", type=float, default=0.30)
    parser.add_argument("--subject-entropy-weight", type=float, default=0.25)
    parser.add_argument("--subject-min-slices", type=int, default=1)
    parser.add_argument("--consistency-views", type=int, default=3)
    parser.add_argument("--consistency-jitter", type=float, default=0.15)
    parser.add_argument("--consistency-noise", type=float, default=0.03)
    parser.add_argument("--skip-aada-score", action="store_true", help="Skip DANN-based AADA target-score generation.")
    parser.add_argument("--aada-epochs", type=int, default=10)
    parser.add_argument("--aada-steps-per-epoch", type=int, default=500)
    parser.add_argument("--aada-lr", type=float, default=1e-4)
    parser.add_argument("--aada-lambda-dann", type=float, default=0.1)
    parser.add_argument("--aada-hidden-dim", type=int, default=256)
    parser.add_argument("--match-topk", type=int, default=8)
    parser.add_argument("--reliability-temperature", type=float, default=1.0)
    parser.add_argument("--reliability-topk", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = list(TARGETS if args.targets == ["all"] else args.targets)
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"unknown targets: {', '.join(unknown)}")
    modules = import_efficientvit(args.efficientvit_root)
    torch = modules.torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested --device cuda, but CUDA is not available")
    device = torch.device(args.device)
    for target in targets:
        prepare_one_target(args, modules, device, target)


if __name__ == "__main__":
    main()
