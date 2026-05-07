# TAC: Target-Anchored Coverage for Source–Target Curation in Cross-Center Medical Image Segmentation

This folder contains the reviewer-facing code for TAC, a joint target
acquisition and source curation method for the four BraTS target centers. It
excludes batch-submission files and reference reconstruction scripts. The
default workflow regenerates target splits, source splits, and training configs
from reproducible inputs; generated artifacts are not part of the clean source
package.

TAC is not only a source selector. It first selects a small target support set
from the unlabeled target pool with uncertainty/local-inconsistency weighted
KMeans, then uses that target anchor to curate a compact labeled source subset
through one deterministic source–target curation rule.

## Layout

- `cache/<target>/`: generated TAC selection cache, produced from raw slices
  and checkpoints by `scripts/prepare_tac_inputs.py`.
- `configs/tac_selector.json`: selector constants used for every target.
- `configs/targets.json`: package-relative TAC target support/eval locations.
- `scripts/prepare_tac_inputs.py`: extracts selection-time embeddings, target
  scores, P/G/M reliability primitives, and gradient influence from raw data.
- `scripts/generate_experiment_splits.py`: regenerates TAC and baseline target/source splits.
- `scripts/select_tac_sources.py`: deterministic TAC source-curation stage.
- `scripts/build_tac_training_configs.py`: EfficientVit config generator.
- `scripts/build_experiment_configs.py`: builds the full non-reference method
  grid, including TAC.
- `scripts/train_configs.py`: runs generated configs with the local Python env.
- `scripts/validate_experiment_package.py`: validates full-grid configs.

## External Dependencies

Requires Python 3.10+, NumPy, PyYAML, and scikit-learn for split/config
generation and validation. Training additionally requires PyTorch and the
external EfficientVit codebase. Raw TAC input preparation additionally requires
PyTorch and the dependencies needed by the EfficientVit codebase. TAC can use
`submodlib` as an optional exact FLMI backend for target-anchored facility
coverage; when `facility_backend` is `auto` and `submodlib` is unavailable, the
selector falls back to a deterministic NumPy facility-style greedy backend.
Optional FLMI dependencies are listed in `requirements-optional.txt`.

Before running, either create these package-local symlinks:

```text
external/EfficientVit
external/BraTS2021_preprocessed
external/selection_inputs
```

or export equivalent paths:

```bash
export EFFICIENTVIT_ROOT=/path/to/EfficientVit
export BRATS_DATA_ROOT=/path/to/BraTS2021/preprocessed
export SPLIT_SOURCE_ROOT=/path/to/selection_inputs
```

`SPLIT_SOURCE_ROOT` provides target candidate lists, validation/test lists, and
target embedding/score arrays. These files can be generated from raw BraTS
slices and EfficientViT checkpoints with `scripts/prepare_tac_inputs.py`.
The package writes generated splits/configs under this folder only, never under
`SPLIT_SOURCE_ROOT`.

Expected `selection_inputs` layout:

```text
selection_inputs/
  split_TCGA_LGG_active/
    target_subject_ids.txt
    target_subject_vecs.npy
    target_uncertainty.npy
    target_subject_fg_score.npy
    lada_li_scores.npy
    aada_importance_scores.npy
    target_consistency.npy
    val_subjects.txt
    test_subjects.txt
  split_C4_active/
  split_C5_active/
  split_TCGA_GBM_active/
```

Each target directory follows the same file layout.

## Raw TAC Input Preparation

The clean reviewer path does not require prebuilt TAC reliability or influence
scores. Generate them from raw BraTS slices, official split files, and warmup
checkpoints before building experiment splits:

```bash
python scripts/prepare_tac_inputs.py \
  --efficientvit-root "$EFFICIENTVIT_ROOT" \
  --data-root "$BRATS_DATA_ROOT" \
  --split-root "$EFFICIENTVIT_ROOT/data" \
  --warmup-checkpoint /path/to/warmup/epoch_020.pt \
  --gradient-checkpoints /path/to/warmup/epoch_010.pt /path/to/warmup/epoch_015.pt /path/to/warmup/epoch_020.pt \
  --targets all
```

This writes:

```text
external/selection_inputs/split_<target>_active/
cache/<target>/
```

The script computes:

- subject embeddings from the frozen EfficientViT warmup model;
- target acquisition scores from foreground-aware entropy, local inconsistency,
  and consistency views;
- `aada_importance_scores.npy` from the AADA DANN rule: train a domain
  discriminator with gradient reversal, then rank targets by
  `((1 - p_source) / p_source) * entropy`;
- `reliability_P.npy` from target-support morphology prototypes;
- `reliability_M.npy` from target-support distribution match;
- `reliability_G.npy`, `influence.npy`, and `late_influence.npy` from
  final-head gradient agreement at the supplied checkpoints;
- `reliability_spec.json` from selection-time adaptive P/G/M reliability
  weights.

No downstream Dice, validation/test metrics, reference subset membership, or
legacy subset membership is read by this preparation script.

## One-Command Preparation

From this folder:

```bash
# Run scripts/prepare_tac_inputs.py first if cache/<target>/ and
# external/selection_inputs/ have not already been generated from raw data.
python scripts/generate_experiment_splits.py --budgets 50 100 150 200
for b in 50 100 150 200; do
  python scripts/build_tac_training_configs.py --budget "$b"
  python scripts/build_experiment_configs.py --budget "$b"
  python scripts/validate_tac_package.py --budget "$b" --require-generated
done
python scripts/validate_experiment_package.py --budgets 50 100 150 200
```

This regenerates all source and target split files and all EfficientVit training
configs from the package inputs. It does not read downstream Dice or reference
subset membership.

Outputs are written to:

```text
data/source_splits/
data/target_splits/
configs/generated/
configs/experiments/
results/
```

## TAC-Only Generation

If you only need TAC source–target curation artifacts:

```bash
python scripts/generate_experiment_splits.py --budgets 150
python scripts/build_tac_training_configs.py --budget 150
python scripts/validate_tac_package.py --budget 150 --require-generated
```

TAC target and source outputs are written to:

```text
data/target_splits/<target>/tac_10/train_subjects.txt
data/source_splits/<target>/tac_150/train_subjects.txt
results/source_selection/b150/TAC_SOURCE_TARGET_CURATION_SUMMARY.md
```

TAC recomputes the reliability utility from generated P/G/M primitive
components and selection-time weights in `cache/<target>/reliability_spec.json`;
it does not load a precomputed reliability scalar. TAC also recomputes source
density, source clusters, source--target query similarity, distribution match,
target-anchored facility coverage, and facility alignment from source embeddings
and the generated `tac_10` target support split. Facility coverage supports
three backends through `configs/tac_selector.json` or
`scripts/select_tac_sources.py --facility-backend`:

- `submodlib`: exact FLMI greedy selection using source-source similarities
  `K` and source-target-support similarities `Q` computed from current TAC
  embeddings.
- `numpy`: package-local deterministic facility-style greedy target coverage
  with redundancy control.
- `auto`: use `submodlib` when installed, otherwise use `numpy`.

To require exact FLMI and fail loudly if the optional dependency is missing:

```bash
python scripts/select_tac_sources.py --budget 150 --facility-backend submodlib
```

The TAC target split is generated by weighted KMeans on target embeddings with
sample weights
`normalize(target_uncertainty * target_subject_fg_score) + 0.50 * normalize(lada_li_scores) + 0.15 * normalize(target_consistency)`.

Baseline target splits are regenerated from their method-specific rules:

- `clue`: entropy-weighted KMeans on target embeddings, selecting the nearest
  target subject to each centroid.
- `lada`: top `(1 + M) * B_t` local-inconsistency candidates followed by
  KMeans diversity selection, with `M=10`.
- `aada`: top subjects by the AADA DANN importance score generated during raw
  input preparation; if that score file is absent in a manually supplied
  `SPLIT_SOURCE_ROOT`, the script computes the same diversity-times-uncertainty
  score with a deterministic source-vs-target embedding domain classifier.

## Local Training

Training is deliberately script-only. Dry-run the commands that would be
executed after building full-grid configs:

```bash
python scripts/train_configs.py --budget 150 --dry-run
python scripts/train_configs.py --budget 150 --target TCGA_LGG --source-method tac --target-method tac --dry-run
```

Remove `--dry-run` to launch local training with the current Python environment.
For TAC-only configs, pass explicit config paths with `--config`; `--budget` is
not required when explicit configs are provided.

## Validate

Validate TAC-only configs:

```bash
python scripts/validate_tac_package.py --budget 150 --require-generated
```

Validate the full non-reference method grid:

```bash
python scripts/validate_experiment_package.py --budgets 50 100 150 200
```

## Full Non-Reference Method Grid

Build configs for all runnable non-reference methods at one budget:

```bash
python scripts/generate_experiment_splits.py --budgets 150
python scripts/build_experiment_configs.py --budget 150
```

Supported source strategies are `target_only`, `full_source`, `random1`,
`random2`, `random3`, `coreset`, `orient`, and `tac`. Supported target
strategies are `none`, `random1`, `random2`, `random3`, `coreset`, `lada`,
`aada`, `ada_clue`, and `tac`.

All budgets include `aada` and `ada_clue` target strategies. Every config
written by `build_experiment_configs.py` is runnable; there are no manifest
entries for unavailable or disallowed combinations.

Full-grid manifests and configs are written to:

```text
results/experiment_manifest_b<budget>.json
configs/experiments/b<budget>/
```

## Generated Artifact Policy

The clean source package should contain code and configs only. TAC cache files,
generated split files, training configs, results, logs, and outputs are ignored
by `.gitignore` and can be regenerated with:

```bash
rm -rf cache/* data/source_splits configs/generated configs/experiments results logs outputs
find data/target_splits -mindepth 1 ! -name README.md -exec rm -rf {} +
python scripts/prepare_tac_inputs.py \
  --warmup-checkpoint /path/to/warmup/epoch_020.pt \
  --gradient-checkpoints /path/to/warmup/epoch_010.pt /path/to/warmup/epoch_015.pt /path/to/warmup/epoch_020.pt \
  --targets all
python scripts/generate_experiment_splits.py --budgets 50 100 150 200
for b in 50 100 150 200; do
  python scripts/build_tac_training_configs.py --budget "$b"
  python scripts/build_experiment_configs.py --budget "$b"
done
```

## Target Support Splits

| Target | TAC target support split |
|---|---|
| `TCGA_LGG` | `data/target_splits/TCGA_LGG/tac_10` |
| `C4` | `data/target_splits/C4/tac_10` |
| `C5` | `data/target_splits/C5/tac_10` |
| `TCGA_GBM` | `data/target_splits/TCGA_GBM/tac_10` |
