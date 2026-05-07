# TAC: Target-Anchored Coverage for Source–Target Curation in Cross-Center Medical Image Segmentation

This folder contains the reviewer-facing code for TAC, a joint target
acquisition and source curation method for the four BraTS target centers. It
excludes batch-submission files and reference reconstruction scripts. The
default workflow regenerates target splits, source splits, and training configs
from reproducible inputs; generated artifacts are not part of the clean source
package.

TAC is not only a source selector. It first selects a small target support set
from the unlabeled target pool, then uses that target anchor to curate a compact
labeled source subset through one deterministic source–target curation rule.

## Layout

- `cache/<target>/`: selection-time primitive arrays used by TAC source curation.
- `configs/tac_selector.json`: selector constants used for every target.
- `configs/targets.json`: package-relative TAC target support/eval locations.
- `scripts/generate_experiment_splits.py`: regenerates TAC and baseline target/source splits.
- `scripts/select_tac_sources.py`: deterministic TAC source-curation stage.
- `scripts/build_tac_training_configs.py`: EfficientVit config generator.
- `scripts/build_experiment_configs.py`: builds the full non-reference method
  grid, including TAC.
- `scripts/train_configs.py`: runs generated configs with the local Python env.
- `scripts/validate_experiment_package.py`: validates full-grid configs.

## External Dependencies

Requires Python 3.10+, NumPy, and PyYAML for split/config generation and
validation. Training additionally requires PyTorch and the external EfficientVit
codebase.

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
target embedding/score arrays. The package writes generated splits/configs under
this folder only, never under `SPLIT_SOURCE_ROOT`.

Expected `selection_inputs` layout:

```text
selection_inputs/
  split_TCGA_LGG_active/
    target_subject_ids.txt
    target_subject_vecs.npy
    target_uncertainty.npy
    target_subject_fg_score.npy
    lada_li_scores.npy
    val_subjects.txt
    test_subjects.txt
  split_C4_active/
  split_C5_active/
  split_TCGA_GBM_active/
```

Each target directory follows the same file layout.

## One-Command Preparation

From this folder:

```bash
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

TAC recomputes the reliability utility from cached P/G/M-style primitive
components and fixed selection-time weights in `cache/<target>/reliability_spec.json`;
it does not load a precomputed reliability scalar.

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

Budget `50` includes `aada` and `ada_clue` target strategies. Budgets `100`,
`150`, and `200` include all non-`aada`/`ada_clue` target strategies.
Every config written by `build_experiment_configs.py` is runnable; there are no
manifest entries for unavailable or disallowed combinations.

Full-grid manifests and configs are written to:

```text
results/experiment_manifest_b<budget>.json
configs/experiments/b<budget>/
```

## Generated Artifact Policy

The clean source package should contain code, configs, and cache primitives only.
Generated split files, training configs, results, logs, and outputs are ignored
by `.gitignore` and can be regenerated with:

```bash
rm -rf data/source_splits configs/generated configs/experiments results logs outputs
find data/target_splits -mindepth 1 ! -name README.md -exec rm -rf {} +
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
