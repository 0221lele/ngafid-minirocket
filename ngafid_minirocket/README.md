# NGAFID MiniRocket Reproduction

This project reproduces the before-vs-after maintenance flight classification task from the paper *A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID* using `MiniRocket`.

## What This Repository Delivers

- One-command execution for cross-validation
- 5-fold cross-validation support
- Fold-level accuracy reporting plus `mean +- std`
- Resume support through `--start-fold`
- AI collaboration log that documents tool assistance and human decisions

## Data layout

`--data-dir` should be the folder that **contains** the `2days` directory (default dataset name), **or** that folder itself when files already sit in `data_dir`. The loader checks, in order:

1. `{data_dir}/2days/{flight_data.pkl,...}` — flat layout  
2. `{data_dir}/2days/2days/{...}` — nested layout (common in archives)  
3. `{data_dir}/{...}` — files directly under `data_dir`

**Example (recommended for GitHub + bundled data):** put data under `data/2days/` or `data/2days/2days/`, then run with `--data-dir data`. You can still use an absolute path instead of `data`.

## Setup (clean environment)

From the repository root (no `PYTHONPATH` needed — scripts add the project root automatically):

```bash
pip install -r requirements.txt
```

On Windows, `reproduce.bat` runs the same install step then `scripts\run_cv.py`; append any extra flags (for example `--data-dir D:\ngafid\2days`).

## Full 5-fold benchmark (reported numbers)

```bash
python scripts/run_cv.py --dataset-name 2days --sample-ratio 0.5 --epochs 50 --num-folds 5
```

If your data is not under `data/`, set `--data-dir` to the parent of the `2days` folder.

## Figures (after a finished run)

Writes PNGs under **`results/benchmark_2days/figures/`** (same folder as that run’s `summary.json`; or `<--results-dir>/figures` if you override):

```bash
python scripts/generate_figures.py
```

## Faster smoke test

```bash
python scripts/run_cv.py --sample-ratio 0.1 --epochs 3 --num-folds 1
```

## Resume from an interrupted run

```bash
python scripts/run_cv.py --sample-ratio 0.5 --epochs 50 --start-fold 4 --num-folds 1
```

## Outputs

Artifacts are written to a run-specific directory, e.g. `results/benchmark_2days/`:

- `fold_metrics.json`, `summary.json`
- `minirocket_fold*_history.csv`
- `models/`
- `results/benchmark_2days/figures/training_curve_fold0.png`, `fold_metrics_bars.png` (after `generate_figures.py`)

## Current reproduction result (`2days`, sample_ratio=0.5, seed 42)

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 0.5894 ± 0.0148 |
| Balanced accuracy | 0.5888 ± 0.0151 |
| F1 | 0.5697 ± 0.0166 |
| ROC-AUC | 0.6246 ± 0.0137 |

Per-fold accuracy: 0.5773, 0.5997, 0.5892, 0.5699, 0.6110.

## Clone from GitHub (when `data/` is in the repo)

From the repository root, after `pip install -r requirements.txt`:

```bash
python scripts/run_cv.py --data-dir data --dataset-name 2days --sample-ratio 0.5 --epochs 50 --num-folds 5
```

Large binaries (`flight_data.pkl`) may require [Git LFS](https://git-lfs.com/) if you publish them on GitHub; otherwise zip the `data/` folder as a release attachment and unzip into the clone.

## Notes

- This repository focuses on `MiniRocket` only, as requested.
- The default run uses the `2days` benchmark subset because it is feasible to reproduce locally.
- The main improvements over the original notebook workflow are code modularization, resumable execution, and persistent result files.
