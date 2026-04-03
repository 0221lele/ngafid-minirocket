from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.data import NGAFIDBinaryDataset
from src.train import run_fold
from src.utils import set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-fold MiniRocket CV for NGAFID before/after maintenance detection.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the dataset folder (e.g. data/2days/ with flight_data.pkl). Override with an absolute path on your machine.",
    )
    parser.add_argument("--dataset-name", default="2days")
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.5,
        help="Fraction of train/test rows per fold. Default 0.5 matches the documented 2days benchmark (not full 11446 flights).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/benchmark_2days", help="Where to write models, history CSVs, and summary JSON.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = NGAFIDBinaryDataset(data_dir=args.data_dir, dataset_name=args.dataset_name, max_length=args.max_length)

    fold_results: list[dict] = []
    for fold in range(args.start_fold, args.start_fold + args.num_folds):
        print(f"\n--- Running Fold {fold} ---")
        fold_metrics = run_fold(
            dataset=dataset,
            fold=fold,
            sample_ratio=args.sample_ratio,
            epochs=args.epochs,
            output_dir=output_dir,
        )
        row = {"fold": fold, **fold_metrics}
        fold_results.append(row)
        write_json(output_dir / "fold_metrics.json", {"fold_results": fold_results})
        print(
            f"Fold {fold} — acc: {fold_metrics['accuracy']:.4f}, "
            f"bal_acc: {fold_metrics['balanced_accuracy']:.4f}, "
            f"f1: {fold_metrics['f1']:.4f}, roc_auc: {fold_metrics['roc_auc']:.4f}"
        )

    def collect(key: str) -> np.ndarray:
        return np.array([item[key] for item in fold_results], dtype=np.float64)

    accuracies = collect("accuracy")
    bal_accs = collect("balanced_accuracy")
    f1s = collect("f1")
    roc_aucs = collect("roc_auc")
    roc_aucs_clean = roc_aucs[~np.isnan(roc_aucs)]

    summary = {
        "dataset_name": args.dataset_name,
        "data_dir": args.data_dir,
        "sample_ratio": args.sample_ratio,
        "epochs": args.epochs,
        "seed": args.seed,
        "fold_accuracies": [round(value, 6) for value in accuracies.tolist()],
        "mean_accuracy": float(accuracies.mean()),
        "std_accuracy": float(accuracies.std()),
        "fold_balanced_accuracy": [round(float(x), 6) for x in bal_accs],
        "mean_balanced_accuracy": float(bal_accs.mean()),
        "std_balanced_accuracy": float(bal_accs.std()),
        "fold_f1": [round(float(x), 6) for x in f1s],
        "mean_f1": float(f1s.mean()),
        "std_f1": float(f1s.std()),
        "fold_roc_auc": [round(float(x), 6) if not np.isnan(x) else None for x in roc_aucs],
        "mean_roc_auc": float(roc_aucs_clean.mean()) if roc_aucs_clean.size else None,
        "std_roc_auc": float(roc_aucs_clean.std()) if roc_aucs_clean.size > 1 else None,
    }
    write_json(output_dir / "summary.json", summary)

    print("\n=== MiniRocket Summary ===")
    print("Fold accuracies:", [round(value, 4) for value in accuracies.tolist()])
    print(f"Mean accuracy: {accuracies.mean():.4f} ± {accuracies.std():.4f}")
    print(f"Mean balanced accuracy: {bal_accs.mean():.4f} ± {bal_accs.std():.4f}")
    print(f"Mean F1: {f1s.mean():.4f} ± {f1s.std():.4f}")
    if roc_aucs_clean.size:
        print(f"Mean ROC-AUC: {roc_aucs_clean.mean():.4f} ± {roc_aucs_clean.std():.4f}")


if __name__ == "__main__":
    main()
