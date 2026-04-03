"""Generate paper-style figures from a finished CV run (history CSV + summary JSON)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curve and 5-fold metric bars from results/.")
    parser.add_argument("--results-dir", default="results/benchmark_2days", help="Directory with summary.json and minirocket_fold*_history.csv")
    parser.add_argument("--fold-history", type=int, default=0, help="Which fold's minirocket_fold{N}_history.csv to plot for the training curve")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="PNG output directory (default: <results-dir>/figures)",
    )
    args = parser.parse_args()

    res = Path(args.results_dir)
    if not res.is_dir():
        raise SystemExit(f"Results directory not found: {res.resolve()}")

    out = Path(args.output_dir) if args.output_dir else res / "figures"
    out.mkdir(parents=True, exist_ok=True)

    hist_path = res / f"minirocket_fold{args.fold_history}_history.csv"
    if not hist_path.exists():
        raise SystemExit(f"History CSV not found: {hist_path}")

    df = pd.read_csv(hist_path)
    epochs = df["epoch"].astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(epochs, df["train_loss"], label="train loss", color="#2563eb")
    ax.plot(epochs, df["valid_loss"], label="valid loss", color="#dc2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"MiniRocket training curve (fold {args.fold_history}, 2days benchmark)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_png = out / f"training_curve_fold{args.fold_history}.png"
    fig.savefig(curve_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {curve_png.resolve()}")

    summary_path = res / "summary.json"
    if not summary_path.exists():
        print("No summary.json; skipping bar chart.")
        return

    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)

    folds = list(range(len(summary["fold_accuracies"])))
    x = range(len(folds))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    acc = summary["fold_accuracies"]
    bal = summary.get("fold_balanced_accuracy", acc)
    f1 = summary.get("fold_f1", acc)
    roc = summary.get("fold_roc_auc", acc)

    ax.bar([i - 1.5 * width for i in x], acc, width, label="Accuracy", color="#1d4ed8")
    ax.bar([i - 0.5 * width for i in x], bal, width, label="Balanced Acc", color="#059669")
    ax.bar([i + 0.5 * width for i in x], f1, width, label="F1", color="#d97706")
    ax.bar([i + 1.5 * width for i in x], roc, width, label="ROC-AUC", color="#7c3aed")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylim(0.45, max(max(acc), max(bal), max(f1), max(roc)) * 1.08)
    ax.set_ylabel("Score")
    ax.set_title("5-fold validation metrics (2days, sample_ratio=0.5, MiniRocket)")
    ax.legend(loc="upper center", ncol=4, fontsize=8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="random (acc)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    bar_png = out / "fold_metrics_bars.png"
    fig.savefig(bar_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {bar_png.resolve()}")


if __name__ == "__main__":
    main()
