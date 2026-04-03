from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from fastai.callback.progress import CSVLogger, ShowGraphCallback
from fastai.learner import Learner
from fastai.metrics import accuracy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from tsai.all import (
    MiniRocketFeatures,
    MiniRocketHead,
    TSClassification,
    TSStandardize,
    build_ts_model,
    default_device,
    get_minirocket_features,
    get_ts_dls,
)

from src.data import NGAFIDBinaryDataset


def _minmax_normalize(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> None:
    denom = maxs - mins
    denom = np.where(denom < 1e-8, 1.0, denom)
    x -= mins
    x /= denom


def run_fold(
    dataset: NGAFIDBinaryDataset,
    fold: int,
    sample_ratio: float,
    epochs: int,
    output_dir: str | Path,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    train_dict = dataset.get_numpy_dataset(fold=fold, training=True)
    test_dict = dataset.get_numpy_dataset(fold=fold, training=False)

    train_limit = max(1, int(len(train_dict["data"]) * sample_ratio))
    test_limit = max(1, int(len(test_dict["data"]) * sample_ratio))

    train_x = np.array(train_dict["data"][:train_limit], dtype=np.float32)
    test_x = np.array(test_dict["data"][:test_limit], dtype=np.float32)
    train_y = np.array(train_dict["before_after"][:train_limit], dtype=np.int64)
    test_y = np.array(test_dict["before_after"][:test_limit], dtype=np.int64)

    del train_dict, test_dict
    gc.collect()

    _minmax_normalize(train_x, dataset.mins, dataset.maxs)
    np.nan_to_num(train_x, copy=False)

    _minmax_normalize(test_x, dataset.mins, dataset.maxs)
    np.nan_to_num(test_x, copy=False)

    splits = [list(np.arange(len(train_y))), list(np.arange(len(test_y)) + len(train_y))]
    mrf = MiniRocketFeatures(train_x.shape[1], train_x.shape[2]).to(default_device())
    chunksize = 32
    mrf.fit(train_x, chunksize=chunksize)

    x_combined = np.concatenate([train_x, test_x])
    y_combined = np.concatenate([train_y, test_y])

    del train_x, test_x, train_y, test_y
    gc.collect()

    x_feat = get_minirocket_features(x_combined, mrf, chunksize=chunksize, to_np=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mrf.state_dict(), model_dir / f"MRF_fold{fold}.pt")

    tfms = [None, TSClassification()]
    batch_tfms = TSStandardize(by_sample=True)
    dls = get_ts_dls(x_feat, y_combined, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
    model = build_ts_model(MiniRocketHead, dls=dls)
    learner = Learner(
        dls,
        model,
        metrics=accuracy,
        cbs=[ShowGraphCallback(), CSVLogger(fname=output_dir / f"minirocket_fold{fold}_history.csv")],
    )
    learner.fit_one_cycle(epochs, 2.5e-5)

    probs, targs = learner.get_preds(ds_idx=1)
    y_true = targs.cpu().numpy().astype(np.int64).ravel()
    p = probs.cpu().numpy()
    y_pred = p.argmax(axis=1).astype(np.int64)
    pos_score = p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()
    fold_accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_true, pos_score))
    except ValueError:
        roc_auc = float("nan")

    del dls, model, learner, x_feat, y_combined, mrf, probs, targs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "accuracy": fold_accuracy,
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "roc_auc": roc_auc,
    }
