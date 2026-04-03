from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from compress_pickle import load
from tqdm.auto import tqdm


@dataclass
class DatasetPaths:
    root: Path
    flight_data: Path
    flight_header: Path
    stats: Path


class NGAFIDBinaryDataset:
    channels = 23

    def __init__(self, data_dir: str | Path, dataset_name: str = "2days", max_length: int = 4096):
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.paths = self._resolve_paths(Path(data_dir), dataset_name)

        self.flight_header_df = pd.read_csv(self.paths.flight_header, index_col="Master Index")
        self.flight_data_array = load(self.paths.flight_data)
        self.flight_stats_df = pd.read_csv(self.paths.stats)
        self.maxs = self.flight_stats_df.iloc[0, 1:24].to_numpy(dtype=np.float32)
        self.mins = self.flight_stats_df.iloc[1, 1:24].to_numpy(dtype=np.float32)
        self._data_dict: list[dict] | None = None

    @staticmethod
    def _resolve_paths(data_dir: Path, dataset_name: str) -> DatasetPaths:
        candidates = [
            data_dir / dataset_name,
            data_dir / dataset_name / dataset_name,
            data_dir,
        ]
        for root in candidates:
            flight_data = root / "flight_data.pkl"
            flight_header = root / "flight_header.csv"
            stats = root / "stats.csv"
            if flight_data.exists() and flight_header.exists() and stats.exists():
                return DatasetPaths(root=root, flight_data=flight_data, flight_header=flight_header, stats=stats)
        raise FileNotFoundError(
            f"Could not find dataset files under '{data_dir}'. Expected flight_data.pkl, flight_header.csv, and stats.csv."
        )

    @property
    def data_dict(self) -> list[dict]:
        if self._data_dict is None:
            self._data_dict = self.construct_data_dictionary()
        return self._data_dict

    def construct_data_dictionary(self) -> list[dict]:
        data_dict: list[dict] = []
        for index, row in tqdm(self.flight_header_df.iterrows(), total=len(self.flight_header_df), desc="Building samples"):
            arr = np.zeros((self.max_length, self.channels), dtype=np.float16)
            to_pad = self.flight_data_array[index][-self.max_length :, :]
            arr[: to_pad.shape[0], :] += to_pad
            data_dict.append(
                {
                    "id": index,
                    "data": arr,
                    "fold": int(row["fold"]),
                    "target_class": int(row["target_class"]),
                    "before_after": int(row["before_after"]),
                }
            )
        return data_dict

    def get_numpy_dataset(self, fold: int = 0, training: bool = False) -> dict[str, list]:
        if training:
            sliced = [example for example in self.data_dict if example["fold"] != fold]
        else:
            sliced = [example for example in self.data_dict if example["fold"] == fold]
        return {key: [item[key] for item in sliced] for key in sliced[0]}
