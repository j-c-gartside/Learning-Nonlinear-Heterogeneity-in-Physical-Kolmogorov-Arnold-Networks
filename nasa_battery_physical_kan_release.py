#!/usr/bin/env python
"""
NASA battery physical KAN: fixed release training script
========================================================

This is the fixed training recipe used for the NASA battery result in our
physical KAN work. It is the version I would actually use to reproduce the
result, rather than the much larger Optuna notebooks used to find it.

A few important things up front:

1. The network is [9, 12, 1], with exactly 12 physical devices per edge.
2. This is the original circuit model: S_min, S_max, two constant control
   voltages V, output gain G_o and output bias B_o. There is no decorrelation
   matrix, no input-dependent y=mx+c control, and no inter-layer tanh.
3. The digital twin is frozen. Only the physical KAN controls are trained.
4. There are no range penalties, projection steps, clamps or weight decay.
5. There is no Optuna in this script. The hyperparameters below were locked
   before the release confirmation runs.

The slightly odd-looking ten-way warm start is intentional. Ten independent
initialisations are trained for ten epochs, the best is selected using the
validation set, and that exact model and Adam state continue to epoch 800.
Nothing is restarted at the promotion point.

Dataset
-------

We cannot redistribute the NASA battery data with this repository. Download
the CSV version here:

    https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset

After extracting it, pass the directory containing ``metadata.csv`` and the
``data`` folder. In the Kaggle archive this is normally ``cleaned_dataset``.

The original NASA data record is here:

    https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

Digital twin
------------

Place ``B1_MLP_3_50_50_1.pt`` beside this script, or pass its location using
``--twin``. The release repository should include this frozen digital twin.

Typical use
-----------

Reproduce the originally selected run:

    python nasa_battery_physical_kan_release.py \
        --data-root /path/to/cleaned_dataset \
        --twin B1_MLP_3_50_50_1.pt

Run the original seed plus the three independent confirmation seeds:

    python nasa_battery_physical_kan_release.py \
        --data-root /path/to/cleaned_dataset \
        --twin B1_MLP_3_50_50_1.pt \
        --seeds 24460 24461 24462 24463

The expected scale is roughly 2.8--3.0e-3 validation nMSE for the fixed
recipe. The original seed gave 2.8278e-3 validation nMSE and 2.9896e-3 test
nMSE in our environment. Exact floating-point agreement is not promised
across PyTorch/CUDA versions, but it should be very close.

Dependencies
------------

    pip install torch numpy pandas scikit-learn matplotlib joblib

Author: Jack Gartside
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# Keep matplotlib's cache local. This avoids permission nonsense on clusters
# and does not affect any numerical result.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("./.matplotlib-cache").resolve())
)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# Fixed scientific recipe
# =============================================================================

DATASET_DOWNLOAD_URL = (
    "https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset"
)
NASA_SOURCE_URL = (
    "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets"
)

ARCHITECTURE = [9, 12, 1]
DEVICES_PER_EDGE = 12
MAX_EPOCHS = 800
WARMUP_INITIALISATIONS = 10
WARMUP_EPOCHS = 10
EVALUATE_EVERY = 10
CHECKPOINT_SOUP_TOP_K = 5

# Early stopping is deliberately conservative. It cannot act before epoch 500
# and requires 100 epochs without a meaningful (>4e-5) training-MSE gain.
EARLY_STOPPING_MIN_EPOCH = 500
EARLY_STOPPING_PATIENCE = 100
EARLY_STOPPING_MIN_DELTA = 0.04e-3

# This is only a memory-management microbatch inside the frozen twin. It does
# not change the optimisation batch size or the circuit.
TWIN_MICROBATCH = 131072

# The nMSE plotted in the paper is MSE after min-max scaling the target to
# [-1, 1]. I retain the name ``paper_nMSE`` below to remove any ambiguity.
FEATURE_COLUMNS = [
    "Time",
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
    "mean_Voltage_measured",
    "mean_Voltage_load",
    "mean_Temperature_measured",
]
TARGET_COLUMN = "seconds_remaining"
SNAPSHOT_COLUMNS = FEATURE_COLUMNS[:6]
HISTORY_SOURCE_COLUMNS = [
    "Voltage_measured", "Voltage_load", "Temperature_measured"
]

# Confirmed release recipe. These values were re-tested on three fresh matched
# split/initialisation seeds after the optimisation was complete. Their fresh
# validation mean was 2.856e-3 with a standard deviation of 0.151e-3.
RELEASE_HP = {
    "range_init": "archive_powerlaw",
    "init_dv_min": 1.2,
    "init_dv_max": 1.8,
    "allow_inverted": True,
    "invert_probability": 0.5,
    "span_jitter": 0.1,
    "pl_alpha": 0.25,
    "pl_beta": 1.75,
    "control_init": "archive_powerlaw",
    "control_init_scale": 1.0,
    "gain_init": "zero",
    "bias_init": "zero",
    "output_gain_scale": 50.0,
    "optimizer": "adam",
    "lr_gobo": 1.660037773e-4,
    "lr_v": 1.660037773e-4,
    "lr_s": 8.300188865e-5,
    "beta1": 0.9166804489,
    "beta2": 0.9019533563,
    "adam_eps": 1e-8,
    "batch_size": 128,
    "grad_clip": 10.0,
    "scheduler": "cosine",
    "lr_min_frac": 0.08575523888,
}


@dataclass
class ReleaseConfig:
    """Run-level settings, kept separate from the locked hyperparameters."""

    data_root: Path
    twin_path: Path
    output_dir: Path
    seeds: list[int]
    sampling_seed: int = 0
    samples_per_cycle: int = 50
    second_half_only: bool = True
    sampling_mode: str = "archive_gap"
    min_gap_fraction: float = 0.05
    history_includes_current_row: bool = False
    use_amp: bool = True
    deterministic_algorithms: bool = False
    device_request: str = "auto"

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.twin_path = Path(self.twin_path)
        self.output_dir = Path(self.output_dir)
        self.seeds = [int(seed) for seed in self.seeds]


@dataclass
class ArchiveSplit:
    seed: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_validation: torch.Tensor
    y_validation: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_scaler: Any
    y_scaler: Any
    train_pool_index: np.ndarray
    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray


@dataclass
class TrainingRuntime:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: Any
    current_epoch: int
    best_validation: float
    best_epoch: int
    best_state: dict[str, torch.Tensor]
    top_checkpoints: list[tuple[float, int, dict[str, torch.Tensor]]]
    train_reference: float
    last_train_improvement_epoch: int
    init_index: int
    init_seed: int
    history: list[dict[str, Any]]
    early_stopped: bool = False
    soup_size: int = 1


DEVICE = torch.device("cpu")
USE_AMP = False
DETERMINISTIC_ALGORITHMS = False


# =============================================================================
# Small utilities
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the fixed NASA battery physical-KAN result."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./cleaned_dataset"),
        help=(
            "Directory containing metadata.csv and the data folder. "
            "The script also checks a nested cleaned_dataset directory."
        ),
    )
    parser.add_argument(
        "--twin",
        type=Path,
        default=Path("./B1_MLP_3_50_50_1.pt"),
        help="Path to the frozen B1 3-50-50-1 digital twin.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./nasa_battery_physical_kan_release_outputs"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[24460],
        help=(
            "Training/split seeds. Use 24460 for the original run, or "
            "24460 24461 24462 24463 for the complete confirmation set."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help=(
            "Ask PyTorch for deterministic algorithms. This can be slower and "
            "is not required for the published recipe."
        ),
    )
    return parser.parse_args()


def choose_device(request: str) -> torch.device:
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(
    seed: int,
    deterministic_algorithms: Optional[bool] = None,
) -> None:
    if deterministic_algorithms is None:
        deterministic_algorithms = DETERMINISTIC_ALGORITHMS
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        chunk = handle.read(block_size)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(block_size)
    return digest.hexdigest()


def autocast_context(enabled: bool):
    # torch.autocast(device_type=...) is not available in the older PyTorch
    # environment used for part of this work.
    if not enabled:
        return nullcontext()
    return torch.cuda.amp.autocast()


def make_grad_scaler(enabled: bool):
    # Keep the older API for compatibility with the original environment.
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(DEVICE),
        "device_name": (
            torch.cuda.get_device_name(DEVICE)
            if DEVICE.type == "cuda" else platform.processor()
        ),
    }


# =============================================================================
# Dataset preparation
# =============================================================================

def locate_dataset_root(requested: Path) -> Path:
    """Accept either cleaned_dataset itself or the directory above it."""
    requested = Path(requested).expanduser()
    candidates = [
        requested,
        requested / "cleaned_dataset",
        requested / "archive" / "cleaned_dataset",
    ]
    for candidate in candidates:
        if (candidate / "metadata.csv").is_file():
            return candidate.resolve()
    checked = "\n".join("  - {}".format(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find metadata.csv. Checked:\n{}\n\n"
        "Download and extract the CSV dataset from:\n{}\n"
        "Then pass the cleaned_dataset directory with --data-root.".format(
            checked, DATASET_DOWNLOAD_URL
        )
    )


def choose_column(columns: Iterable[str], candidates: Iterable[str]):
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if str(candidate).lower() in lookup:
            return lookup[str(candidate).lower()]
    return None


def infer_battery_id(row: pd.Series, filename: str) -> str:
    explicit = choose_column(
        row.index,
        ["battery_id", "battery", "cell_id", "cell", "device_id"],
    )
    if explicit is not None and pd.notna(row[explicit]):
        return str(row[explicit])
    stem = Path(filename).stem
    patterns = [
        r"(?i)(B\d{4})",
        r"(?i)(battery[_-]?\d+)",
        r"(?i)(cell[_-]?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return match.group(1).lower()
    return re.split(r"[_-]", stem)[0].lower()


def resolve_cycle_path(root: Path, raw_name: Any) -> Path:
    raw_path = Path(str(raw_name))
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    candidates.extend([
        root / raw_path,
        root / "data" / raw_path,
        root / "data" / raw_path.name,
        root / raw_path.name,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # This is a fallback for archives unpacked with an extra folder level. It
    # should not be hit for the normal Kaggle layout.
    matches = list(root.rglob(raw_path.name))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(
        "Could not resolve cycle file {!r} beneath {}".format(raw_name, root)
    )


def sample_cycle_indices(
    n_rows: int,
    n_samples: int,
    second_half_only: bool,
    rng: np.random.RandomState,
    mode: str,
    min_gap_fraction: float,
) -> np.ndarray:
    """Reproduce the archived random-pick/minimum-gap sampling rule."""
    lower = n_rows // 2 if second_half_only else 0
    eligible = np.arange(lower, n_rows, dtype=np.int64)
    if len(eligible) <= int(n_samples):
        return eligible
    if mode == "uniform_without_replacement":
        return np.sort(rng.choice(eligible, size=int(n_samples), replace=False))
    if mode != "archive_gap":
        raise ValueError("Unknown sampling mode: {}".format(mode))

    upper = n_rows - 1
    minimum_gap = max(1, int(float(min_gap_fraction) * n_rows))
    candidates = np.sort(
        rng.randint(
            lower,
            upper + 1,
            size=max(int(n_samples) * 5, int(n_samples)),
        )
    )
    picks = []
    for candidate in candidates:
        candidate = int(candidate)
        if all(abs(candidate - previous) >= minimum_gap for previous in picks):
            picks.append(candidate)
        if len(picks) >= int(n_samples):
            break
    if len(picks) < int(n_samples):
        for candidate in np.linspace(lower, upper, int(n_samples), dtype=int):
            if int(candidate) not in picks:
                picks.append(int(candidate))
    return np.asarray(
        sorted(set(picks))[: int(n_samples)], dtype=np.int64
    )


def build_regression_table(
    data_root: Path,
    sampling_seed: int,
    samples_per_cycle: int,
    second_half_only: bool,
    sampling_mode: str,
    min_gap_fraction: float,
    history_includes_current_row: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the nine-input remaining-discharge-time regression table.

    Historical means use rows 0..r-1 by default. They never use future values.
    The final timestamp is used only to construct the supervised target.
    """
    metadata_path = data_root / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    filename_column = choose_column(
        metadata.columns, ["filename", "file_name", "file", "path"]
    )
    type_column = choose_column(
        metadata.columns, ["type", "cycle_type", "operation"]
    )
    if filename_column is None or type_column is None:
        raise ValueError(
            "metadata.csv must contain filename and type columns"
        )

    discharge = metadata[
        metadata[type_column].astype(str).str.lower().eq("discharge")
    ].copy()
    if discharge.empty:
        raise RuntimeError("No discharge cycles were found in metadata.csv")

    print(
        "Building regression table from {:,} discharge cycles...".format(
            len(discharge)
        )
    )
    rng = np.random.RandomState(int(sampling_seed))
    rows = []
    audit = []
    for cycle_number, (_, metadata_row) in enumerate(
        discharge.iterrows(), start=0
    ):
        raw_name = str(metadata_row[filename_column])
        battery_id = infer_battery_id(metadata_row, raw_name)
        try:
            cycle_path = resolve_cycle_path(data_root, raw_name)
            frame = pd.read_csv(cycle_path)
            missing = [
                column for column in SNAPSHOT_COLUMNS
                if column not in frame.columns
            ]
            if missing:
                raise ValueError("missing columns {}".format(missing))

            values = frame[SNAPSHOT_COLUMNS].apply(
                pd.to_numeric, errors="coerce"
            )
            values = values.loc[values.notna().all(axis=1)].reset_index(drop=True)
            if len(values) < 2:
                raise ValueError("fewer than two valid measurement rows")

            sampled_indices = sample_cycle_indices(
                n_rows=len(values),
                n_samples=samples_per_cycle,
                second_half_only=second_half_only,
                rng=rng,
                mode=sampling_mode,
                min_gap_fraction=min_gap_fraction,
            )
            end_time = float(values["Time"].iloc[-1])
            prefix_sum = values[HISTORY_SOURCE_COLUMNS].cumsum().to_numpy(
                np.float64
            )
            cycle_id = "{}::{}::{}".format(
                battery_id, Path(raw_name).stem, cycle_number
            )

            for row_index in sampled_indices:
                row_index = int(row_index)
                snapshot = values.iloc[row_index]
                if history_includes_current_row or row_index == 0:
                    means = prefix_sum[row_index] / float(row_index + 1)
                else:
                    means = prefix_sum[row_index - 1] / float(row_index)

                item = {
                    column: float(snapshot[column])
                    for column in SNAPSHOT_COLUMNS
                }
                item.update({
                    "mean_{}".format(column): float(value)
                    for column, value in zip(HISTORY_SOURCE_COLUMNS, means)
                })
                item.update({
                    TARGET_COLUMN: end_time - float(snapshot["Time"]),
                    "battery_id": battery_id,
                    "cycle_id": cycle_id,
                    "source_file": str(cycle_path),
                    "source_row": row_index,
                })
                rows.append(item)

            audit.append({
                "cycle_id": cycle_id,
                "battery_id": battery_id,
                "source_file": str(cycle_path),
                "valid_rows": int(len(values)),
                "sampled_rows": int(len(sampled_indices)),
                "status": "ok",
            })
        except Exception as exc:
            audit.append({
                "cycle_id": "unreadable::{}".format(cycle_number),
                "battery_id": battery_id,
                "source_file": raw_name,
                "valid_rows": 0,
                "sampled_rows": 0,
                "status": "ERROR: {}: {}".format(type(exc).__name__, exc),
            })
        if (cycle_number + 1) % 250 == 0:
            print(
                "  processed {:,}/{:,} discharge cycles".format(
                    cycle_number + 1, len(discharge)
                ),
                flush=True,
            )

    regression_table = pd.DataFrame(rows)
    audit_table = pd.DataFrame(audit)
    if regression_table.empty:
        raise RuntimeError(
            "No usable samples were built. First audit entries:\n{}".format(
                audit_table.head(20).to_string(index=False)
            )
        )
    if (regression_table[TARGET_COLUMN] < 0).any():
        warnings.warn(
            "Negative remaining-time targets were found; inspect non-monotonic "
            "Time columns in the source data."
        )
    errors = audit_table[~audit_table["status"].eq("ok")]
    if len(errors):
        warnings.warn(
            "{} discharge cycles could not be read. Inspect cycle_audit.csv.".format(
                len(errors)
            )
        )
    return regression_table, audit_table


def build_archive_split(
    regression_table: pd.DataFrame,
    seed: int,
) -> ArchiveSplit:
    """Archived row-level RandomState 80/8/20 split.

    RandomState(seed) first reserves 20% as test. Scalers are fitted on the
    complete 80% training pool. Ten percent of that pool then becomes
    validation, giving effective fractions of 72% train, 8% validation and
    20% test.
    """
    seed = int(seed)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(regression_table))
    train_pool_size = int(0.8 * len(regression_table))
    train_pool_index = order[:train_pool_size]
    test_index = order[train_pool_size:]

    x_pool_raw = regression_table.iloc[train_pool_index][
        FEATURE_COLUMNS
    ].to_numpy(np.float32)
    y_pool_raw = regression_table.iloc[train_pool_index][
        [TARGET_COLUMN]
    ].to_numpy(np.float32)
    x_test_raw = regression_table.iloc[test_index][
        FEATURE_COLUMNS
    ].to_numpy(np.float32)
    y_test_raw = regression_table.iloc[test_index][
        [TARGET_COLUMN]
    ].to_numpy(np.float32)

    x_scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(x_pool_raw)
    y_scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(y_pool_raw)
    x_pool = x_scaler.transform(x_pool_raw).astype(np.float32)
    y_pool = y_scaler.transform(y_pool_raw).astype(np.float32)
    x_test = x_scaler.transform(x_test_raw).astype(np.float32)
    y_test = y_scaler.transform(y_test_raw).astype(np.float32)

    x_pool_tensor = torch.from_numpy(x_pool).to(DEVICE)
    y_pool_tensor = torch.from_numpy(y_pool).to(DEVICE)
    try:
        generator = torch.Generator(device=DEVICE)
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(seed)
    try:
        pool_permutation = torch.randperm(
            len(x_pool_tensor), generator=generator, device=DEVICE
        )
    except RuntimeError:
        pool_permutation = torch.randperm(
            len(x_pool_tensor), generator=generator
        ).to(DEVICE)

    validation_size = int(max(1, round(0.10 * len(x_pool_tensor))))
    validation_local = pool_permutation[:validation_size]
    train_local = pool_permutation[validation_size:]

    return ArchiveSplit(
        seed=seed,
        x_train=x_pool_tensor[train_local],
        y_train=y_pool_tensor[train_local],
        x_validation=x_pool_tensor[validation_local],
        y_validation=y_pool_tensor[validation_local],
        x_test=torch.from_numpy(x_test).to(DEVICE),
        y_test=torch.from_numpy(y_test).to(DEVICE),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        train_pool_index=train_pool_index,
        train_index=train_pool_index[
            train_local.detach().cpu().numpy()
        ],
        validation_index=train_pool_index[
            validation_local.detach().cpu().numpy()
        ],
        test_index=test_index,
    )


# =============================================================================
# Frozen 3-50-50-1 digital twin
# =============================================================================

class MLP3x50x50x1(nn.Module):
    """Architecture used for the supplied B1 digital twin state dictionary."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def strip_state_prefix(state: dict, prefix: str) -> dict:
    if state and all(str(key).startswith(prefix) for key in state):
        return {
            str(key)[len(prefix):]: value for key, value in state.items()
        }
    return state


def load_digital_twin(path: Path) -> nn.Module:
    """Load common module/state-dict formats and normalise to float32."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "Digital twin not found: {}\n"
            "Place B1_MLP_3_50_50_1.pt beside this script or pass --twin.".format(
                path
            )
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    model = None
    state = None
    if isinstance(payload, nn.Module):
        model = payload
    elif isinstance(payload, (tuple, list)):
        if payload and isinstance(payload[0], nn.Module):
            model = payload[0]
        elif payload and isinstance(payload[0], dict):
            state = payload[0]
    elif isinstance(payload, dict):
        for key in ("model", "twin", "dnpu", "device_model"):
            if isinstance(payload.get(key), nn.Module):
                model = payload[key]
                break
        if model is None:
            state = payload.get(
                "state_dict", payload.get("model_state_dict")
            )
            if state is None and payload and all(
                isinstance(key, str) and torch.is_tensor(value)
                for key, value in payload.items()
            ):
                state = payload

    if model is None and state is not None:
        state = strip_state_prefix(state, "module.")
        variants = [state]
        for prefix in ("model.", "twin.", "network.", "net."):
            variant = strip_state_prefix(state, prefix)
            if variant is not state:
                variants.append(variant)

        candidates = [
            MLP3x50x50x1(),
            nn.Sequential(
                nn.Linear(3, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.ReLU(),
                nn.Linear(50, 1),
            ),
        ]
        errors = []
        for candidate in candidates:
            for candidate_state in variants:
                try:
                    candidate.load_state_dict(candidate_state, strict=True)
                    model = candidate
                    break
                except RuntimeError as exc:
                    errors.append(str(exc))
            if model is not None:
                break
        if model is None:
            raise RuntimeError(
                "Could not map the state dictionary in {} to 3-50-50-1. "
                "First error: {}".format(
                    path, errors[0] if errors else "unknown"
                )
            )

    if model is None:
        raise TypeError(
            "Unsupported twin payload type {} in {}".format(
                type(payload).__name__, path
            )
        )

    # Some archived twins were serialised in half precision. Converting the
    # frozen module to float32 here prevents the Half/Float mismatch which can
    # otherwise appear before autocast has a chance to do anything sensible.
    model = model.to(DEVICE).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        probe = torch.linspace(
            -1.0, 1.0, 24, device=DEVICE
        ).reshape(8, 3)
        output = model(probe).reshape(-1)
    if output.numel() != 8 or not bool(torch.isfinite(output).all()):
        raise ValueError("The digital twin failed its finite 8x3 probe")
    return model


# =============================================================================
# Original physical KAN circuit
# =============================================================================

class OriginalCircuitPhysicalKAN(nn.Module):
    """Original constant-control physical KAN used for the paper result."""

    def __init__(
        self,
        twin: nn.Module,
        architecture: list[int],
        devices_per_edge: int,
        hp: dict[str, Any],
    ):
        super().__init__()
        self.twin = twin
        self.architecture = [int(value) for value in architecture]
        self.devices_per_edge = int(devices_per_edge)
        self.hp = dict(hp)
        self.controls_per_device = 2
        self.squash_gain = 6.0
        self.physical_epsilon = 0.02
        self.output_gain_scale = float(hp["output_gain_scale"])
        for parameter in self.twin.parameters():
            parameter.requires_grad_(False)
        self.twin.eval()

        self.S_min = nn.ParameterList()
        self.S_max = nn.ParameterList()
        self.V = nn.ParameterList()
        self.G_o = nn.ParameterList()
        self.B_o = nn.ParameterList()
        for n_in, n_out in zip(
            self.architecture[:-1], self.architecture[1:]
        ):
            shape = (n_in, n_out, self.devices_per_edge)
            self.S_min.append(nn.Parameter(torch.zeros(shape, device=DEVICE)))
            self.S_max.append(nn.Parameter(torch.zeros(shape, device=DEVICE)))
            self.V.append(nn.Parameter(
                torch.zeros(
                    *shape, self.controls_per_device, device=DEVICE
                )
            ))
            self.G_o.append(nn.Parameter(torch.zeros(shape, device=DEVICE)))
            self.B_o.append(nn.Parameter(torch.zeros(shape, device=DEVICE)))
        self._initialise()

    def train(self, mode: bool = True):
        super().train(mode)
        # The twin never enters training mode. This matters if someone releases
        # a twin containing dropout or batch normalisation in future.
        self.twin.eval()
        return self

    def squash_signal(self, raw: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.squash_gain * raw) - 1.0

    @staticmethod
    def squash_control(raw: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(raw) - 1.0

    def inverse_signal(self, value: torch.Tensor) -> torch.Tensor:
        probability = ((value + 1.0) * 0.5).clamp(1e-6, 1.0 - 1e-6)
        return (
            torch.log(probability / (1.0 - probability)) /
            self.squash_gain
        )

    def _initialise_ranges(self, layer: int, n_in: int, n_out: int) -> None:
        shape = (n_in, n_out, self.devices_per_edge)
        minimum_width = float(self.hp["init_dv_min"])
        maximum_width = min(
            float(self.hp["init_dv_max"]),
            2.0 - 2.0 * self.physical_epsilon,
        )
        count = int(np.prod(shape))

        # Start from a stratified collection of physical voltage spans, add a
        # small amount of jitter, then apply the archived power-law shrinkage.
        widths = torch.linspace(
            minimum_width, maximum_width, count, device=DEVICE
        )
        step = (maximum_width - minimum_width) / float(max(count - 1, 1))
        jitter = (
            (torch.rand(count, device=DEVICE) - 0.5) *
            step * 2.0 * float(self.hp["span_jitter"])
        )
        widths = (widths + jitter).clamp(
            minimum_width, maximum_width
        )
        widths = widths[
            torch.randperm(count, device=DEVICE)
        ].reshape(shape)
        fan = float(n_in * self.devices_per_edge)
        powerlaw_scale = max(
            0.1, min(1.0, fan ** (-float(self.hp["pl_beta"])))
        )
        widths = minimum_width + (
            widths - minimum_width
        ) * powerlaw_scale

        inverted = torch.rand(shape, device=DEVICE) < float(
            self.hp["invert_probability"]
        )
        centre_low = (
            -1.0 + self.physical_epsilon + 0.5 * widths
        )
        centre_high = (
            1.0 - self.physical_epsilon - 0.5 * widths
        )
        centre = centre_low + torch.rand(shape, device=DEVICE) * (
            centre_high - centre_low
        )
        # This zero-centre choice is part of the archived inverted-device
        # initialisation, not a projection imposed during training.
        centre = torch.where(inverted, torch.zeros_like(centre), centre)
        physical_minimum = centre - 0.5 * widths
        physical_maximum = centre + 0.5 * widths
        original_minimum = physical_minimum
        physical_minimum = torch.where(
            inverted, physical_maximum, physical_minimum
        )
        physical_maximum = torch.where(
            inverted, original_minimum, physical_maximum
        )
        self.S_min[layer].copy_(
            self.inverse_signal(physical_minimum.clamp(-0.98, 0.98))
        )
        self.S_max[layer].copy_(
            self.inverse_signal(physical_maximum.clamp(-0.98, 0.98))
        )

    def _initialise(self) -> None:
        with torch.no_grad():
            for layer, (n_in, n_out) in enumerate(
                zip(self.architecture[:-1], self.architecture[1:])
            ):
                self._initialise_ranges(layer, n_in, n_out)

                # pl_alpha applies independently to the two constant control
                # voltages. This is separate from pl_beta above.
                shrink = float(
                    n_in * self.devices_per_edge
                ) ** (-float(self.hp["pl_alpha"]))
                limit = (
                    float(self.hp["control_init_scale"]) *
                    math.sqrt(6.0 / 4.0) * shrink
                )
                raw_minimum = math.log(0.4 / 0.6)
                self.V[layer].uniform_(-limit, limit)
                mask = self.V[layer] < raw_minimum
                while bool(mask.any()):
                    self.V[layer][mask] = torch.empty_like(
                        self.V[layer][mask]
                    ).uniform_(raw_minimum, limit)
                    mask = self.V[layer] < raw_minimum

                # Zero gain/bias is deliberate. The gain immediately moves
                # away from zero under BP; the frozen twin itself is untouched.
                self.G_o[layer].zero_()
                self.B_o[layer].zero_()

    def _twin_forward(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, len(flat_inputs), TWIN_MICROBATCH):
            outputs.append(
                self.twin(
                    flat_inputs[start:start + TWIN_MICROBATCH]
                ).reshape(-1)
            )
        return torch.cat(outputs, dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activation = inputs
        for layer, (n_in, n_out) in enumerate(
            zip(self.architecture[:-1], self.architecture[1:])
        ):
            scalar = activation.unsqueeze(2).unsqueeze(3)
            physical_minimum = self.squash_signal(self.S_min[layer])
            physical_maximum = self.squash_signal(self.S_max[layer])
            signal = (
                0.5 * (physical_maximum - physical_minimum).unsqueeze(0) *
                scalar +
                0.5 * (physical_maximum + physical_minimum).unsqueeze(0)
            ).unsqueeze(-1)
            controls = self.squash_control(
                self.V[layer]
            ).unsqueeze(0).expand(
                len(activation), -1, -1, -1, -1
            )
            twin_inputs = torch.cat([signal, controls], dim=-1)
            twin_outputs = self._twin_forward(
                twin_inputs.reshape(-1, 3)
            ).reshape(
                len(activation),
                n_in,
                n_out,
                self.devices_per_edge,
            )
            contribution = (
                self.output_gain_scale *
                self.G_o[layer].unsqueeze(0) * twin_outputs +
                self.B_o[layer].unsqueeze(0)
            )
            activation = contribution.sum(dim=(1, 3))
        return activation

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )


def build_physical_kan(twin: nn.Module) -> OriginalCircuitPhysicalKAN:
    model = OriginalCircuitPhysicalKAN(
        twin=twin,
        architecture=ARCHITECTURE,
        devices_per_edge=DEVICES_PER_EDGE,
        hp=RELEASE_HP,
    ).to(DEVICE)
    if model.trainable_parameter_count() != 8640:
        raise AssertionError(
            "[9,12,1] with 12 devices per edge should contain exactly "
            "8,640 trainable circuit parameters; found {}".format(
                model.trainable_parameter_count()
            )
        )
    return model


# =============================================================================
# Optimisation and scoring
# =============================================================================

def compact_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy the KAN state without duplicating the frozen digital twin."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("twin.")
    }


def average_states(
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    averaged = {}
    for key in states[0]:
        reference = states[0][key]
        if torch.is_floating_point(reference):
            value = sum(
                state[key].float() for state in states
            ) / float(len(states))
            averaged[key] = value.to(dtype=reference.dtype)
        else:
            averaged[key] = reference.clone()
    return averaged


def make_parameter_groups(
    model: OriginalCircuitPhysicalKAN,
) -> list[dict[str, Any]]:
    """Three LRs, corresponding to three genuinely different controls."""
    return [
        {
            "name": "gains_biases",
            "params": list(model.G_o) + list(model.B_o),
            "lr": float(RELEASE_HP["lr_gobo"]),
            "weight_decay": 0.0,
        },
        {
            "name": "controls",
            "params": list(model.V),
            "lr": float(RELEASE_HP["lr_v"]),
            "weight_decay": 0.0,
        },
        {
            "name": "ranges",
            "params": list(model.S_min) + list(model.S_max),
            "lr": float(RELEASE_HP["lr_s"]),
            "weight_decay": 0.0,
        },
    ]


def make_optimizer(
    model: OriginalCircuitPhysicalKAN,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        make_parameter_groups(model),
        betas=(float(RELEASE_HP["beta1"]), float(RELEASE_HP["beta2"])),
        eps=float(RELEASE_HP["adam_eps"]),
    )


def cosine_scale(epoch: int) -> float:
    progress = (int(epoch) - 1) / float(max(MAX_EPOCHS - 1, 1))
    minimum = float(RELEASE_HP["lr_min_frac"])
    return minimum + 0.5 * (1.0 - minimum) * (
        1.0 + math.cos(math.pi * progress)
    )


def set_learning_rates(
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    scale = cosine_scale(epoch)
    base_rates = [
        float(RELEASE_HP["lr_gobo"]),
        float(RELEASE_HP["lr_v"]),
        float(RELEASE_HP["lr_s"]),
    ]
    for group, base_rate in zip(optimizer.param_groups, base_rates):
        group["lr"] = base_rate * scale


@torch.no_grad()
def predict_tensor(
    model: nn.Module,
    inputs: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    outputs = []
    amp_enabled = bool(USE_AMP and DEVICE.type == "cuda")
    for start in range(0, len(inputs), int(chunk_size)):
        with autocast_context(amp_enabled):
            output = model(inputs[start:start + int(chunk_size)])
        outputs.append(output.float())
    if was_training:
        model.train()
    return torch.cat(outputs, dim=0)


def validation_mse(model: nn.Module, split: ArchiveSplit) -> float:
    return float(
        F.mse_loss(
            predict_tensor(model, split.x_validation),
            split.y_validation,
        ).item()
    )


def train_one_epoch(
    model: OriginalCircuitPhysicalKAN,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    split: ArchiveSplit,
    epoch: int,
) -> float:
    set_learning_rates(optimizer, epoch)
    model.train()
    order = torch.randperm(len(split.x_train), device=DEVICE)
    running_loss = 0.0
    seen = 0
    amp_enabled = bool(USE_AMP and DEVICE.type == "cuda")
    for start in range(0, len(order), int(RELEASE_HP["batch_size"])):
        index = order[start:start + int(RELEASE_HP["batch_size"])]
        x_batch = split.x_train[index]
        y_batch = split.y_train[index]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(amp_enabled):
            prediction = model(x_batch)
            loss = F.mse_loss(prediction, y_batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "Non-finite training loss at epoch {}".format(epoch)
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
            ],
            float(RELEASE_HP["grad_clip"]),
        )
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.detach()) * len(index)
        seen += len(index)
    return running_loss / float(max(seen, 1))


def select_initialisation(
    twin: nn.Module,
    split: ArchiveSplit,
    run_seed: int,
) -> tuple[TrainingRuntime, pd.DataFrame]:
    """Train 10 starts for 10 epochs and retain the complete best runtime."""
    print(
        "\nWarm-start selection: {} initialisations x {} epochs".format(
            WARMUP_INITIALISATIONS, WARMUP_EPOCHS
        ),
        flush=True,
    )
    best_runtime = None
    rows = []
    for init_index in range(WARMUP_INITIALISATIONS):
        init_seed = int(run_seed) * 100 + init_index
        seed_everything(init_seed)
        model = build_physical_kan(twin)
        optimizer = make_optimizer(model)
        scaler = make_grad_scaler(USE_AMP and DEVICE.type == "cuda")
        train_reference = float("inf")
        last_improvement_epoch = 0
        history = []
        train_mse = float("nan")
        for epoch in range(1, WARMUP_EPOCHS + 1):
            train_mse = train_one_epoch(
                model, optimizer, scaler, split, epoch
            )
            history.append({
                "epoch": epoch,
                "train_mse": float(train_mse),
                "validation_mse": np.nan,
            })
            if train_mse < train_reference - EARLY_STOPPING_MIN_DELTA:
                train_reference = float(train_mse)
                last_improvement_epoch = epoch

        val_mse = validation_mse(model, split)
        history[-1]["validation_mse"] = float(val_mse)
        state = compact_trainable_state(model)
        rows.append({
            "init_index": init_index,
            "init_seed": init_seed,
            "train_mse_epoch10": float(train_mse),
            "validation_mse_epoch10": float(val_mse),
        })
        print(
            "  init {:02d}/{:02d} seed={} | train={:.6e} | val={:.6e}".format(
                init_index + 1,
                WARMUP_INITIALISATIONS,
                init_seed,
                train_mse,
                val_mse,
            ),
            flush=True,
        )
        candidate = TrainingRuntime(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            current_epoch=WARMUP_EPOCHS,
            best_validation=float(val_mse),
            best_epoch=WARMUP_EPOCHS,
            best_state=state,
            top_checkpoints=[(float(val_mse), WARMUP_EPOCHS, state)],
            train_reference=train_reference,
            last_train_improvement_epoch=last_improvement_epoch,
            init_index=init_index,
            init_seed=init_seed,
            history=history,
        )
        if (
            best_runtime is None or
            candidate.best_validation < best_runtime.best_validation
        ):
            if best_runtime is not None:
                del best_runtime.model
                del best_runtime.optimizer
                del best_runtime.scaler
            best_runtime = candidate
        else:
            del candidate.model
            del candidate.optimizer
            del candidate.scaler
            del candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_runtime is None:
        raise RuntimeError("No finite warm-start initialisation survived")
    print(
        "Selected init {} (seed {}) | epoch-10 val={:.6e}".format(
            best_runtime.init_index,
            best_runtime.init_seed,
            best_runtime.best_validation,
        ),
        flush=True,
    )
    return best_runtime, pd.DataFrame(rows)


def update_top_checkpoints(
    runtime: TrainingRuntime,
    validation: float,
    epoch: int,
) -> bool:
    qualifies = (
        len(runtime.top_checkpoints) < CHECKPOINT_SOUP_TOP_K or
        validation < runtime.top_checkpoints[-1][0]
    )
    improved = validation < runtime.best_validation - 1e-12
    state = (
        compact_trainable_state(runtime.model)
        if qualifies or improved else None
    )
    if improved:
        runtime.best_validation = float(validation)
        runtime.best_epoch = int(epoch)
        runtime.best_state = state
        runtime.soup_size = 1
    if qualifies:
        runtime.top_checkpoints.append(
            (float(validation), int(epoch), state)
        )
        runtime.top_checkpoints = sorted(
            runtime.top_checkpoints, key=lambda item: item[0]
        )[:CHECKPOINT_SOUP_TOP_K]
    return improved


def continue_training(
    runtime: TrainingRuntime,
    split: ArchiveSplit,
) -> TrainingRuntime:
    """Continue the selected start to epoch 800 without restarting Adam."""
    # This makes the continuation independent of how many losing warm starts
    # were evaluated after the selected one.
    seed_everything(
        runtime.init_seed + 10_000_000 + runtime.current_epoch
    )
    started = time.time()
    for epoch in range(runtime.current_epoch + 1, MAX_EPOCHS + 1):
        train_mse = train_one_epoch(
            runtime.model,
            runtime.optimizer,
            runtime.scaler,
            split,
            epoch,
        )
        runtime.current_epoch = epoch
        meaningful = (
            train_mse <
            runtime.train_reference - EARLY_STOPPING_MIN_DELTA
        )
        if meaningful:
            runtime.train_reference = float(train_mse)
            runtime.last_train_improvement_epoch = epoch

        evaluate = epoch % EVALUATE_EVERY == 0 or epoch == MAX_EPOCHS
        val_mse = np.nan
        improved = False
        if evaluate:
            val_mse = validation_mse(runtime.model, split)
            improved = update_top_checkpoints(runtime, val_mse, epoch)

        runtime.history.append({
            "epoch": epoch,
            "train_mse": float(train_mse),
            "validation_mse": float(val_mse),
        })
        if evaluate:
            learning_rates = ", ".join(
                "{}={:.2e}".format(
                    group.get("name", "lr"), group["lr"]
                )
                for group in runtime.optimizer.param_groups
            )
            print(
                "[seed {}] epoch {:4d}/{:4d} | train={:.6e} | "
                "val={:.6e} | best={:.6e} @ {:4d}{} | "
                "elapsed={:.1f} min | {}".format(
                    split.seed,
                    epoch,
                    MAX_EPOCHS,
                    train_mse,
                    val_mse,
                    runtime.best_validation,
                    runtime.best_epoch,
                    " *" if improved else "",
                    (time.time() - started) / 60.0,
                    learning_rates,
                ),
                flush=True,
            )

        if (
            epoch >= EARLY_STOPPING_MIN_EPOCH and
            epoch - runtime.last_train_improvement_epoch >=
            EARLY_STOPPING_PATIENCE
        ):
            runtime.early_stopped = True
            print(
                "EARLY STOP | no training-MSE improvement larger than "
                "{:.2e} for {} epochs; best validation {:.6e} @ {}".format(
                    EARLY_STOPPING_MIN_DELTA,
                    EARLY_STOPPING_PATIENCE,
                    runtime.best_validation,
                    runtime.best_epoch,
                ),
                flush=True,
            )
            break
    return runtime


def select_checkpoint_soup(
    runtime: TrainingRuntime,
    split: ArchiveSplit,
) -> tuple[TrainingRuntime, pd.DataFrame]:
    """Validation-select an average of the strongest late checkpoints."""
    ranked = sorted(
        runtime.top_checkpoints, key=lambda item: item[0]
    )
    rows = [{
        "soup_size": 1,
        "validation_mse": float(runtime.best_validation),
        "selected": True,
    }]
    for soup_size in range(2, len(ranked) + 1):
        soup_state = average_states([
            item[2] for item in ranked[:soup_size]
        ])
        runtime.model.load_state_dict(soup_state, strict=False)
        score = validation_mse(runtime.model, split)
        print(
            "[seed {} soup top {}] val={:.6e}".format(
                split.seed, soup_size, score
            ),
            flush=True,
        )
        rows.append({
            "soup_size": soup_size,
            "validation_mse": float(score),
            "selected": False,
        })
        if score < runtime.best_validation:
            runtime.best_validation = float(score)
            runtime.best_state = soup_state
            runtime.best_epoch = -int(soup_size)
            runtime.soup_size = int(soup_size)

    # Make the in-memory model exactly the validation-selected model before any
    # train/test metric is evaluated.
    runtime.model.load_state_dict(runtime.best_state, strict=False)
    for row in rows:
        row["selected"] = int(row["soup_size"]) == int(runtime.soup_size)
    return runtime, pd.DataFrame(rows)


def regression_metrics(
    target_scaled: np.ndarray,
    prediction_scaled: np.ndarray,
    scaler: MinMaxScaler,
) -> dict[str, float]:
    target_scaled = np.asarray(
        target_scaled, dtype=np.float64
    ).reshape(-1, 1)
    prediction_scaled = np.asarray(
        prediction_scaled, dtype=np.float64
    ).reshape(-1, 1)
    target_seconds = scaler.inverse_transform(target_scaled).ravel()
    prediction_seconds = scaler.inverse_transform(
        prediction_scaled
    ).ravel()
    mse_seconds = mean_squared_error(
        target_seconds, prediction_seconds
    )
    variance = float(np.var(target_seconds))
    return {
        "paper_nMSE": float(
            mean_squared_error(target_scaled, prediction_scaled)
        ),
        "variance_nMSE": (
            float(mse_seconds / variance)
            if variance > 0 else float("nan")
        ),
        "RMSE_seconds": float(math.sqrt(mse_seconds)),
        "MAE_seconds": float(
            mean_absolute_error(target_seconds, prediction_seconds)
        ),
        "R2": float(r2_score(target_seconds, prediction_seconds)),
    }


def score_partition(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    scaler: MinMaxScaler,
) -> dict[str, float]:
    predictions = predict_tensor(model, inputs).detach().cpu().numpy()
    return regression_metrics(
        targets.detach().cpu().numpy(), predictions, scaler
    )


def plot_history(history: pd.DataFrame, output_path: Path, seed: int) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(
        history["epoch"],
        history["train_mse"],
        color="#0b3b70",
        linewidth=1.5,
        label="train MSE",
    )
    validation = history.dropna(subset=["validation_mse"])
    axis.plot(
        validation["epoch"],
        validation["validation_mse"],
        "o-",
        color="#e56b1f",
        markersize=3.0,
        linewidth=1.2,
        label="validation MSE",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Scaled-target MSE (paper nMSE)")
    axis.set_title("NASA battery physical KAN — seed {}".format(seed))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# =============================================================================
# One complete seed and final release entry point
# =============================================================================

def run_one_seed(
    twin: nn.Module,
    regression_table: pd.DataFrame,
    config: ReleaseConfig,
    seed: int,
) -> dict[str, Any]:
    seed = int(seed)
    seed_dir = config.output_dir / "seed_{}".format(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 100)
    print("NASA BATTERY PHYSICAL KAN | SEED {}".format(seed))
    print("=" * 100)

    split = build_archive_split(regression_table, seed)
    print(
        "rows | train={:,} | validation={:,} | test={:,}".format(
            len(split.x_train),
            len(split.x_validation),
            len(split.x_test),
        )
    )
    started = time.time()
    runtime, initialisations = select_initialisation(
        twin=twin,
        split=split,
        run_seed=seed,
    )
    runtime = continue_training(runtime, split)
    runtime, soup_table = select_checkpoint_soup(runtime, split)

    # Hyperparameters and checkpoint selection are now completely locked. Only
    # at this point is the test partition evaluated.
    train_metrics = score_partition(
        runtime.model, split.x_train, split.y_train, split.y_scaler
    )
    validation_metrics = score_partition(
        runtime.model,
        split.x_validation,
        split.y_validation,
        split.y_scaler,
    )
    test_metrics = score_partition(
        runtime.model, split.x_test, split.y_test, split.y_scaler
    )

    initialisations.to_csv(
        seed_dir / "warm_start_initialisations.csv", index=False
    )
    history = pd.DataFrame(runtime.history)
    history.to_csv(seed_dir / "training_history.csv", index=False)
    soup_table.to_csv(
        seed_dir / "checkpoint_soup_selection.csv", index=False
    )
    plot_history(
        history,
        seed_dir / "training_curve.png",
        seed,
    )
    joblib.dump(
        {
            "x_scaler": split.x_scaler,
            "y_scaler": split.y_scaler,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        seed_dir / "scalers.joblib",
    )
    torch.save(
        {
            "format_version": 1,
            "architecture": ARCHITECTURE,
            "devices_per_edge": DEVICES_PER_EDGE,
            "digital_twin_filename": config.twin_path.name,
            "digital_twin_sha256": sha256_file(config.twin_path),
            "seed": seed,
            "selected_init_index": int(runtime.init_index),
            "selected_init_seed": int(runtime.init_seed),
            "best_epoch": int(runtime.best_epoch),
            "checkpoint_soup_size": int(runtime.soup_size),
            "hyperparameters": RELEASE_HP,
            "trainable_state_dict": runtime.best_state,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        seed_dir / "best_physical_kan_state.pt",
    )

    elapsed_minutes = (time.time() - started) / 60.0
    result = {
        "seed": seed,
        "selected_init_index": int(runtime.init_index),
        "selected_init_seed": int(runtime.init_seed),
        "best_epoch": int(runtime.best_epoch),
        "checkpoint_soup_size": int(runtime.soup_size),
        "epochs_ran": int(runtime.current_epoch),
        "early_stopped": bool(runtime.early_stopped),
        "elapsed_minutes": float(elapsed_minutes),
    }
    for partition, metrics in (
        ("train", train_metrics),
        ("validation", validation_metrics),
        ("test", test_metrics),
    ):
        for name, value in metrics.items():
            result["{}_{}".format(partition, name)] = float(value)

    with (seed_dir / "metrics.json").open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print("\nSEED {} COMPLETE".format(seed))
    print(
        "validation paper nMSE: {:.6e}".format(
            result["validation_paper_nMSE"]
        )
    )
    print(
        "test paper nMSE:       {:.6e}".format(
            result["test_paper_nMSE"]
        )
    )
    print("test R2:                {:.6f}".format(result["test_R2"]))
    print(
        "test RMSE:              {:.2f} seconds".format(
            result["test_RMSE_seconds"]
        )
    )
    return result


def main() -> None:
    global DEVICE, USE_AMP, DETERMINISTIC_ALGORITHMS

    arguments = parse_arguments()
    DEVICE = choose_device(arguments.device)
    USE_AMP = bool(not arguments.no_amp and DEVICE.type == "cuda")
    DETERMINISTIC_ALGORITHMS = bool(arguments.deterministic_algorithms)
    config = ReleaseConfig(
        data_root=arguments.data_root,
        twin_path=arguments.twin,
        output_dir=arguments.output_dir,
        seeds=arguments.seeds,
        use_amp=USE_AMP,
        deterministic_algorithms=bool(arguments.deterministic_algorithms),
        device_request=arguments.device,
    )
    config.data_root = locate_dataset_root(config.data_root)
    config.twin_path = config.twin_path.expanduser().resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("device:", DEVICE)
    print("automatic mixed precision:", USE_AMP)
    print("dataset:", config.data_root)
    print("digital twin:", config.twin_path)
    print("outputs:", config.output_dir.resolve())
    print("seeds:", config.seeds)
    print("versions:")
    print(json.dumps(versions(), indent=2, sort_keys=True))

    seed_everything(
        config.sampling_seed,
        deterministic_algorithms=config.deterministic_algorithms,
    )
    regression_table, cycle_audit = build_regression_table(
        data_root=config.data_root,
        sampling_seed=config.sampling_seed,
        samples_per_cycle=config.samples_per_cycle,
        second_half_only=config.second_half_only,
        sampling_mode=config.sampling_mode,
        min_gap_fraction=config.min_gap_fraction,
        history_includes_current_row=config.history_includes_current_row,
    )
    manifest_path = config.output_dir / "sampled_rows_manifest.csv"
    audit_path = config.output_dir / "cycle_audit.csv"
    regression_table.to_csv(manifest_path, index=False)
    cycle_audit.to_csv(audit_path, index=False)

    valid_cycles = int(cycle_audit["status"].eq("ok").sum())
    dataset_summary = {
        "metadata_sha256": sha256_file(config.data_root / "metadata.csv"),
        "sampled_rows_manifest_sha256": sha256_file(manifest_path),
        "cycle_audit_sha256": sha256_file(audit_path),
        "valid_discharge_cycles": valid_cycles,
        "inferred_batteries": int(regression_table["battery_id"].nunique()),
        "sampled_rows": int(len(regression_table)),
        "sampling_seed": int(config.sampling_seed),
        "samples_per_cycle": int(config.samples_per_cycle),
        "second_half_only": bool(config.second_half_only),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    print("dataset summary:")
    print(json.dumps(dataset_summary, indent=2, sort_keys=True))

    # These are the counts seen in the full archive used for the paper. A
    # warning here is useful: silently running a truncated Kaggle extraction is
    # much worse than stopping to inspect what was actually downloaded.
    expected = {
        "valid_discharge_cycles": 2766,
        "inferred_batteries": 33,
        "sampled_rows": 138300,
    }
    observed = {name: dataset_summary[name] for name in expected}
    if observed != expected:
        warnings.warn(
            "Dataset counts differ from the full paper archive. "
            "Expected {}; observed {}. The run will continue, but inspect "
            "cycle_audit.csv and the extracted dataset before comparing the "
            "number directly with the paper.".format(expected, observed)
        )

    twin = load_digital_twin(config.twin_path)
    preflight = build_physical_kan(twin)
    assert preflight.architecture == [9, 12, 1]
    assert preflight.devices_per_edge == 12
    assert preflight.trainable_parameter_count() == 8640
    assert not hasattr(preflight, "M")
    assert not hasattr(preflight, "R")
    assert len(make_parameter_groups(preflight)) == 3
    probe_split = build_archive_split(regression_table, config.seeds[0])
    probe_prediction = preflight(probe_split.x_train[:2])
    assert tuple(probe_prediction.shape) == (2, 1)
    assert bool(torch.isfinite(probe_prediction).all())
    print(
        "PREFLIGHT PASSED | [9,12,1] | 12 devices/edge | "
        "8,640 trainable circuit parameters | original circuit"
    )
    del preflight, probe_prediction, probe_split
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_manifest = {
        "script": Path(__file__).name,
        "dataset_download_url": DATASET_DOWNLOAD_URL,
        "nasa_source_url": NASA_SOURCE_URL,
        "dataset": dataset_summary,
        "digital_twin_path": str(config.twin_path),
        "digital_twin_sha256": sha256_file(config.twin_path),
        "architecture": ARCHITECTURE,
        "devices_per_edge": DEVICES_PER_EDGE,
        "training_hyperparameters": RELEASE_HP,
        "max_epochs": MAX_EPOCHS,
        "warmup_initialisations": WARMUP_INITIALISATIONS,
        "warmup_epochs": WARMUP_EPOCHS,
        "checkpoint_soup_top_k": CHECKPOINT_SOUP_TOP_K,
        "seeds": config.seeds,
        "versions": versions(),
        "config": {
            **asdict(config),
            "data_root": str(config.data_root),
            "twin_path": str(config.twin_path),
            "output_dir": str(config.output_dir),
        },
    }
    with (config.output_dir / "run_manifest.json").open("w") as handle:
        json.dump(run_manifest, handle, indent=2, sort_keys=True)

    results = []
    for seed in config.seeds:
        results.append(
            run_one_seed(
                twin=twin,
                regression_table=regression_table,
                config=config,
                seed=seed,
            )
        )
        pd.DataFrame(results).to_csv(
            config.output_dir / "results_by_seed.csv", index=False
        )

    results_table = pd.DataFrame(results)
    metric_columns = [
        "validation_paper_nMSE",
        "test_paper_nMSE",
        "test_RMSE_seconds",
        "test_MAE_seconds",
        "test_R2",
    ]
    summary = {
        "n_seeds": int(len(results_table)),
        "seeds": config.seeds,
        "metrics": {},
    }
    for column in metric_columns:
        values = results_table[column].astype(float)
        summary["metrics"][column] = {
            "mean": float(values.mean()),
            "std": (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    with (config.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("\n" + "=" * 100)
    print("ALL REQUESTED SEEDS COMPLETE")
    print("=" * 100)
    print(results_table[[
        "seed",
        "selected_init_seed",
        "checkpoint_soup_size",
        "validation_paper_nMSE",
        "test_paper_nMSE",
        "test_RMSE_seconds",
        "test_R2",
    ]].to_string(index=False))
    print("\nAggregate summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nOutputs:", config.output_dir.resolve())


if __name__ == "__main__":
    main()
