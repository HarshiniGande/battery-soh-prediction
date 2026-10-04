"""Load the NASA PCoE lithium-ion battery aging files and restructure them into tidy tables.

Each raw .mat file stores one battery as a nested MATLAB struct: a list of cycles, where every
cycle has a type (charge, discharge, impedance) and its own arrays of time-series measurements.
This module flattens that structure into two pandas DataFrames:

* a long time-series table (one row per measurement), and
* a cycle-level table (one row per discharge cycle, paired with the charge cycle before it).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

# Discharge cutoff voltages used in the NASA experiment (from the dataset README).
CUTOFF_VOLTAGE = {"B0005": 2.7, "B0006": 2.5, "B0007": 2.2, "B0018": 2.5}

# NASA defines end of life (EOL) as a 30% capacity fade, from 2.0 Ah rated to 1.4 Ah.
EOL_CAPACITY_AH = 1.4


def load_battery(path: Path, battery_id: str) -> list:
    """Return the raw list of cycle dicts for one battery."""
    raw = sio.loadmat(path, simplify_cells=True)
    return raw[battery_id]["cycle"]


def to_timeseries(cycles: list, battery_id: str) -> pd.DataFrame:
    """Flatten every charge and discharge cycle into one long table."""
    frames = []
    for idx, cycle in enumerate(cycles):
        if cycle["type"] not in ("charge", "discharge"):
            continue
        data = cycle["data"]
        frames.append(
            pd.DataFrame(
                {
                    "battery_id": battery_id,
                    "raw_cycle_index": idx,
                    "cycle_type": cycle["type"],
                    "time_s": np.asarray(data["Time"], dtype=float),
                    "voltage_v": np.asarray(data["Voltage_measured"], dtype=float),
                    "current_a": np.asarray(data["Current_measured"], dtype=float),
                    "temperature_c": np.asarray(data["Temperature_measured"], dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def pair_cycles(cycles: list) -> list:
    """Pair each discharge cycle with the most recent charge cycle that came before it.

    Returns a list of (charge_cycle or None, discharge_cycle, raw_index) tuples.
    """
    pairs = []
    last_charge = None
    for idx, cycle in enumerate(cycles):
        if cycle["type"] == "charge":
            last_charge = cycle
        elif cycle["type"] == "discharge":
            pairs.append((last_charge, cycle, idx))
            last_charge = None  # each charge cycle is used at most once
    return pairs


def flag_capacity_outliers(cap: pd.Series, window: int = 7, threshold_ah: float = 0.08) -> pd.Series:
    """Flag cycles whose capacity jumps far from the local rolling median.

    Small upward jumps after rest periods are a real battery effect (capacity regeneration),
    so the threshold is deliberately loose and only catches measurement glitches.
    """
    rolling_median = cap.rolling(window, center=True, min_periods=3).median()
    return (cap - rolling_median).abs() > threshold_ah


CHARGE_FEATURES = ["cc_charge_time_s", "cv_charge_time_s"]
MIN_VALID_CC_TIME_S = 300  # a full charge from the discharge cutoff takes well over 20 minutes
START_VOLTAGE_MARGIN_V = 0.2  # charge started this far above neighboring cycles
MIN_RELATIVE_CC_JUMP = 0.05  # and its charge time moved more than 5% from neighboring cycles


def clean_cycle_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fix invalid charge features, drop capacity glitches, and return a cleaning log."""
    df = df.copy()
    log = {"rows_in": len(df)}

    # 1. Charge cycles that did not start from a fully discharged cell (top-up charges, and the
    #    very first charge of each experiment) have a shorter constant-current phase, so their
    #    timings do not describe battery health. A charge is flagged when it started at an
    #    unusually high voltage AND its charge time jumped away from neighboring cycles. Requiring
    #    both keeps real effects, like capacity regeneration after rest periods, in the data.
    #    Capacity (the target) is never used here, so cleaning cannot leak the answer.
    def local_median(col, window):
        return df.groupby("battery_id")[col].transform(
            lambda s: s.rolling(window, center=True, min_periods=3).median()
        )

    cc = df["cc_charge_time_s"]
    relative_jump = (cc - local_median("cc_charge_time_s", 7)).abs() / local_median("cc_charge_time_s", 7)
    high_start = (df["charge_start_voltage_v"] - local_median("charge_start_voltage_v", 9)) > START_VOLTAGE_MARGIN_V
    invalid_charge = (cc < MIN_VALID_CC_TIME_S) | (high_start & (relative_jump > MIN_RELATIVE_CC_JUMP))
    log["abnormal_charge_cycles_reset"] = int(invalid_charge.sum())
    df.loc[invalid_charge, CHARGE_FEATURES] = np.nan
    log["missing_charge_features_filled"] = int(df[CHARGE_FEATURES].isna().any(axis=1).sum())

    # 2. Fill those gaps (plus the first cycle, which has no charge before it) by interpolating
    #    along the cycle axis within each battery, so no information leaks across batteries.
    df[CHARGE_FEATURES] = (
        df.groupby("battery_id")[CHARGE_FEATURES]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )

    # 3. Drop measurement glitches in the target.
    df["capacity_outlier"] = df.groupby("battery_id")["capacity_ah"].transform(flag_capacity_outliers)
    log["capacity_outliers_dropped"] = int(df["capacity_outlier"].sum())
    df = df[~df["capacity_outlier"]].drop(columns="capacity_outlier")

    # 4. Drop any cycle still missing a feature.
    df = df.drop(columns="charge_start_voltage_v")
    before = len(df)
    df = df.dropna()
    log["rows_with_missing_dropped"] = before - len(df)
    log["rows_out"] = len(df)
    return df.reset_index(drop=True), log
