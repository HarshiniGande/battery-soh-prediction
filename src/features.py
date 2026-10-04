"""Engineer battery health indicators for every charge and discharge cycle pair.

Design rule: no feature may reveal the answer. Capacity is current multiplied by full discharge
time, so the full discharge duration (and anything that only happens at the very end of a
discharge, like the temperature peak) would leak the target. Every feature below comes either
from the charge cycle before the discharge or from an early, partial window of the discharge.
"""

import numpy as np
import pandas as pd

from .data_processing import CUTOFF_VOLTAGE, pair_cycles

FEATURE_COLUMNS = [
    "cc_charge_time_s",
    "cv_charge_time_s",
    "v_drop_time_3p8_3p5_s",
    "voltage_at_500s_v",
    "voltage_at_1000s_v",
    "temp_rise_first_1000s_c",
    "initial_voltage_sag_v",
]

FEATURE_DESCRIPTIONS = {
    "cc_charge_time_s": "Seconds in constant-current charging before voltage reaches 4.2 V",
    "cv_charge_time_s": "Seconds in constant-voltage charging until current falls below 0.1 A",
    "v_drop_time_3p8_3p5_s": "Seconds for discharge voltage to fall from 3.8 V to 3.5 V",
    "voltage_at_500s_v": "Discharge voltage 500 s into the cycle",
    "voltage_at_1000s_v": "Discharge voltage 1000 s into the cycle",
    "temp_rise_first_1000s_c": "Temperature rise (C) in the first 1000 s of discharge",
    "initial_voltage_sag_v": "Voltage drop in the first few seconds of discharge (internal resistance proxy)",
}


def _first_time_at_or_below(t, v, level):
    hits = np.where(v <= level)[0]
    return t[hits[0]] if hits.size else np.nan


def _value_at_time(t, y, when):
    if t[-1] < when:
        return np.nan
    return float(np.interp(when, t, y))


def charge_features(charge: dict) -> dict:
    if charge is None:
        return {"cc_charge_time_s": np.nan, "cv_charge_time_s": np.nan, "charge_start_voltage_v": np.nan}
    d = charge["data"]
    t = np.asarray(d["Time"], float)
    v = np.asarray(d["Voltage_measured"], float)
    i = np.asarray(d["Current_measured"], float)

    reached = np.where(v >= 4.2)[0]
    if reached.size == 0:
        cc_time = cv_time = np.nan
    else:
        k = reached[0]
        cc_time = t[k]
        tapered = np.where((t > t[k]) & (i < 0.1))[0]
        cv_time = t[tapered[0]] - t[k] if tapered.size else np.nan
    # Note: charge temperature rise was tested and dropped. Charge cycles often start right
    # after a discharge while the cell is still warm, so the cell cools during charging and the
    # "rise" measures rest time rather than battery health (it reads 0 for 101 of 132 B0018 cycles).
    # charge_start_voltage_v is a quality-control column, not a model feature: a charge that starts
    # well above normal voltage began from a cell that was not fully discharged.
    return {"cc_charge_time_s": cc_time, "cv_charge_time_s": cv_time, "charge_start_voltage_v": float(v[0])}


def discharge_features(discharge: dict) -> dict:
    d = discharge["data"]
    t = np.asarray(d["Time"], float)
    v = np.asarray(d["Voltage_measured"], float)
    temp = np.asarray(d["Temperature_measured"], float)

    t38 = _first_time_at_or_below(t, v, 3.8)
    t35 = _first_time_at_or_below(t, v, 3.5)
    early = t <= 30
    sag = float(v[0] - v[early].min()) if early.sum() > 1 else np.nan
    t1000 = _value_at_time(t, temp, 1000)
    return {
        "v_drop_time_3p8_3p5_s": t35 - t38,
        "voltage_at_500s_v": _value_at_time(t, v, 500),
        "voltage_at_1000s_v": _value_at_time(t, v, 1000),
        "temp_rise_first_1000s_c": t1000 - temp[0] if not np.isnan(t1000) else np.nan,
        "initial_voltage_sag_v": sag,
    }


def build_cycle_table(cycles: list, battery_id: str) -> pd.DataFrame:
    """One row per discharge cycle: identifiers, engineered features, and targets."""
    rows = []
    for n, (charge, discharge, raw_idx) in enumerate(pair_cycles(cycles), start=1):
        row = {
            "battery_id": battery_id,
            "cycle_number": n,
            "raw_cycle_index": raw_idx,
            "cutoff_voltage_v": CUTOFF_VOLTAGE[battery_id],
            "capacity_ah": float(discharge["data"]["Capacity"]),
        }
        row.update(charge_features(charge))
        row.update(discharge_features(discharge))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["soh_pct"] = 100 * df["capacity_ah"] / df["capacity_ah"].iloc[:5].median()
    return df
