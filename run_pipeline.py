"""Run the full project end to end: load, clean, engineer features, model, evaluate, and plot.

Usage:
    python run_pipeline.py                     # standard run
    python run_pipeline.py --save-timeseries   # also export the 2M-row long time-series table (~60 MB)
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src import visualize as viz
from src.data_processing import BATTERIES, clean_cycle_table, load_battery, to_timeseries
from src.features import FEATURE_COLUMNS, build_cycle_table
from src.modeling import feature_importance, leave_one_battery_out

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

CHARGE_FEATURES = ["cc_charge_time_s", "cv_charge_time_s"]
FEATURE_SETS = {
    "Charge only (2)": CHARGE_FEATURES,
    "Discharge only (5)": [f for f in FEATURE_COLUMNS if f not in CHARGE_FEATURES],
    "All features (7)": FEATURE_COLUMNS,
}
METRIC_COLS = ["rmse_ah", "mae_ah", "mape_pct", "r2"]


def load_and_build(save_timeseries: bool = False) -> tuple[pd.DataFrame, dict]:
    cycle_tables, n_measurements, timeseries = [], 0, []
    for bid in BATTERIES:
        cycles = load_battery(DATA_DIR / f"{bid}.mat", bid)
        ts = to_timeseries(cycles, bid)
        n_measurements += len(ts)
        if save_timeseries:
            timeseries.append(ts)
        cycle_tables.append(build_cycle_table(cycles, bid))
    raw = pd.concat(cycle_tables, ignore_index=True)
    print(f"Restructured {n_measurements:,} raw measurements into {len(raw)} discharge cycles")
    if save_timeseries:
        pd.concat(timeseries, ignore_index=True).to_csv(PROCESSED_DIR / "timeseries_long.csv.gz", index=False)
    df, log = clean_cycle_table(raw)
    log["raw_measurements"] = n_measurements
    return df, log


def main(save_timeseries: bool = False):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, log = load_and_build(save_timeseries)
    print("Cleaning log:", log)
    df.to_csv(PROCESSED_DIR / "cycle_features.csv", index=False)
    (OUTPUT_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2))

    # Compare every feature set with every model, holding out one battery at a time
    all_metrics, all_preds = [], []
    for set_name, feats in FEATURE_SETS.items():
        metrics, preds = leave_one_battery_out(df, feats)
        all_metrics.append(metrics.assign(feature_set=set_name))
        all_preds.append(preds.assign(feature_set=set_name))
    metrics = pd.concat(all_metrics, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    summary = (
        metrics.groupby(["feature_set", "model"])[METRIC_COLS].mean().reset_index().sort_values("rmse_ah")
    )
    best = summary.iloc[0]
    print("\nMean across the four held-out batteries:")
    print(summary.round(4).to_string(index=False))
    print(f"\nBest: {best['model']} on {best['feature_set']} "
          f"(RMSE {best['rmse_ah']:.4f} Ah, MAPE {best['mape_pct']:.2f}%, R2 {best['r2']:.3f})")

    metrics.to_csv(OUTPUT_DIR / "metrics_by_battery.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    preds.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    importance = feature_importance(df, FEATURE_COLUMNS, best["model"])
    importance.rename("importance_rmse_ah").to_csv(OUTPUT_DIR / "feature_importance.csv")

    # Figures
    viz.plot_capacity_fade(df, FIG_DIR)
    viz.plot_feature_correlations(df, FEATURE_COLUMNS, FIG_DIR)
    viz.plot_features_vs_capacity(df, list(importance.index), FIG_DIR)
    viz.plot_model_comparison(summary, FIG_DIR, order=list(FEATURE_SETS))
    best_preds = preds[preds["feature_set"] == best["feature_set"]]
    viz.plot_predictions(best_preds, best["model"], FIG_DIR, label=f", {best['feature_set'].split(' (')[0].lower()} features")
    viz.plot_feature_importance(importance, best["model"], FIG_DIR)
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-timeseries", action="store_true")
    main(parser.parse_args().save_timeseries)
