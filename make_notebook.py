"""Build and execute notebooks/battery_soh_analysis.ipynb (run from the project root)."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Lithium-Ion Battery State of Health Prediction
**Data:** NASA Prognostics Center of Excellence (PCoE) battery aging dataset, cells B0005, B0006, B0007, B0018

**Question:** Can we estimate how much capacity a lithium-ion cell has left, for a cell the model has never seen, using signals that are cheap to measure during normal operation?

**Approach:** restructure 2 million raw measurements into cycle-level tables, clean them, engineer health indicators that do not leak the answer, and compare three models under leave-one-battery-out validation."""),
    code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
from IPython.display import Image, display

from src.data_processing import BATTERIES, EOL_CAPACITY_AH, clean_cycle_table, load_battery, to_timeseries
from src.features import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, build_cycle_table
from src.modeling import feature_importance, leave_one_battery_out
from src import visualize as viz

DATA_DIR = Path("../data")
FIG_DIR = Path("../outputs/figures")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)"""),
    md("""## 1. Load and restructure the raw data
Each `.mat` file is a nested MATLAB struct: one battery, a list of cycles (charge, discharge, impedance), and arrays of time-series readings inside each cycle. The first step flattens that into a long table so it can be explored with pandas."""),
    code("""cycles = {b: load_battery(DATA_DIR / f"{b}.mat", b) for b in BATTERIES}
ts = pd.concat([to_timeseries(c, b) for b, c in cycles.items()], ignore_index=True)
print(f"{len(ts):,} raw measurements")
ts.groupby(["battery_id", "cycle_type"]).raw_cycle_index.nunique().unstack()"""),
    code("""ts.head()"""),
    md("""## 2. Engineer health indicators (without leaking the target)
Capacity equals discharge current multiplied by full discharge time, so using full discharge time (or anything that only happens at the end of a discharge, like the temperature peak) would let the model read the answer instead of learning it. Every feature below comes from the **charge cycle before** the discharge or from an **early, partial window** of the discharge."""),
    code("""pd.Series(FEATURE_DESCRIPTIONS, name="description").to_frame()"""),
    code("""raw = pd.concat([build_cycle_table(c, b) for b, c in cycles.items()], ignore_index=True)
raw.head()"""),
    md("""## 3. Clean the data
Three data quality issues showed up during exploration:

1. **Abnormal charge cycles.** Some charges did not start from a fully discharged cell (top-up charges, and the very first charge of each experiment). Their constant-current phase is much shorter, so their timings do not describe battery health. A charge is flagged only when it started at an unusually high voltage **and** its charge time jumped away from neighboring cycles; requiring both keeps real capacity regeneration after rest periods in the data. Flagged values are interpolated from neighboring cycles of the same battery. Capacity is never used in this step.
2. **Capacity measurement glitches.** Rare spikes far from the local rolling median are dropped.
3. **A misleading feature.** Charge temperature rise was tested and removed: charges often begin while the cell is still warm from the previous discharge, so it cools during charging and the "rise" measures rest time, not health (it read 0 for 101 of 132 B0018 cycles)."""),
    code("""df, log = clean_cycle_table(raw)
log"""),
    md("""## 4. Explore"""),
    code("""viz.plot_capacity_fade(df, FIG_DIR)
display(Image(FIG_DIR / "01_capacity_fade.png"))"""),
    md("""All four cells lose capacity steadily, with small upward jumps after rest periods (capacity regeneration). B0006 starts highest and fades fastest, crossing the 1.4 Ah end-of-life line around cycle 109; B0007 never reaches it."""),
    code("""viz.plot_feature_correlations(df, FEATURE_COLUMNS, FIG_DIR)
display(Image(FIG_DIR / "02_feature_correlations.png"))"""),
    code("""importance = feature_importance(df, FEATURE_COLUMNS, "Linear Regression")
viz.plot_features_vs_capacity(df, list(importance.index), FIG_DIR)
display(Image(FIG_DIR / "03_features_vs_capacity.png"))"""),
    md("""The discharge features track capacity closely **within** each battery, but B0006 sits on a shifted curve: at the same voltage behavior it holds more capacity. That offset is cell-to-cell variation, and it matters for generalizing to new cells. The constant-current charge time lines up much more consistently across all four batteries."""),
    md("""## 5. Model with leave-one-battery-out validation
A random train/test split would put neighboring cycles of the same battery on both sides, and the model would score well just by interpolating. Instead, each model is trained on three batteries and tested on the fourth, repeated for all four. Three feature sets are compared to see which signals actually generalize."""),
    code("""charge = ["cc_charge_time_s", "cv_charge_time_s"]
feature_sets = {
    "Charge only (2)": charge,
    "Discharge only (5)": [f for f in FEATURE_COLUMNS if f not in charge],
    "All features (7)": FEATURE_COLUMNS,
}
metrics, preds = [], []
for name, feats in feature_sets.items():
    m, p = leave_one_battery_out(df, feats)
    metrics.append(m.assign(feature_set=name)); preds.append(p.assign(feature_set=name))
metrics, preds = pd.concat(metrics), pd.concat(preds)
summary = (metrics.groupby(["feature_set", "model"])[["rmse_ah", "mae_ah", "mape_pct", "r2"]]
           .mean().reset_index().sort_values("rmse_ah"))
summary.round(4)"""),
    code("""viz.plot_model_comparison(summary, FIG_DIR, order=list(feature_sets))
display(Image(FIG_DIR / "04_model_comparison.png"))"""),
    md("""**Key finding: two simple charge features beat all seven combined.** The discharge features carry battery-specific offsets (like B0006's shifted curve) that do not transfer to a new cell, so adding them makes predictions on unseen batteries worse. The tree models also struggle here because they cannot predict values outside the range they were trained on, and with only three training batteries that range is narrow. Linear Regression extrapolates naturally."""),
    code("""best = summary.iloc[0]
viz.plot_predictions(preds[preds.feature_set == best.feature_set], best.model, FIG_DIR,
                     label=f", {best.feature_set.split(' (')[0].lower()} features")
display(Image(FIG_DIR / "05_predicted_vs_actual.png"))"""),
    code("""best_rows = metrics[(metrics.feature_set == best.feature_set) & (metrics.model == best.model)]
best_rows[["held_out_battery", "rmse_ah", "mape_pct", "r2", "actual_eol_cycle", "predicted_eol_cycle", "eol_error_cycles"]].round(4)"""),
    md("""For the three batteries that reached end of life (1.4 Ah), the predicted end-of-life cycle lands within 1, 4, and 7 cycles of the actual one."""),
    code("""viz.plot_feature_importance(importance, "Linear Regression", FIG_DIR)
display(Image(FIG_DIR / "06_feature_importance.png"))"""),
    md("""## 6. Conclusions
* **Result:** Linear Regression on two charge-phase features estimates capacity on never-seen batteries with **1.2% mean absolute percentage error (R² 0.98)**, averaged over four held-out cells.
* **Why it matters:** both features come from the charge cycle, so a battery management system could estimate health before the battery is even used, without a full discharge test.
* **Data quality drove the result.** Fixing abnormal charge cycles (without touching the target) cut error from 5.9% to 1.2% MAPE and raised R² from 0.66 to 0.98; removing a misleading temperature feature avoided learning rest time instead of health.
* **Simpler generalized better.** More features and more flexible models fit each battery's quirks rather than the shared aging pattern.

**Limitations and next steps**
* Four cells from one lab at room temperature (24 C). Results should be checked on larger datasets with varied temperatures and charge rates (for example the Stanford/MIT/Toyota fast-charging dataset).
* Remaining useful life here is estimated by applying the 1.4 Ah threshold to predicted capacity. A dedicated forecasting model (for example a Gaussian process or LSTM on the capacity trajectory) would forecast further ahead.
* Battery-level random effects or transfer learning could account for cell-to-cell offsets like B0006's."""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path("notebooks/battery_soh_analysis.ipynb")
out.parent.mkdir(exist_ok=True)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
nbf.write(nb, out)
print(f"Wrote and executed {out}")
