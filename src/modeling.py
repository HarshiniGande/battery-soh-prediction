"""Train and evaluate capacity (state of health) models with leave-one-battery-out validation.

Why leave-one-battery-out: a random train/test split would put neighboring cycles from the same
battery on both sides, and the model would score well just by interpolating between them. Holding
out a whole battery asks the honest question: can the model estimate the health of a cell it has
never seen?
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .data_processing import EOL_CAPACITY_AH

RANDOM_STATE = 42


def get_models() -> dict:
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE,
        ),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse_ah": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_ah": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(100 * mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def eol_cycle(cycle_numbers, capacity, threshold=EOL_CAPACITY_AH, smooth=5):
    """First cycle where the (lightly smoothed) capacity falls below the end-of-life threshold."""
    smoothed = pd.Series(np.asarray(capacity)).rolling(smooth, center=True, min_periods=1).median()
    below = np.where(smoothed.values < threshold)[0]
    return int(np.asarray(cycle_numbers)[below[0]]) if below.size else None


def leave_one_battery_out(df: pd.DataFrame, features: list, target: str = "capacity_ah"):
    """Return (metrics table, out-of-fold predictions) for every model and held-out battery."""
    metric_rows, prediction_frames = [], []
    for held_out in sorted(df["battery_id"].unique()):
        train, test = df[df["battery_id"] != held_out], df[df["battery_id"] == held_out]
        for name, model in get_models().items():
            model.fit(train[features], train[target])
            pred = model.predict(test[features])
            row = {"model": name, "held_out_battery": held_out, **regression_metrics(test[target], pred)}
            actual_eol = eol_cycle(test["cycle_number"], test[target])
            predicted_eol = eol_cycle(test["cycle_number"], pred)
            row["actual_eol_cycle"] = actual_eol
            row["predicted_eol_cycle"] = predicted_eol
            row["eol_error_cycles"] = (
                predicted_eol - actual_eol if actual_eol is not None and predicted_eol is not None else None
            )
            metric_rows.append(row)
            prediction_frames.append(
                test[["battery_id", "cycle_number", target]].assign(model=name, predicted=pred)
            )
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average each metric across the four held-out batteries."""
    cols = ["rmse_ah", "mae_ah", "mape_pct", "r2"]
    return metrics.groupby("model")[cols].mean().sort_values("rmse_ah")


def feature_importance(df: pd.DataFrame, features: list, model_name: str, target: str = "capacity_ah"):
    """Permutation importance, averaged over the four leave-one-battery-out folds."""
    results = []
    for held_out in sorted(df["battery_id"].unique()):
        train, test = df[df["battery_id"] != held_out], df[df["battery_id"] == held_out]
        model = get_models()[model_name].fit(train[features], train[target])
        imp = permutation_importance(
            model, test[features], test[target], n_repeats=20, random_state=RANDOM_STATE,
            scoring="neg_root_mean_squared_error",
        )
        results.append(pd.Series(imp.importances_mean, index=features, name=held_out))
    return pd.concat(results, axis=1).mean(axis=1).sort_values(ascending=False)
