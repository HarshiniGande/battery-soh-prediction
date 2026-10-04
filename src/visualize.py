"""Figures for the battery state of health project."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_processing import EOL_CAPACITY_AH

PALETTE = {"B0005": "#1f77b4", "B0006": "#d62728", "B0007": "#2ca02c", "B0018": "#9467bd"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_fade(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for bid, g in df.groupby("battery_id"):
        ax.plot(g["cycle_number"], g["capacity_ah"], label=bid, color=PALETTE[bid], lw=1.8)
    ax.axhline(EOL_CAPACITY_AH, color="black", ls="--", lw=1)
    ax.text(2, EOL_CAPACITY_AH + 0.01, "End of life (1.4 Ah, 30% fade)", fontsize=9)
    ax.set(xlabel="Discharge cycle", ylabel="Capacity (Ah)", title="Capacity fade across four NASA Li-ion cells")
    ax.legend(title="Battery")
    _save(fig, out_dir, "01_capacity_fade.png")


def plot_feature_correlations(df: pd.DataFrame, features: list, out_dir: Path):
    corr = df[features + ["capacity_ah"]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation between engineered features and capacity")
    _save(fig, out_dir, "02_feature_correlations.png")


def plot_features_vs_capacity(df: pd.DataFrame, features: list, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), features[:4]):
        for bid, g in df.groupby("battery_id"):
            ax.scatter(g[feat], g["capacity_ah"], s=10, alpha=0.7, color=PALETTE[bid], label=bid)
        ax.set(xlabel=feat, ylabel="Capacity (Ah)")
    axes[0, 0].legend(title="Battery", fontsize=8)
    fig.suptitle("Top engineered health indicators vs capacity", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "03_features_vs_capacity.png")


def plot_model_comparison(summary: pd.DataFrame, out_dir: Path, order=None):
    """summary: one row per (feature_set, model) with mean MAPE across held-out batteries."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="feature_set", y="mape_pct", hue="model", order=order,
                hue_order=["Linear Regression", "Random Forest", "XGBoost"], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
    ax.set(xlabel="Feature set", ylabel="Mean MAPE across held-out batteries (%)",
           title="Capacity error on unseen batteries, by feature set and model (lower is better)")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, out_dir, "04_model_comparison.png")


def plot_predictions(preds: pd.DataFrame, model_name: str, out_dir: Path, label: str = ""):
    sub = preds[preds["model"] == model_name]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, (bid, g) in zip(axes.ravel(), sub.groupby("battery_id")):
        ax.plot(g["cycle_number"], g["capacity_ah"], color="black", lw=1.6, label="Actual")
        ax.plot(g["cycle_number"], g["predicted"], color=PALETTE[bid], lw=1.4, ls="--", label="Predicted")
        ax.axhline(EOL_CAPACITY_AH, color="grey", ls=":", lw=1)
        ax.set(title=f"{bid} (held out)", xlabel="Discharge cycle", ylabel="Capacity (Ah)")
        ax.legend(fontsize=8)
    fig.suptitle(f"{model_name}{label}: predicted vs actual capacity on unseen batteries", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "05_predicted_vs_actual.png")


def plot_feature_importance(importance: pd.Series, model_name: str, out_dir: Path):
    """Permutation importance for a model trained on all seven features."""
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.sort_values().plot.barh(ax=ax, color="#1f77b4")
    ax.set(xlabel="Increase in RMSE when the feature is shuffled (Ah)",
           title=f"Permutation feature importance, all 7 features ({model_name})")
    _save(fig, out_dir, "06_feature_importance.png")
