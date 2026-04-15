import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

# ── Shared dark theme for all plots ──────────────────────────────────────────
_BG    = "#0f1117"
_PANEL = "#1a1d27"
_TEXT  = "#e8e8f0"
_ACC   = "#ff6b35"

def _apply_dark(fig, ax):
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e3347")


def plot_correlation(df: pd.DataFrame):
    """Correlation heatmap on numeric columns only."""
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        raise ValueError("No numeric columns found for correlation.")

    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_dark(fig, ax)

    corr = num_df.corr()
    mask = None  # show full matrix

    cmap = sns.diverging_palette(220, 15, as_cmap=True)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=cmap,
        linewidths=0.5, linecolor="#2e3347",
        annot_kws={"size": 8, "color": _TEXT},
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, pad=12)
    plt.tight_layout()
    return fig


def plot_distribution(df: pd.DataFrame, column: str):
    """Histogram + KDE for a single column."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_dark(fig, ax)

    sns.histplot(
        df[column].dropna(), kde=True,
        color=_ACC, edgecolor="#2e3347",
        line_kws={"linewidth": 2},
        ax=ax
    )
    ax.set_title(f"Distribution of  {column}", fontsize=12)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_fire_risk_gauge(probability: float):
    """
    Donut-style gauge showing fire probability (0–1).
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    filled = probability
    empty  = 1 - probability

    color  = "#ff3b3b" if probability > 0.6 else ("#ffa500" if probability > 0.35 else "#22c55e")
    wedge_colors = [color, "#2e3347"]

    wedges, _ = ax.pie(
        [filled, empty],
        colors=wedge_colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor=_BG, linewidth=2)
    )

    ax.text(0, 0,  f"{probability*100:.0f}%", ha="center", va="center",
            fontsize=24, fontweight="bold", color=_TEXT)
    ax.text(0, -0.25, "Fire Risk", ha="center", va="center",
            fontsize=10, color="#9999bb")
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names: list, coef_values):
    """
    Horizontal bar chart of model coefficients / importances.
    Works for linear models (coef_) and tree models (feature_importances_).
    """
    import numpy as np
    vals = np.array(coef_values).flatten()
    # show top 10 by absolute magnitude
    idx  = np.argsort(np.abs(vals))[-10:]
    names = [feature_names[i] for i in idx]
    vals  = vals[idx]

    colors = [_ACC if v >= 0 else "#4f8ff7" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    _apply_dark(fig, ax)
    ax.barh(names, vals, color=colors, edgecolor=_BG, height=0.6)
    ax.axvline(0, color="#555577", linewidth=1)
    ax.set_title("Top Feature Weights", fontsize=12)
    ax.set_xlabel("Coefficient / Importance")
    plt.tight_layout()
    return fig
