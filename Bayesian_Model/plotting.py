import math

import arviz as az
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Epidemiological week order: EW41 - EW52 - EW01 - EW40
EW_SEQ = list(range(41, 53)) + list(range(1, 41))

PI_COLORS = {
    "95": "#bdd7e7",
    "90": "#6baed6",
    "80": "#3182bd",
    "50": "#08519c",
}


# Posterior interval extraction
def extract_pi(samples: np.ndarray) -> pd.DataFrame:
    qs   = [0.025, 0.050, 0.100, 0.250, 0.500, 0.750, 0.900, 0.950, 0.975]
    cols = [
        "lower_95", "lower_90", "lower_80", "lower_50",
        "median",
        "upper_50", "upper_80", "upper_90", "upper_95",
    ]
    return pd.DataFrame(
        dict(zip(cols, [np.quantile(samples, q, axis=0) for q in qs]))
    )


# Forecast / validation grid
def plot_forecast_grid(
    results_list: list,
    season: str,
    mode: str = "validation",
    save_path: str = None,
) -> None:
    results_list = sorted(results_list, key=lambda r: r["uf"])
    n     = len(results_list)
    ncols = min(6, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.5, nrows * 3.0),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, res in enumerate(results_list):
        ax     = axes_flat[idx]
        uf     = res["uf"]
        df_s   = res["df_season"]
        pi_df  = res["pi_df"].reset_index(drop=True)
        scores = res.get("scores", {})

        # Sort epiweeks
        if "ew_num" in pi_df.columns:
            pi_df["_sort"] = pi_df["ew_num"].apply(
                lambda w: w - 41 if w >= 41 else w + 11
            )
            pi_df = pi_df.sort_values("_sort").reset_index(drop=True)

        x = np.arange(len(pi_df))

        for lvl in ["95", "90", "80", "50"]:
            ax.fill_between(
                x, pi_df[f"lower_{lvl}"], pi_df[f"upper_{lvl}"],
                color=PI_COLORS[lvl], alpha=0.85, linewidth=0,
            )
        ax.plot(x, pi_df["median"], color="#e50f0f", lw=1.6, zorder=5)

        actual = None
        if mode == "validation" and "casos" in df_s.columns:
            if "ew_num" in df_s.columns:
                df_s = df_s.copy()
                df_s["_sort"] = df_s["ew_num"].apply(
                    lambda w: w - 41 if w >= 41 else w + 11
                )
                df_s = df_s.sort_values("_sort").reset_index(drop=True)
            actual  = df_s["casos"].clip(lower=0).values
            n_plot  = min(len(actual), len(x))
            ax.plot(
                x[:n_plot], actual[:n_plot],
                color="black", lw=1.3, ls="--", zorder=6,
            )

        # Per-state y-axis
        y_max_vals = [pi_df["upper_95"].max()]
        if actual is not None:
            y_max_vals.append(actual.max())
        ax.set_ylim(0, max(y_max_vals) * 1.05)

        ew_nums = (
            pi_df["ew_num"].values if "ew_num" in pi_df.columns
            else [EW_SEQ[i % 52] for i in range(len(pi_df))]
        )
        step = 2 if len(x) <= 26 else 4
        ax.set_xticks(x[::step])
        ax.set_xticklabels(
            [f"EW{int(ew_nums[i]):02d}" for i in range(0, len(x), step)],
            fontsize=6.5, rotation=45, ha="right",
        )
        ax.tick_params(axis="y", labelsize=7)

        if mode == "validation" and scores:
            title = (
                f"{uf}\n"
                f"WIS={scores.get('WIS', 0):.0f} | "
                f"CRPS={scores.get('CRPS', 0):.0f} | "
                f"LogS={scores.get('LogS', 0):.2f}\n"
                f"MAE={scores.get('MAE', 0):.0f} | "
                f"RMSE={scores.get('RMSE', 0):.0f}"
            )
        else:
            title = f"{uf}\nPeak={pi_df['median'].max():.0f}"

        ax.set_title(title, fontsize=8, fontweight="bold", pad=6)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    label = "Validation" if mode == "validation" else "Forecast"
    fig.suptitle(
        f"{label} — {season} [Bayesian]",
        fontsize=12, fontweight="bold", y=1.05,
    )
    fig.supylabel("Dengue Cases", fontsize=9)
    fig.supxlabel("Epidemiological Week", fontsize=9)

    handles = [
        mpatches.Patch(color="#bdd7e7", label="95% PI"),
        mpatches.Patch(color="#6baed6", label="90% PI"),
        mpatches.Patch(color="#3182bd", label="80% PI"),
        mpatches.Patch(color="#08519c", label="50% PI"),
        plt.Line2D([0], [0], color="#e50f0f", lw=1.6, label="Forecast (median)"),
        plt.Line2D([0], [0], color="black",   lw=1.3, ls="--", label="Actual"),
    ]
    fig.legend(
        handles=handles, loc="lower center", fontsize=8, ncol=6,
        bbox_to_anchor=(0.5, -0.06), framealpha=0.9,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# WIS heatmap
def plot_scores_heatmap(scores_df: pd.DataFrame, save_path: str = None) -> None:
    pivot = scores_df.pivot_table(index="uf", columns="season", values="WIS")
    fig, ax = plt.subplots(
        figsize=(max(7, len(pivot.columns) * 2.5), max(5, len(pivot) * 0.5))
    )
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "WIS (lower = better)"},
    )
    ax.set_title(
        "Bayesian Validation WIS — State & Season",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# LOO importance heatmap
def plot_loo_heatmap(loo_results: list, save_path: str = None) -> None:
    rows = []
    for r in loo_results:
        for uf, delta in r["state_deltas"].items():
            rows.append({"covariate": r["covariate"], "uf": uf, "delta_elpd": delta})

    if not rows:
        print("No LOO results to plot.")
        return

    hm_df = pd.DataFrame(rows)
    pivot = hm_df.pivot_table(index="covariate", columns="uf", values="delta_elpd")
    pivot["MEAN"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("MEAN")

    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.7), max(5, len(pivot) * 0.55))
    )
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, linewidths=0.4, ax=ax,
        cbar_kws={"label": "delta_ELPD (more negative = more important)"},
    )
    ax.set_title(
        "LOO Covariate Importance by State\n"
        "More negative = covariate helps | MEAN = average across states",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close()
