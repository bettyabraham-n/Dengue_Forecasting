import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from utils import TARGET

PI_COLORS = {
    '95': '#bdd7e7',
    '90': '#6baed6',
    '80': '#3182bd',
    '50': '#08519c',
}
EW_SEQ = list(range(41, 53)) + list(range(1, 41))

EW_TO_MONTH = {
    41:"Oct", 42:"Oct", 43:"Oct", 44:"Oct",
    45:"Nov", 46:"Nov", 47:"Nov", 48:"Nov",
    49:"Dec", 50:"Dec", 51:"Dec", 52:"Dec",
     1:"Jan",  2:"Jan",  3:"Jan",  4:"Jan",
     5:"Feb",  6:"Feb",  7:"Feb",  8:"Feb",
     9:"Mar", 10:"Mar", 11:"Mar", 12:"Mar",
    13:"Apr", 14:"Apr", 15:"Apr", 16:"Apr",
    17:"May", 18:"May", 19:"May", 20:"May",
    21:"Jun", 22:"Jun", 23:"Jun", 24:"Jun",
    25:"Jul", 26:"Jul", 27:"Jul", 28:"Jul",
    29:"Aug", 30:"Aug", 31:"Aug", 32:"Aug",
    33:"Sep", 34:"Sep", 35:"Sep", 36:"Sep",
    37:"Sep", 38:"Sep", 39:"Sep", 40:"Sep",
}

def plot_season(
    season,
    all_preds,
    df_hist,
    df_scores,
    state_order,
    is_forecast=False,
    save_path=None,
):
    
    ncols, nrows = 6, 5
    fig, axes    = plt.subplots(nrows, ncols, figsize=(24, 20))
    axes_flat    = axes.flatten()
    year         = int(season[5:])

    label = 'Forecast' if is_forecast else 'Validation'
    fig.suptitle(f'{label} — {season}  [N-HiTS]',
                 fontsize=18, fontweight='bold', y=1.05)

    for idx, state in enumerate(state_order):
        ax  = axes_flat[idx]
        key = (state, season)
        if key not in all_preds:
            ax.axis('off')
            continue

        df_p   = all_preds[key].reset_index(drop=True)
        x_vals = np.arange(len(df_p))

        for lvl in ['95', '90', '80', '50']:
            ax.fill_between(
                x_vals, df_p[f'lower_{lvl}'], df_p[f'upper_{lvl}'],
                color=PI_COLORS[lvl], alpha=0.9, linewidth=0,
            )
        ax.plot(x_vals, df_p['pred'], color='red', linewidth=1.4)

        if not is_forecast:
            df_a = (df_hist[(df_hist['uf'] == state) & (df_hist['ew_year'] == year)]
                    .sort_values('epiweek')
                    .reset_index(drop=True))
            n = min(len(df_a), len(df_p))
            ax.plot(
                x_vals[:n], df_a[TARGET].clip(lower=0).values[:n],
                color='black', linewidth=1.2, linestyle='--',
            )

        if not is_forecast and df_scores is not None:
            row = df_scores[
                (df_scores['state'] == state) & (df_scores['season'] == season)
            ]
            if len(row):
                r = row.iloc[0]
                ax.set_title(
                    f"{state}\n"
                    f"WIS={r['WIS']:.0f} | CRPS={r['CRPS']:.0f} | LogS={r['LogS']:.2f}\n"
                    f"MAE={r['MAE']:.0f} | RMSE={r['RMSE']:.0f}",
                    fontsize=12, fontweight='bold', pad=3,
                )
            else:
                ax.set_title(state, fontsize=14, fontweight='bold')
        else:
            peak = df_p['pred'].max()
            hw   = (df_p.loc[df_p['pred'].idxmax(), 'upper_95'] -
                    df_p.loc[df_p['pred'].idxmax(), 'lower_95']) / 2
            ax.set_title(f"{state}\nPeak={peak:.0f} | +/-{hw:.0f}",
                         fontsize=14, fontweight='bold', pad=6)

        tick_idx = list(range(0, len(x_vals), 4))

        # Show only the first EW of each month to avoid label crowding
        month_ticks = []
        month_labels = []
        seen = set()
        for i in range(0, len(x), step):
            ew  = int(ew_nums[i])
            mon = EW_TO_MONTH.get(ew, "")
            if mon not in seen:
                month_ticks.append(x[i])
                month_labels.append(mon)
                seen.add(mon)

        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels, fontsize=9, rotation=45, ha="right")
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(-0.5, len(x_vals) - 0.5)

    for idx in range(len(state_order), len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.supylabel('Dengue Cases', fontsize=14)
    fig.text(0.5, -0.02, 'Month', ha='center', fontsize=14)

    patches = [
        mpatches.Patch(color=PI_COLORS['95'], label='95% PI'),
        mpatches.Patch(color=PI_COLORS['90'], label='90% PI'),
        mpatches.Patch(color=PI_COLORS['80'], label='80% PI'),
        mpatches.Patch(color=PI_COLORS['50'], label='50% PI'),
        plt.Line2D([0], [0], color='red',   lw=1.4, label='Forecast (median)'),
    ]
    if not is_forecast:
        patches.append(
            plt.Line2D([0], [0], color='black', lw=1.2, ls='--', label='Actual')
        )

    fig.legend(
        handles=patches, loc='lower center', ncol=len(patches),
        fontsize=12, bbox_to_anchor=(0.5, -0.04),
        frameon=True, fancybox=True,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.show()
    plt.close()


def plot_wis_heatmap(df_scores, state_order, save_path=None):
    heatmap_data = (
        df_scores
        .pivot(index='state', columns='season', values='WIS')
        .reindex(state_order)
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        heatmap_data, ax=ax, cmap='YlOrRd',
        annot=True, fmt='.0f', annot_kws={'size': 8},
        linewidths=0.4, linecolor='white',
        cbar_kws={'label': 'WIS (lower = better)', 'shrink': 0.6},
    )
    ax.set_title('N-HiTS WIS (State & Season)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Season', fontsize=10)
    ax.set_ylabel('State',  fontsize=10)
    ax.tick_params(axis='x', rotation=15)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    plt.show()
    plt.close()
