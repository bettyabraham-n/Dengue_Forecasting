import numpy as np
import pandas as pd
from epiweeks import Week


# Constants
REGIONS = {
    'Norte':        ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO'],
    'Nordeste':     ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
    'Centro-Oeste': ['DF', 'GO', 'MS', 'MT'],
    'Sudeste':      ['ES', 'MG', 'RJ', 'SP'],
    'Sul':          ['PR', 'RS', 'SC'],
}

state_order = sorted([s for states in REGIONS.values() for s in states])

TARGET   = 'casos'
H        = 52       # forecast horizon (weeks)
VAL_CHECK  = 50     # val check steps for NeuralForecast
EARLY_STOP = 30     # early stop patience_steps


# Epiweek helpers
def ew_to_date(epiweek_int):
    """Convert integer epiweek (YYYYWW) to a pandas Timestamp."""
    s = str(int(epiweek_int))
    return pd.Timestamp(Week(int(s[:4]), int(s[4:])).startdate())


def gen_forecast_dates(year):
    """
    Generate the 52 weekly dates for a dengue season
    """
    dates = []
    for y, week in zip(
        np.concatenate((np.repeat(year - 1, 12), np.repeat(year, 40))),
        np.concatenate((np.arange(41, 53), np.arange(1, 41)))
    ):
        dates.append(pd.Timestamp(Week(y, int(week)).startdate()))
    return dates


# Data preparation
def prepare_global_nf(
    df_hist,
    df_fcst,
    active_states,
    season,
    features,
    forecast_season,
    input_size=89,
):
    
    year         = int(season[5:])
    is_forecast  = (season == forecast_season)
    season_dates = gen_forecast_dates(year)

    train_rows = []
    cov_map    = {}

    for state in active_states:
        df_s  = (df_hist[df_hist['uf'] == state]
                 .sort_values('epiweek')
                 .reset_index(drop=True)
                 .copy())
        df_tr = df_s[df_s['ew_year'] < year].copy()

        if len(df_tr) < input_size + H:
            print(f'  [{state}] insufficient history — skipped')
            continue

        df_tr['ds']        = df_tr['epiweek'].apply(ew_to_date)
        df_tr['unique_id'] = state
        df_tr['y']         = np.log1p(df_tr[TARGET].clip(lower=0))
        train_rows.append(df_tr[['unique_id', 'ds', 'y'] + features])

        # Future covariates — positional, one row per forecast week
        if is_forecast:
            df_fc    = (df_fcst[df_fcst['uf'] == state]
                        .sort_values('epiweek')
                        .reset_index(drop=True)
                        .copy())
            cov_vals = df_fc[features].head(H).reset_index(drop=True)
        else:
            df_te    = df_s[df_s['ew_year'] == year].copy()
            cov_vals = df_te[features].head(H).reset_index(drop=True)

        # Pad if short
        if len(cov_vals) < H:
            last_row = cov_vals.iloc[[-1]]
            pad      = pd.concat([last_row] * (H - len(cov_vals)), ignore_index=True)
            cov_vals = pd.concat([cov_vals, pad], ignore_index=True)

        cov_map[state] = cov_vals.reset_index(drop=True)

    df_train = pd.concat(train_rows, ignore_index=True)
    print(f'  Train rows: {len(df_train):,}  |  states: {df_train["unique_id"].nunique()}')
    return df_train, cov_map, season_dates


def nhits_preds_to_df(fcst_df, state, season_dates):
    """
    Convert NeuralForecast log1p-space output to raw case counts.
    """
    if 'unique_id' not in fcst_df.columns:
        fcst_df = fcst_df.reset_index()
    df_s = fcst_df[fcst_df['unique_id'] == state].reset_index(drop=True)
    if len(df_s) == 0:
        return None

    lo_cols = ['NHITS-lo-95', 'NHITS-lo-90', 'NHITS-lo-80', 'NHITS-lo-50']
    hi_cols = ['NHITS-hi-50', 'NHITS-hi-80', 'NHITS-hi-90', 'NHITS-hi-95']

    # Handle both integer (50) and float (50.0) column suffixes
    def resolve_cols(cols):
        resolved = []
        for c in cols:
            if c in df_s.columns:
                resolved.append(c)
            else:
                for alt in [c + '.0',
                            c.replace('-50', '-50.0')
                             .replace('-80', '-80.0')
                             .replace('-90', '-90.0')
                             .replace('-95', '-95.0')]:
                    if alt in df_s.columns:
                        resolved.append(alt)
                        break
                else:
                    raise KeyError(f'{c} not found in {list(df_s.columns)}')
        return resolved

    lo_cols   = resolve_cols(lo_cols)
    hi_cols   = resolve_cols(hi_cols)
    lo_sorted = np.sort(df_s[lo_cols].values, axis=1)
    hi_sorted = np.sort(df_s[hi_cols].values, axis=1)

    n = len(df_s)
    return pd.DataFrame({
        'pred'    : np.expm1(df_s['NHITS-median'].clip(lower=0).values).clip(min=0),
        'lower_95': np.expm1(lo_sorted[:, 0]).clip(min=0),
        'lower_90': np.expm1(lo_sorted[:, 1]).clip(min=0),
        'lower_80': np.expm1(lo_sorted[:, 2]).clip(min=0),
        'lower_50': np.expm1(lo_sorted[:, 3]).clip(min=0),
        'upper_50': np.expm1(hi_sorted[:, 0]).clip(min=0),
        'upper_80': np.expm1(hi_sorted[:, 1]).clip(min=0),
        'upper_90': np.expm1(hi_sorted[:, 2]).clip(min=0),
        'upper_95': np.expm1(hi_sorted[:, 3]).clip(min=0),
        'date'    : pd.to_datetime(season_dates[:n]),
    })
