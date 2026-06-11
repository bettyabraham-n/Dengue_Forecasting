import numpy as np
import pandas as pd
from epiweeks import Week



# Region-state mappings
Region_states = {
    "North"        : ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "Northeast"    : ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Central-West" : ["DF", "GO", "MS", "MT"],
    "Southeast"    : ["ES", "MG", "RJ", "SP"],
    "South"        : ["PR", "RS", "SC"],
}

state2region = {s: r for r, states in Region_states.items() for s in states}



# Standardiser
class Standardiser:
    """Z-score standardiser fitted only on training data to prevent leakage."""

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


# Season labels
def add_season_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag every row with its dengue season (EW41 -> EW40).
    Seasons run EW41 of year Y to EW40 of year Y+1.
    """
    df = df.copy()
    df["ew_year"] = df["epiweek"] // 100
    df["ew_num"]  = df["epiweek"] %  100
    df["season"]  = df.apply(
        lambda r: f"{r['ew_year']}-{r['ew_year']+1}" if r["ew_num"] >= 41
                  else f"{r['ew_year']-1}-{r['ew_year']}",
        axis=1,
    )
    return df


# Epiweek helpers
def epiweek_to_date(ew: int):
    """Convert integer epiweek (YYYYWW) to a date object. Returns None on error."""
    try:
        return Week(int(str(ew)[:4]), int(str(ew)[4:])).startdate()
    except Exception:
        return None


def pad_forecast_climate(df_fc: pd.DataFrame, target_weeks: int = 52) -> pd.DataFrame:
    """Pad a forecast climate dataframe to target_weeks rows by repeating the last row."""
    if len(df_fc) >= target_weeks:
        return df_fc
    last = df_fc.iloc[-1]
    last_year, last_week = int(last["epiweek"]) // 100, int(last["epiweek"]) % 100
    extra = []
    for i in range(1, target_weeks - len(df_fc) + 1):
        nw, ny = last_week + i, last_year
        while nw > 52:
            nw -= 52
            ny += 1
        r = last.to_dict()
        r["epiweek"] = int(ny * 100 + nw)
        r["ew_num"]  = int(nw)
        r["ew_year"] = int(ny)
        r["sin_ew"]  = float(np.sin(2 * np.pi * nw / 52))
        r["cos_ew"]  = float(np.cos(2 * np.pi * nw / 52))
        extra.append(r)
    return pd.concat([df_fc, pd.DataFrame(extra)], ignore_index=True)


# Data preparation
def prepare_global_data(
    df_subset: pd.DataFrame,
    covariates: list,
    active_states: list,
) -> tuple:
    
    active_state_to_idx = {s: i for i, s in enumerate(active_states)}
    parts = []
    for uf in active_states:
        df_uf = df_subset[df_subset["uf"] == uf].copy().sort_values("epiweek")
        if len(df_uf) == 0:
            continue
        df_uf["_state_idx"] = active_state_to_idx[uf]
        parts.append(df_uf)

    df_pooled = pd.concat(parts, axis=0).reset_index(drop=True)
    avail     = [c for c in covariates if c in df_pooled.columns]

    y         = df_pooled["casos"].fillna(0).values.astype(float)
    X_raw     = df_pooled[avail].fillna(0).values.astype(float)
    log_pop   = np.log(df_pooled["population"].fillna(1).values.astype(float))
    state_idx = df_pooled["_state_idx"].values.astype(int)
    row_meta  = df_pooled[["uf", "epiweek", "ew_num", "season"]].copy()

    assert state_idx.max() < len(active_states), (
        f"state_idx max={state_idx.max()} >= n_active={len(active_states)}"
    )

    return y, X_raw, log_pop, state_idx, row_meta, avail, active_state_to_idx


def extract_state_samples(idata_pred, row_meta: pd.DataFrame, uf: str) -> np.ndarray:
    """Extract posterior predictive samples for a single state."""
    mask = (row_meta["uf"] == uf).values
    obs  = idata_pred.posterior_predictive["obs"].values
    flat = obs.reshape(-1, obs.shape[-1])
    return flat[:, mask]
