"""
loo/loo.py
----------
Leave-One-Covariate-Out (LOO) selection for the global Bayesian model.

Method
------
Zero out one covariate at a time (keeps tensor shape constant, so JAX does
not recompile between refits). For each dropped covariate compute delta-ELPD
vs the full model. A covariate is selected if delta_ELPD < -1 SE (i.e. it
meaningfully improves out-of-sample accuracy). More negative = more important.

Structural covariates (sin_ew, cos_ew) are always kept regardless of ELPD.

Runs sequentially to avoid joblib/pickle errors with JAX/PyMC objects.
Each refit still gets full GPU acceleration via NumPyro.

Usage
-----
Run from LOO.ipynb or directly:

    from loo.loo import run_loo
    loo_results, selected, loo_df = run_loo(df, config)
"""

import json
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Adjust path so imports work when called from the loo/ subdirectory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils   import Standardiser, prepare_global_data
from model   import bayesian_negbin_global


# ---------------------------------------------------------------------------
# Single LOO refit
# ---------------------------------------------------------------------------

def _loo_single_refit(
    drop_cov:      str,
    avail_loo:     list,
    X_loo_raw:     np.ndarray,
    y_loo:         np.ndarray,
    lp_loo:        np.ndarray,
    sidx_loo:      np.ndarray,
    elpd_full:     float,
    loo_full_i:    np.ndarray,
    meta_loo:      pd.DataFrame,
    active_states: list,
    n_active:      int,
    loo_draws:     int,
    loo_tune:      int,
    loo_chains:    int,
    target_accept: float,
    seed:          int,
) -> dict:
    """Fit the model with drop_cov zeroed out and return delta-ELPD stats."""
    # Zero out the covariate — keeps shape constant, no JAX recompilation
    cov_idx   = avail_loo.index(drop_cov)
    X_red_raw = X_loo_raw.copy()
    X_red_raw[:, cov_idx] = 0.0

    sc    = Standardiser()
    X_red = sc.fit_transform(X_red_raw)

    try:
        idata_red, _, _ = bayesian_negbin_global(
            y_loo, X_red, lp_loo, sidx_loo,
            avail_loo, n_active,
            draws=loo_draws, tune=loo_tune, chains=loo_chains,
            target_accept=target_accept, seed=seed,
        )
        loo_red  = az.loo(idata_red, pointwise=True)
        elpd_red = float(loo_red.elpd_loo)
        delta    = elpd_red - elpd_full
        pw_diff  = loo_red.loo_i.values - loo_full_i
        se_delta = float(np.sqrt(len(pw_diff) * np.var(pw_diff)))

        state_deltas = {}
        for uf in active_states:
            mask = (meta_loo["uf"] == uf).values
            if mask.sum() > 0:
                state_deltas[uf] = round(
                    float((loo_red.loo_i.values[mask] - loo_full_i[mask]).sum()), 2
                )

        return {
            "covariate"   : drop_cov,
            "elpd_full"   : round(elpd_full,  2),
            "elpd_reduced": round(elpd_red,   2),
            "delta_elpd"  : round(delta,      2),
            "se_delta"    : round(se_delta,   2),
            "important"   : delta < -se_delta,
            "state_deltas": state_deltas,
            "error"       : None,
        }
    except Exception as e:
        return {
            "covariate"   : drop_cov,
            "elpd_full"   : elpd_full,
            "elpd_reduced": None,
            "delta_elpd"  : None,
            "se_delta"    : None,
            "important"   : False,
            "state_deltas": {},
            "error"       : str(e),
        }


# ---------------------------------------------------------------------------
# Main LOO entry point
# ---------------------------------------------------------------------------

def run_loo(
    df:                  pd.DataFrame,
    active_states:       list,
    candidate_covs:      list,
    structural_covs:     list,
    forecast_season:     str,
    loo_draws:           int   = 500,
    loo_tune:            int   = 500,
    loo_chains:          int   = 4,
    target_accept:       float = 0.90,
    seed:                int   = 42,
    output_path:         str   = "outputs/scores/selected_covariates.json",
) -> tuple:
    """
    Run the full LOO covariate selection pipeline.

    Steps
    -----
    1. Fit full model on all training data.
    2. For each non-structural covariate, refit with it zeroed out.
    3. Select covariates where delta_ELPD < -1 SE.
    4. Save selected covariate list to output_path.

    Parameters
    ----------
    df                : historical dataset (with season column)
    active_states     : list of state codes to include
    candidate_covs    : all covariates to consider (including structural)
    structural_covs   : always-kept covariates (e.g. sin_ew, cos_ew)
    forecast_season   : excluded from LOO training data
    loo_draws/tune/chains/target_accept/seed : MCMC settings
    output_path       : where to save selected_covariates.json

    Returns
    -------
    loo_results  : list of raw result dicts (including state_deltas)
    selected     : list of selected covariate names
    loo_df       : summary DataFrame (without state_deltas)
    """
    n_active = len(active_states)

    df_loo = df[
        (df["uf"].isin(active_states)) &
        (df["season"] != forecast_season)
    ].copy()

    y_loo, X_loo_raw, lp_loo, sidx_loo, meta_loo, avail_loo, _ = prepare_global_data(
        df_loo, candidate_covs, active_states
    )

    print(f"LOO training: {len(y_loo):,} rows × {len(avail_loo)} covariates")
    print(f"Seasons: {sorted(df_loo['season'].unique())}")

    # Step 1 — full model
    print("\n" + "="*55)
    print("LOO Step 1: Fitting full model...")
    print("="*55)

    scaler_loo = Standardiser()
    X_loo      = scaler_loo.fit_transform(X_loo_raw)

    idata_full, _, _ = bayesian_negbin_global(
        y_loo, X_loo, lp_loo, sidx_loo,
        avail_loo, n_active,
        draws=loo_draws, tune=loo_tune, chains=loo_chains,
        target_accept=target_accept, seed=seed,
    )

    loo_full  = az.loo(idata_full, pointwise=True)
    elpd_full = float(loo_full.elpd_loo)
    print(f"Full model ELPD: {elpd_full:.2f}")

    conv = az.summary(idata_full, var_names=["intercept", "alpha"], stat_focus="mean")
    rhat = float(conv["r_hat"].dropna().max()) if len(conv["r_hat"].dropna()) > 0 else float("nan")
    ess  = float(conv["ess_bulk"].min())
    print(f"R-hat={rhat:.3f}  ESS={ess:.0f}  {'WARNING' if rhat > 1.05 else 'OK'}")

    # Step 2 — sequential refits
    covs_to_test = [c for c in avail_loo if c not in structural_covs]
    print(f"\nRunning {len(covs_to_test)} LOO refits sequentially...")

    loo_raw = []
    for k, drop_cov in enumerate(covs_to_test):
        print(f"  [{k+1}/{len(covs_to_test)}] Dropping '{drop_cov}'...")
        result = _loo_single_refit(
            drop_cov, avail_loo, X_loo_raw, y_loo,
            lp_loo, sidx_loo, elpd_full, loo_full.loo_i.values,
            meta_loo, active_states, n_active,
            loo_draws, loo_tune, loo_chains, target_accept, seed,
        )
        if result["error"]:
            print(f"    ERROR: {result['error']}")
        else:
            print(
                f"    delta_ELPD={result['delta_elpd']:.2f}  "
                f"SE={result['se_delta']:.2f}  "
                f"important={result['important']}"
            )
        loo_raw.append(result)

    # Step 3 — select covariates
    failed      = [r for r in loo_raw if r["error"] is not None]
    loo_results = [r for r in loo_raw if r["error"] is None]

    if failed:
        print(f"\nWARNING: {len(failed)} LOO refits failed:")
        for r in failed:
            print(f"  {r['covariate']}: {r['error']}")

    loo_df_rows = [
        {k: v for k, v in r.items() if k != "state_deltas"}
        for r in loo_results
    ]
    struct_rows = [
        {
            "covariate": c, "elpd_full": elpd_full, "elpd_reduced": None,
            "delta_elpd": None, "se_delta": None, "important": True, "error": None,
        }
        for c in structural_covs
    ]
    loo_df = (
        pd.DataFrame(loo_df_rows + struct_rows)
        .sort_values("delta_elpd")
        .reset_index(drop=True)
    )

    selected = structural_covs + [
        r["covariate"] for r in loo_results if r["important"]
    ]

    print("\n" + "="*55)
    print("LOO COVARIATE SELECTION")
    print("="*55)
    print(
        loo_df[["covariate", "delta_elpd", "se_delta", "important"]]
        .sort_values("delta_elpd")
        .to_string(index=False)
    )
    print(f"\nSelected ({len(selected)}): {selected}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"selected": selected, "elpd_full": elpd_full}, f, indent=2)
    print(f"\nSaved: {output_path}")

    return loo_results, selected, loo_df
