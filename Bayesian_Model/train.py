import numpy as np
import pandas as pd
import arviz as az

from utils   import Standardiser, prepare_global_data, extract_state_samples, pad_forecast_climate
from model   import bayesian_negbin_global
from scoring import compute_all_scores
from plotting import extract_pi


# Phase 1 — Validation
def run_validation(
    df:               pd.DataFrame,
    val_seasons:      list,
    active_states:    list,
    selected_covs:    list,
    draws:            int   = 1000,
    tune:             int   = 1000,
    chains:           int   = 4,
    target_accept:    float = 0.90,
    seed:             int   = 42,
) -> tuple:
    
    all_scores  = []
    val_results = {s: [] for s in val_seasons}
    n_active    = len(active_states)

    for season in val_seasons:
        print(f"\n{'='*55}")
        print(f"VALIDATION: {season}")
        print(f"{'='*55}")

        df_train = df[
            (df["uf"].isin(active_states)) & (df["season"] < season)
        ].copy()
        df_val = df[
            (df["uf"].isin(active_states)) & (df["season"] == season)
        ].copy()

        if len(df_train) < 100 or len(df_val) < 10:
            print(f"  Skip — train={len(df_train)} val={len(df_val)}")
            continue

        y_tr, X_tr_raw, lp_tr, sidx_tr, _, avail_tr, _ = prepare_global_data(
            df_train, selected_covs, active_states
        )
        y_val, X_val_raw, lp_val, sidx_val, meta_val, _, _ = prepare_global_data(
            df_val, selected_covs, active_states
        )

        scaler = Standardiser()
        X_tr   = scaler.fit_transform(X_tr_raw)
        X_val  = scaler.transform(X_val_raw)

        try:
            idata_val, idata_pred, _ = bayesian_negbin_global(
                y_tr, X_tr, lp_tr, sidx_tr, avail_tr, n_active,
                X_pred=X_val, log_pop_pred=lp_val, state_idx_pred=sidx_val,
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, seed=seed,
            )
        except Exception as e:
            print(f"  Model failed: {e}")
            continue

        # Convergence summary
        conv = az.summary(idata_val, var_names=["intercept", "alpha"], stat_focus="mean")
        rhat = float(conv["r_hat"].dropna().max()) if len(conv["r_hat"].dropna()) > 0 else float("nan")
        ess  = float(conv["ess_bulk"].min())
        print(f"  R-hat={rhat:.3f}  ESS={ess:.0f}  {'WARNING' if rhat > 1.05 else 'OK'}")

        for uf in active_states:
            samples_uf = extract_state_samples(idata_pred, meta_val, uf)
            if samples_uf.shape[1] == 0:
                continue
            pi_df  = extract_pi(samples_uf)
            y_uf   = y_val[(meta_val["uf"] == uf).values]
            scores = compute_all_scores(y_uf, pi_df, samples_uf)
            scores.update({"uf": uf, "season": season})
            all_scores.append(scores)

            df_uf_val = (
                df_val[df_val["uf"] == uf]
                .sort_values("epiweek")
                .reset_index(drop=True)
            )
            val_results[season].append({
                "uf":        uf,
                "df_season": df_uf_val,
                "pi_df":     pi_df,
                "scores":    scores,
            })
            print(
                f"  {uf}: WIS={scores['WIS']}  CRPS={scores['CRPS']}  "
                f"LogS={scores['LogS']}  MAE={scores['MAE']}  RMSE={scores['RMSE']}"
            )

    print("\nValidation complete.")
    return all_scores, val_results


# Phase 2 — Forecast 
def run_forecast(
    df:               pd.DataFrame,
    df_fcst_clim:     pd.DataFrame,
    forecast_season:  str,
    active_states:    list,
    selected_covs:    list,
    draws:            int   = 1000,
    tune:             int   = 1000,
    chains:           int   = 4,
    target_accept:    float = 0.90,
    seed:             int   = 42,
) -> tuple:

    print(f"\n{'='*55}")
    print(f"FORECAST: {forecast_season}")
    print(f"{'='*55}")

    n_active = len(active_states)

    df_fc_train = df[
        (df["uf"].isin(active_states)) & (df["season"] != forecast_season)
    ].copy()

    y_fc, X_fc_raw, lp_fc, sidx_fc, _, avail_fc, _ = prepare_global_data(
        df_fc_train, selected_covs, active_states
    )

    # Build pooled forecast climate arrays
    local_idx = {s: i for i, s in enumerate(active_states)}
    fc_parts  = []
    for uf in active_states:
        df_uf_fc = df_fcst_clim[df_fcst_clim["uf"] == uf].copy()
        if len(df_uf_fc) == 0:
            print(f"  WARNING: no forecast climate for {uf}")
            continue
        df_uf_fc = pad_forecast_climate(
            df_uf_fc.sort_values("epiweek").reset_index(drop=True)
        )
        for col in selected_covs:
            if col not in df_uf_fc.columns:
                df_uf_fc[col] = 0.0
        df_uf_fc["_state_idx"] = local_idx[uf]
        fc_parts.append(df_uf_fc)

    df_fc_pooled = pd.concat(fc_parts, axis=0).reset_index(drop=True)
    X_pred_raw   = df_fc_pooled[selected_covs].fillna(0).values.astype(float)
    lp_pred      = np.log(df_fc_pooled["population"].fillna(1).values.astype(float))
    sidx_pred    = df_fc_pooled["_state_idx"].values.astype(int)
    meta_pred    = df_fc_pooled[["uf", "epiweek", "ew_num"]].copy()

    scaler_fc = Standardiser()
    X_fc      = scaler_fc.fit_transform(X_fc_raw)
    X_pred    = scaler_fc.transform(X_pred_raw)

    print(f"Training rows : {len(y_fc):,}")
    print(f"Forecast rows : {len(X_pred):,}")

    try:
        _, idata_forecast, _ = bayesian_negbin_global(
            y_fc, X_fc, lp_fc, sidx_fc, avail_fc, n_active,
            X_pred=X_pred, log_pop_pred=lp_pred, state_idx_pred=sidx_pred,
            draws=draws, tune=tune, chains=chains,
            target_accept=target_accept, seed=seed,
        )
        print("Forecast sampling complete.")
    except Exception as e:
        print(f"Forecast failed: {e}")
        return [], []

    all_forecasts    = []
    forecast_results = []

    for uf in active_states:
        samples_uf = extract_state_samples(idata_forecast, meta_pred, uf)
        if samples_uf.shape[1] == 0:
            continue
        pi_fc = extract_pi(samples_uf)

        df_uf_fc = df_fc_pooled[df_fc_pooled["uf"] == uf].reset_index(drop=True)
        pi_fc["uf"]      = uf
        pi_fc["season"]  = forecast_season
        pi_fc["ew_num"]  = df_uf_fc["ew_num"].values
        pi_fc["epiweek"] = df_uf_fc["epiweek"].values

        all_forecasts.append(pi_fc)
        forecast_results.append({
            "uf":        uf,
            "df_season": df_uf_fc,
            "pi_df":     pi_fc,
            "scores":    {},
        })
        print(f"  {uf}: peak median = {pi_fc['median'].max():.0f}")

    print(f"Forecast complete — {len(forecast_results)} states.")
    return all_forecasts, forecast_results
