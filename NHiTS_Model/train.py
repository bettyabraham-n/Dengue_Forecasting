import numpy as np
import pandas as pd
import torch
import optuna

from neuralforecast import NeuralForecast

from utils   import TARGET, H, prepare_global_nf, nhits_preds_to_df
from model   import build_nhits_model
from scoring import wis_score, crps_score_energy, logs_score


# Single-season training helper (used by Optuna and run_all_seasons)
def _train_season(
    df_hist,
    df_fcst,
    state_order,
    season,
    forecast_season,
    features,
    input_size,
    n_stacks,
    mlp_units,
    learning_rate,
    batch_size,
    max_steps,
    seed=42,
):
    """
    Fit one global NHITs model for a single season and return raw predictions.
    """
    df_train, cov_map, season_dates = prepare_global_nf(
        df_hist, df_fcst, state_order, season,
        features=features,
        forecast_season=forecast_season,
        input_size=input_size,
    )

    is_forecast = (season == forecast_season)
    model = build_nhits_model(
        input_size, n_stacks, mlp_units,
        learning_rate, batch_size, max_steps,
        features=features, seed=seed,
    )
    nf = NeuralForecast(models=[model], freq='W-SUN')

    nf.fit(df=df_train, val_size=H)

    # Assign future covariates positionally to make_future_dataframe() template
    df_future       = nf.make_future_dataframe()
    df_future['ds'] = pd.to_datetime(df_future['ds'])
    for state in state_order:
        if state not in cov_map:
            continue
        mask = df_future['unique_id'] == state
        n    = mask.sum()
        for col in features:
            df_future.loc[mask, col] = cov_map[state][col].values[:n]

    nan_count = df_future[features].isna().sum().sum()
    if nan_count > 0:
        print(f'  WARNING: {nan_count} NaN in covariates — forward-filled')
        df_future[features] = df_future[features].ffill().bfill()

    fcst_raw = nf.predict(futr_df=df_future)
    return fcst_raw, season_dates, cov_map


# Optuna
def run_optuna(
    df_hist,
    df_fcst,
    state_order,
    val_seasons,
    forecast_season,
    features,
    n_trials=30,
    seed=42,
):
    """
    Run Optuna TPE search over NHITs hyperparameters, minimising mean WIS
    across validation seasons.
    """
    def objective(trial):
        input_size    = trial.suggest_categorical('input_size',    [4, 13, 52, 89, 104])
        n_stacks      = trial.suggest_categorical('n_stacks',      [2, 3])
        mlp_units     = trial.suggest_categorical('mlp_units',     [128, 256, 512])
        learning_rate = trial.suggest_float('learning_rate',       1e-4, 5e-3, log=True)
        batch_size    = trial.suggest_categorical('batch_size',    [16, 32, 64])
        max_steps     = trial.suggest_categorical('max_steps',     [400, 800, 1000, 3000])

        all_wis = []
        for i, season in enumerate(val_seasons):
            try:
                fcst_raw, season_dates, _ = _train_season(
                    df_hist, df_fcst, state_order, season, forecast_season,
                    features, input_size, n_stacks, mlp_units,
                    learning_rate, batch_size, max_steps, seed=seed,
                )
            except Exception as e:
                print(f'    fit failed [{season}]: {e}')
                return 1e9

            year     = int(season[5:])
            wis_vals = []
            for state in state_order:
                df_out = nhits_preds_to_df(fcst_raw, state, season_dates)
                if df_out is None:
                    continue
                df_a = (df_hist[(df_hist['uf'] == state) & (df_hist['ew_year'] == year)]
                        .sort_values('epiweek')
                        .reset_index(drop=True))
                if len(df_a) == 0:
                    continue
                actual = df_a[TARGET].clip(lower=0).values
                n      = min(len(actual), len(df_out))
                wis_vals.append(wis_score(actual[:n], df_out.iloc[:n]))

            season_wis = float(np.mean(wis_vals)) if wis_vals else 1e9
            all_wis.append(season_wis)
            trial.report(season_wis, step=i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(all_wis))

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study   = optuna.create_study(
        direction='minimize', sampler=sampler, pruner=pruner,
        study_name='nhits_global',
    )
    study.optimize(
        objective,
        n_trials          = n_trials,
        show_progress_bar = True,
        catch             = (ValueError, RuntimeError, IndexError),
    )

    best_params = study.best_trial.params
    print('\n' + '='*55)
    print('BEST HYPERPARAMETERS')
    print('='*55)
    for k, v in best_params.items():
        print(f'  {k:20s}: {v}')
    print(f'  {"Best WIS":20s}: {study.best_trial.value:.2f}')

    return best_params, study


# Full training run
def run_all_seasons(
    df_hist,
    df_fcst,
    state_order,
    all_seasons,
    val_seasons,
    forecast_season,
    features,
    best_params,
    seed=42,
):
    """
    Train one global NHITs model per season and score validation seasons.
    """
    torch.cuda.empty_cache()

    input_size    = best_params['input_size']
    n_stacks      = best_params['n_stacks']
    mlp_units     = best_params['mlp_units']
    learning_rate = best_params['learning_rate']
    batch_size    = best_params['batch_size']
    max_steps     = best_params['max_steps']

    all_preds  = {}
    all_scores = []

    for season in all_seasons:
        is_forecast = (season == forecast_season)
        year        = int(season[5:])
        print(f'\n{"="*50}')
        print(f'  Season: {season} [{"FORECAST" if is_forecast else "VALIDATION"}]')
        print(f'{"="*50}')

        try:
            fcst_raw, season_dates, _ = _train_season(
                df_hist, df_fcst, state_order, season, forecast_season,
                features, input_size, n_stacks, mlp_units,
                learning_rate, batch_size, max_steps, seed=seed,
            )
        except Exception as e:
            print(f'  Season failed: {e}')
            continue

        for state in state_order:
            df_out = nhits_preds_to_df(fcst_raw, state, season_dates)
            if df_out is None:
                print(f'  [{state}] no predictions returned')
                continue

            df_out.insert(0, 'season', season)
            df_out.insert(0, 'uf', state)
            all_preds[(state, season)] = df_out

            if is_forecast:
                print(f'  [{state}] peak median: {df_out["pred"].max():.0f}')
            else:
                df_a = (df_hist[(df_hist['uf'] == state) & (df_hist['ew_year'] == year)]
                        .sort_values('epiweek')
                        .reset_index(drop=True))
                if len(df_a) == 0:
                    continue
                actual = df_a[TARGET].clip(lower=0).values
                n      = min(len(actual), len(df_out))
                scores = {
                    'state' : state,
                    'season': season,
                    'WIS'   : wis_score(actual[:n],           df_out.iloc[:n]),
                    'CRPS'  : crps_score_energy(actual[:n],   df_out.iloc[:n]),
                    'LogS'  : logs_score(actual[:n],          df_out.iloc[:n]),
                    'MAE'   : float(np.mean(np.abs(df_out['pred'].values[:n] - actual[:n]))),
                    'RMSE'  : float(np.sqrt(np.mean((df_out['pred'].values[:n] - actual[:n]) ** 2))),
                }
                all_scores.append(scores)
                print(
                    f'  [{state}] WIS={scores["WIS"]:.1f}  '
                    f'CRPS={scores["CRPS"]:.1f}  LogS={scores["LogS"]:.2f}'
                )

    print(f'\nDone. {len(all_preds)} predictions stored.')
    return all_preds, all_scores
