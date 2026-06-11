import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def wis_score(y_obs: np.ndarray, pi_df: pd.DataFrame) -> float:
    levels = {"50": 0.50, "80": 0.80, "90": 0.90, "95": 0.95}
    wis = np.abs(y_obs - pi_df["median"].values) / 2.0
    for lvl, cov in levels.items():
        a = 1 - cov
        L = pi_df[f"lower_{lvl}"].values
        U = pi_df[f"upper_{lvl}"].values
        wis += (a / 2) * (
            (U - L)
            + 2 * np.maximum(0, L - y_obs)
            + 2 * np.maximum(0, y_obs - U)
        )
    return float(np.mean(wis))


def crps_score(y_obs: np.ndarray, samples: np.ndarray) -> float:
    M     = samples.shape[0]
    term1 = np.abs(samples - y_obs).mean(axis=0)
    fs    = np.sort(samples, axis=0)
    w     = (2 * np.arange(1, M + 1) - M - 1).reshape(-1, 1)
    return float(np.mean(term1 - (w * fs).sum(axis=0) / M ** 2))


def log_score(y_obs: np.ndarray, samples: np.ndarray, eps: float = 1e-10) -> float:
    scores = []
    for t in range(samples.shape[1]):
        d = samples[:, t].astype(float)
        if d.std() < 1e-8:
            scores.append(30.0)
            continue
        scores.append(
            -np.log(float(gaussian_kde(d).evaluate([y_obs[t]])[0]) + eps)
        )
    return float(np.mean(scores))


def compute_all_scores(
    y_obs: np.ndarray,
    pi_df: pd.DataFrame,
    samples: np.ndarray,
) -> dict:
    med = pi_df["median"].values
    return {
        "WIS" : round(wis_score(y_obs, pi_df),                          2),
        "CRPS": round(crps_score(y_obs, samples),                        2),
        "LogS": round(log_score(y_obs, samples),                         4),
        "MAE" : round(float(np.mean(np.abs(y_obs - med))),               2),
        "RMSE": round(float(np.sqrt(np.mean((y_obs - med) ** 2))),       2),
    }
