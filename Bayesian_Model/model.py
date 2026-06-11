"""
model.py
--------
Global Bayesian Negative Binomial model with non-centred state random effects.

Non-centred parameterisation eliminates the funnel geometry that causes
slow sampling in centred hierarchical models. Expected: 3-15 leapfrog
steps per sample vs 127-255 in the centred version.

Model specification
-------------------
    intercept       ~ Normal(log(mean_y + 1), 2)
    sigma_state     ~ HalfNormal(1)
    state_effects_  ~ Normal(0, 1)              # non-centred
    state_effects   = state_effects_ * sigma_state
    beta[k]         ~ StudentT(nu=4, 0, 1)
    alpha           ~ Gamma(2, 0.5)
    log(mu_t)       = intercept + state_effects[s_t] + beta @ X_t + log(pop_t)
    casos_t         ~ NegativeBinomial(mu_t, alpha)
"""

import numpy as np
import pymc as pm


def bayesian_negbin_global(
    y_train:         np.ndarray,
    X_train:         np.ndarray,
    log_pop_train:   np.ndarray,
    state_idx_train: np.ndarray,
    covariate_names: list,
    n_states:        int,
    X_pred:          np.ndarray = None,
    log_pop_pred:    np.ndarray = None,
    state_idx_pred:  np.ndarray = None,
    draws:           int   = 1000,
    tune:            int   = 1000,
    chains:          int   = 4,
    target_accept:   float = 0.90,
    seed:            int   = 42,
    nuts_sampler:    str   = "numpyro",
):
    """
    Fit the global hierarchical NegBin model and optionally generate
    out-of-sample posterior predictive samples.

    Parameters
    ----------
    y_train, X_train, log_pop_train, state_idx_train : training arrays
    covariate_names : list of covariate column names (for reference)
    n_states        : number of unique states in this run
    X_pred, log_pop_pred, state_idx_pred : optional prediction arrays
    draws, tune, chains, target_accept, seed : MCMC settings
    nuts_sampler    : "numpyro" (GPU/JAX) with CPU NUTS fallback

    Returns
    -------
    idata       : InferenceData from training (includes log_likelihood)
    idata_pred  : InferenceData with posterior predictive on X_pred, or None
    model       : the compiled PyMC model
    """
    intercept_mu = np.log(np.mean(y_train) + 1)

    with pm.Model() as model:
        X_data    = pm.Data("X_data",    X_train,               dims=("obs", "cov"))
        log_pop   = pm.Data("log_pop",   log_pop_train)
        state_idx = pm.Data("state_idx", state_idx_train)
        y_obs_d   = pm.Data("y_obs",     y_train.astype(int))

        intercept = pm.Normal("intercept", mu=intercept_mu, sigma=2)

        # Non-centred parameterisation — eliminates funnel geometry
        sigma_state    = pm.HalfNormal("sigma_state", sigma=1)
        state_effects_ = pm.Normal("state_effects_", mu=0, sigma=1, shape=n_states)
        state_effects  = pm.Deterministic(
            "state_effects", state_effects_ * sigma_state
        )

        beta  = pm.StudentT("beta", nu=4, mu=0, sigma=1, shape=X_train.shape[1])
        alpha = pm.Gamma("alpha", alpha=2, beta=0.5)

        eta = (
            intercept
            + state_effects[state_idx]
            + pm.math.dot(X_data, beta)
            + log_pop
        )
        mu  = pm.Deterministic("mu", pm.math.exp(eta))
        _   = pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=y_obs_d)

        # Sample — NumPyro/JAX (GPU) with CPU NUTS fallback
        try:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=seed,
                nuts_sampler=nuts_sampler,
                return_inferencedata=True,
                progressbar=True,
                idata_kwargs={"log_likelihood": True},
            )
        except Exception as e:
            print(f"  NumPyro failed ({e}), falling back to CPU NUTS")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=seed,
                return_inferencedata=True,
                progressbar=True,
                idata_kwargs={"log_likelihood": True},
            )

    pm.sample_posterior_predictive(
        idata, model=model, random_seed=seed,
        extend_inferencedata=True, progressbar=False,
    )

    idata_pred = None
    if X_pred is not None:
        with model:
            pm.set_data({
                "X_data"   : X_pred,
                "log_pop"  : log_pop_pred,
                "state_idx": state_idx_pred,
                "y_obs"    : np.zeros(len(X_pred), dtype=int),
            })
            idata_pred = pm.sample_posterior_predictive(
                idata,
                var_names=["obs", "mu"],
                random_seed=seed,
                return_inferencedata=True,
                progressbar=False,
            )

    return idata, idata_pred, model
