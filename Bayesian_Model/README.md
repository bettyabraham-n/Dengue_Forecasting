## Bayesian Negative Binomial Model

Global hierarchical Bayesian Negative Binomial model for dengue case forecasting across all 27 Brazilian ufs.

## Workflow

### Step 1 — LOO covariate selection (run once)

Open `LOO/apply_LOO.ipynb` and run all cells. This fits the full model plus one
reduced model per candidate covariate, selects those that meaningfully improve
out-of-sample ELPD, and saves the result to outputs/scores/selected_covariates.json

### Step 2 — Training and forecasting

Open `Bayesian_Global_Model.ipynb` and run all cells. It will:

1. Load the selected covariates from 'outputs/scores/selected_covariates.json'
   (or fall back to a hardcoded default list if LOO has not been run yet)
2. Run leave-one-season-out validation on 2022-2023, 2023-2024, 2024-2025
3. Fit on all historical data and forecast the 2025-2026 season
4. Save plots, scores, and prediction CSVs to 'outputs'


### Scoring Metrics
All metrics are computed on held-out seasons. Lower is better for all.
WIS: Weighted Interval Score — interval calibration and sharpness 
CRPS: Continuous Ranked Probability Score — full distributional accuracy 
LogS: Log-Score — distributional shape, especially tails 
MAE: Mean Absolute Error on the median forecast 
RMSE: Root Mean Squared Error on the median forecast 

### Test Mode

Set `TEST_MODE = True` in either notebook to run a quick sanity check on 3–5
states with reduced MCMC iterations before committing to a full run.
