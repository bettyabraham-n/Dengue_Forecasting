## N-HiTS Global Model For Dengue Forecasting

Global N-HiTS (Neural Hierarchical Interpolation for Time Series) model for dengue case forecasting across all 27 Brazilian states. 


### Model

**Architecture:**
- One global NHITs model trained jointly on all 27 states per season
- Target: `log1p(casos)` — predictions back-transformed via `expm1`
- Future exogenous features: LOO-selected climate covariates + `sin_ew` / `cos_ew`
- Loss: `DistributionLoss(StudentT, level=[50, 80, 90, 95])` — produces calibrated prediction intervals
- Hyperparameters selected via Optuna (TPE sampler, 30 trials), minimising mean WIS across validation seasons



### Workflow

Open `NHITs_Global.ipynb` and run all cells:

1. Config — set `TEST_MODE`, seasons, features
2. Load data
3. Optuna— runs hyperparameter search, saves `outputs/scores/optuna_trials.csv`
4. Extract best params — auto-extracted; can be overridden manually in the notebook
5. Train all seasons — one model per season, validation + forecast
6. Scores — mean WIS/CRPS/LogS/MAE/RMSE by season and state
7. Plots — validation grids, forecast grid, WIS heatmap
8. Save — CSVs to outputs




### Scoring Metrics
All metrics are computed on held-out seasons. Lower is better for all.
WIS: Weighted Interval Score — interval calibration and sharpness 
CRPS: Continuous Ranked Probability Score — full distributional accuracy 
LogS: Log-Score — distributional shape, especially tails 
MAE: Mean Absolute Error on the median forecast 
RMSE: Root Mean Squared Error on the median forecast 


### Test Mode

Set `TEST_MODE = True` in the Config cell for a trial run before committing to a full run.
