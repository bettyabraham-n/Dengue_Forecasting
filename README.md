## Forecasting the 2025–2026 Dengue Season in Brazil

This repository contains the code developed for my MSc thesis at AIMS South Africa on probabilistic dengue forecasting in Brazil. The project investigates whether modern deep learning architectures can improve long-horizon dengue forecasts relative to a Bayesian statistical baseline and whether ensemble forecasting further enhances predictive performance.

### Overview

The forecasting framework combines:

* **Bayesian Negative Binomial Model** (statistical baseline)
* **N-HiTS (Neural Hierarchical Interpolation for Time Series)**
* **PatchTST (Patch Time Series Transformer)**
* **Probabilistic Ensemble Forecasting** using equal-weight and WIS-optimised logarithmic pooling

The models generate 52-week-ahead probabilistic forecasts for all 27 Brazilian federative units using historical dengue surveillance data, local climate variables, and large-scale climate indices.

### Objectives

* Identify climate drivers that contribute most to dengue forecast skill.
* Compare the performance of Bayesian and deep learning forecasting approaches.
* Evaluate whether ensemble methods improve probabilistic forecast accuracy.
* Generate a prospective forecast for the 2025–2026 dengue season in Brazil.

### Data

Data were obtained through the International Mosquito-Borne Disease Challenge (IMDC), including:

* Weekly dengue case counts
* Temperature, precipitation, and relative humidity
* Large-scale climate indices (ENSO, IOD, PDO)

### Evaluation

Models were evaluated across the 2022–2023, 2023–2024, and 2024–2025 dengue seasons using:

* Weighted Interval Score (WIS)
* Continuous Ranked Probability Score (CRPS)
* Logarithmic Score (LogS)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

### Key Findings

* N-HiTS and PatchTST consistently outperformed the Bayesian baseline on probabilistic forecasting metrics.
* N-HiTS was the most stable model across validation seasons.
* E2 WIS-optimised ensemble achieved the strongest overall performance.
* The 2025–2026 forecast identified Goias, Minas Gerais, and Sao Paulo as the states with the highest projected dengue burden.





