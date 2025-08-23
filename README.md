# AQI Hybrid Lite — Forecasting Daily Air Quality (7‑day horizon)

Github Repository for DATA 612 PWS1 AQI Deep Learning Group Final Project at UMD.

Group members are Alima Suleimenova, Mahammad Afandiyev, Noah Shaw, Dan Ding, Aaron Kim

This repository contains the final code and artifacts for our deep‑learning AQI forecaster for our final group project for DATA 612 PWS1.
The model is a compact hybrid architecture:

- **RevIN** per-sample normalization (and exact de‑normalization at inference)
- **Moving-average decomposition** → seasonal (high‑frequency) + trend (low‑frequency)
- **Seasonal encoder:** patchified tokens processed by a **Transformer with depthwise conv**, optional Fourier low‑pass residual
- **Trend branch:** light **temporal CNN** with global pooling
- **Gated fusion** of seasonal/trend embeddings
- **Head:** predicts normalized corrections which are added to a learned blend of Last value and Last‑week value

---

## Data & Preprocessing

- Source: Kaggle dataset `azminetoushikwasi/aqi-air-quality-index-scheduled-daily-update` (downloaded via **kagglehub**).
- Link to dataset: https://www.kaggle.com/datasets/azminetoushikwasi/aqi-air-quality-index-scheduled-daily-update
- For each city/station we:
  1. Parse dates and compute the daily mean AQI.
  2. Reindex to a continuous daily range; forward‑fill values for model inputs while keeping a boolean obs mask of true observations.
  3. Build sliding windows of length **LOOKBACK** → predict the next **PRED_LEN=7** days.
  4. Drop windows whose observation mask contains a gap of more than `max_consec_gap=7` consecutive missing days.

**Splits (time‑based):** last 28 days = test, previous 28 days = validation, remainder = train (per group).

**Observation threshold rule:** `min_obs = 100` when `LOOKBACK=7`, and `min_obs = 123` when `LOOKBACK∈{30,60}`.

---

## Training

- Optimizer: AdamW
- Objective: SmoothL1 (Huber) + λ‖Δ‖² smoothness on day‑to‑day forecast deltas
- Mixed precision (AMP) on CUDA
- LR scheduler: ReduceLROnPlateau on validation RMSE
- Early stopping: stop when no meaningful improvement; best (by val RMSE) checkpoint is restored before test.

---

## Experiments

We sweep **LOOKBACK × BATCH_SIZE**:

```
LOOKBACK   = [7, 30, 60]
BATCH_SIZE = [32, 64, 96, 128]
```

Call `run_training_experiment(lookback, batch_size, epochs)` for each pair.

---

## Outputs

All artifacts are saved in `./aqi_outputs/`.

For each configuration we print the metrics and save four plots:

1. **Train vs Validation Loss** – (SmoothL1 + λ‖Δ‖²)  
   Filename: `trainloss_vs_valloss_lb{LB}_bs{BS}.png`
**Train vs Validation RMSE**  
   Filename: `train_vs_val_rmse_lb{LB}_bs{BS}.png`
3. **MAE per Horizon** (Error over forecast day 1 through 7)  
   Filename: `error_over_horizon_lb{LB}_bs{BS}.png`
4. **Predicted vs Actual** scatter with dashed red ideal line and R² in title  
   Filename: `pred_vs_actual_lb{LB}_bs{BS}.png`

Additionally, the notebook writes:
- Best model weights (`model_best.pt`) for the primary single model run
- Training history CSV (`training_history.csv`)

---

## Quickstart

Python version that was used was **3.10.18**

```bash
# (Optional) create & activate a virtual env
# macOS/Linux
python3 -m venv .venv && source .venv/bin/activate

# Windows (PowerShell)
py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# (If Jupyter is missing)
pip install jupyterlab

# Launch Jupyter
jupyter lab
```

Open `AQI_Forecast_Project_Final_Version.ipynb` and run all cells.  
To run the sweeps explicitly, execute the cell that loops over `SWEEP_LOOKBACKS` and `SWEEP_BATCH_SIZES`; plots for each pair will be written under `./aqi_outputs/`.

You can also open your preferred environment of choice (Visual Studio Code, etc) and make sure that all of the dependencies in requirements.txt are installed correctly and that your environment has access to them.
From there you can open the notebook and run all cells from top to bottom from there.

For more detailed instructions please see `Instructions.md`.

### Notes on Reproducibility
We set seeds where appropriate but there can be small run‑to‑run variations. Please expect minor differences in RMSE/R² around ±0.01.

### Implementation and Runtime Notes
The implementation tools that we used for this project are Visual Studio Code and Google Colab.

The runtime of the primary single model run was around 2-3 minutes and the runtime of the experiments was about 30 minutes on Noah Shaw's home PC.

---

## Repository Structure

```
.
├── AQI_Forecast_Project_Final_Version.ipynb
├── requirements.txt
├── README.md
├── Instructions.md
└── aqi_outputs/        # can also be created by the notebook
```
