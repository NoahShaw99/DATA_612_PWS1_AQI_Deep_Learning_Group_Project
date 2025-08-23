# How to Reproduce Our Results

This guide reproduces the training runs, evaluation metrics, and the four output plots per configuration.

## 1) Environment

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
```

## 2) Run the Notebook

Start Jupyter and open the final notebook:

```bash
jupyter lab
```

You can also open your preferred environment of choice (Visual Studio Code, etc) and make sure that all of the dependencies in requirements.txt are installed correctly and that your environment has access to them.
From there you can open the notebook and run all cells from top to bottom from there.

Instructions on how to install all of the dependencies:

Python: https://www.python.org/download/releases/2.5/msi/ and https://realpython.com/installing-python/

NumPy: https://numpy.org/install/

Pandas: https://pandas.pydata.org/docs/getting_started/install.html

Scikit-learn: https://scikit-learn.org/stable/install.html

Matplotlib: https://matplotlib.org/stable/install/index.html

Seaborn: https://seaborn.pydata.org/installing.html

PyTorch: https://pytorch.org/get-started/locally/

Kagglehub: https://pypi.org/project/kagglehub/

In `AQI_Forecast_Project_Final_Version.ipynb`:
1. Run the **Project Overview & Configuration** cells (Should automatically pip install and import all necessary packages).
2. Run the **Data Fetch** cells (KaggleHub will fetch the dataset).
3. Run the **Per-Group Daily Series** and **Windowing & Splits** cells.
4. Run the **Model Architecture** cell.
5. Run the **Training Helpers** cells.
6. Run the **Model Training** and **Evaluation & Report** cells.
7. Run the **Experiments** cells that calls:

```python
for lb in SWEEP_LOOKBACKS:      # [7, 30, 60]
    for bs in SWEEP_BATCH_SIZES:  # [32, 64, 96, 128]
        res, history = run_training_experiment(lb, bs, epochs=EPOCHS)
```
This will train and evaluate each configuration and save plots.

## 3) Expected Artifacts (`./aqi_outputs/`)

For **each (lb, bs)** you should see **four** plots:

- `trainloss_vs_valloss_lb{LB}_bs{BS}.png`
- `trainloss_vs_valrmse_lb{LB}_bs{BS}.png`
- `error_over_horizon_lb{LB}_bs{BS}.png`
- `pred_vs_actual_lb{LB}_bs{BS}.png`

For the primary run you’ll also see:
- `model_best.pt` (best checkpoint by validation RMSE)
- `training_history.csv` (epoch-by-epoch metrics)

## 4) Notes

We set seeds where appropriate but there can be small run‑to‑run variations. Please expect minor differences in RMSE/R² around ±0.01.
