# Using the DeepValve Pipeline

This page provides a step-by-step guide on how to use the DeepValve pipeline for mitral valve detection in cardiac MRI.

## 1. Installation

1. Clone the repository:

```bash
git clone https://github.com/giuliamonopoli/deepvalve-paper.git
cd "Supplementary code for the paper: DeepValve: an automatic detection pipeline for the mitral valve in cardiac magnetic resonance imaging"
```

2. Create and activate a Python environment (optional but recommended).

3. Install dependencies (either with `requirements.txt` or `environment.yml`):

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate deepvalve
```

## 2. Repository structure

Key folders and scripts:

- `code/` – core Python code
  - `regression/` – landmark regression models (e.g. DSNT, U-Net)
  - `segmentation/` – segmentation models and utilities
  - `evaluate_models.py` – utilities to evaluate trained models
- `data/` – data-related utilities and example configuration

## 3. Data preparation

### 3.1 Segmentation data

Use the helpers in `data/segmentation_data/` to prepare segmentation data:

- `data/segmentation_data/create_data.py` – script to create the dataset structure expected by the segmentation pipeline.

Adapt paths and settings in this script to match your local data organization.

### 3.2 Regression data

For regression experiments, ensure you have a CSV similar to `data/regression/data_new.csv` with the required columns (e.g. subject IDs, frame information, landmark coordinates).

## 4. Running the regression pipeline

The main entry points for the regression models are in `code/regression/`:

- `dsnt_main.py` – training and evaluation for the DSNT-based regression model.
- `unet_main.py` – training and evaluation for the U-Net-based regression model (if present in the repo).

Example (from the `code/regression` folder):

```bash
cd code/regression
python dsnt_main.py --help
```

Use the command-line options (or configuration inside the script) to specify:

- Input data paths
- Output directory for models and logs
- Training hyperparameters (epochs, learning rate, batch size, etc.)

## 5. Running the segmentation pipeline

The segmentation entry point is in `code/segmentation/main.py`.

From the repository root:

```bash
cd code/segmentation
python main.py --help
```

Configure:

- Paths to training/validation/test images
- Paths to segmentation labels
- Model and training hyperparameters (configured in `config_seg.py`).

## 6. Evaluating trained models

Use `code/evaluate_models.py` to evaluate regression and segmentation models.

From the repository root:

```bash
cd code
python evaluate_models.py --help
```

Typical steps:

1. Point the script to your trained model checkpoint(s).
2. Provide the path to the evaluation dataset (images and labels/landmarks).
3. Run the script to compute metrics such as:
   - Regression: MAE, MSE, RMSE, MAPE, cosine similarity, Procrustes distance.
   - Segmentation: Dice/F-score, IoU/Jaccard, and other custom metrics.

## 7. Visualising results

The repository includes plotting utilities, for example in `code/plotter.py`, to:

- Plot predicted vs. ground-truth splines or landmarks.
- Summarise metrics across the dataset.

Refer to the docstrings in these modules or examples in `evaluate_models.py` for usage patterns.

## 8. Reproducibility tips

- Fix random seeds (NumPy, PyTorch) in your training scripts for reproducible experiments.
- Keep track of the commit hash of the code used for each experiment.
- Store configuration files (JSON/YAML) alongside model checkpoints when possible.

## 9. Getting help

- See the `about` page for an overview of this supplementary material.
- Check the repository README for environment and dataset details.
- For issues or questions, use the GitHub Issues button in the top-right of this documentation site (if enabled).