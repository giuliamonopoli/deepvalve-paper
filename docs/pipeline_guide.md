# Using the DeepValve Pipeline

This page provides a step-by-step guide on how to use the DeepValve pipeline for mitral valve detection in cardiac MRI.

## 1. Installation

1. Clone the repository:

```bash
git clone https://github.com/giuliamonopoli/deepvalve-paper.git
cd "Supplementary code for the paper: DeepValve: an automatic detection pipeline for the mitral valve in cardiac magnetic resonance imaging"
```

2. Create and activate a Python environment.

3. Install dependencies (`requirements.txt`):

```bash
pip install -r requirements.txt

conda activate deepvalve
```

## 2. Repository structure

Key folders and scripts:

- `code/` – core Python code
  - `regression/` – landmark regression models (e.g. DSNT, U-Net)
  - `segmentation/` – segmentation models and utilities
  - `evaluate_models.py` – utilities to evaluate trained models
- `data/` – data-related utilities and example configuration

