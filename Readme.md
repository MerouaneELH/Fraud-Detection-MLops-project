# Fraud Detection MLOps Project

## Overview
This project focuses on building a machine learning pipeline for fraud detection. It implements modern MLOps practices, utilizing XGBoost for modeling, Polars for fast data processing, DVC for data versioning, and MLflow for experiment tracking and model registry.

## Project Structure

- `Data/`: Directory for data versioned by DVC. Contains `Raw/` and `Processed/` datasets.
- `NoteBooks/`: Jupyter notebooks for data processing and model experimentation.
  - `Data_Processing.ipynb`: Handles data cleaning, preprocessing, filling missing values, enum casting for categorical features, and saves ready-to-use parquet files.
  - `Model_Experimentation.ipynb`: Handles train/test splitting, frequency encoding for high-cardinality features, XGBoost model training, evaluation (PR-AUC, F1), and MLflow logging.
- `Scripts/`: Python scripts for pipeline execution and automation.
- `Models/`: Directory intended for saved or exported models.
- `Fraud_env/`: Python virtual environment.
- `Data.dvc`: DVC tracking file for the dataset.

## Technologies Used
- **XGBoost**: Gradient boosting framework for baseline model training with native categorical support.
- **Polars**: Lightning-fast DataFrame library for data manipulation and preprocessing.
- **MLflow**: Used for tracking experiments, hyperparameters, metrics, and models.
- **DVC (Data Version Control)**: Manages and versions large data artifacts.

## Getting Started

### 1. Set Up the Environment
Activate the virtual environment and install dependencies:
```bash
# On Windows:
Fraud_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Get the Data
Pull the versioned dataset using DVC:
```bash
dvc pull
```

### 3. Run the Pipeline
1. Execute `NoteBooks/Data_Processing.ipynb` to parse the raw data and generate `train_ready.parquet` and `test_ready.parquet` in the `Data/Processed/` directory.
2. Execute `NoteBooks/Model_Experimentation.ipynb` to encode features, train the baseline model, and track the experiment.

### 4. View Experiments in MLflow
To view the logged runs, parameters, and metrics, launch the MLflow UI from the project root directory:

```bash
mlflow ui --backend-store-uri file:///%cd%/mlruns
```
*(Note: Change the URI to `sqlite:///%cd%/mlflow.db` if using a SQLite backend for tracking.)*

