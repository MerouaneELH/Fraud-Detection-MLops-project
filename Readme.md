# Fraud Detection MLOps Project

## Overview
An end-to-end MLOps pipeline for credit card fraud detection on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset. The project covers the full lifecycle: data preprocessing with feature engineering, hyperparameter tuning via Optuna, model training with XGBoost, experiment tracking with MLflow, reproducible pipelines with DVC, and a serving API for real-time inference.

## Project Structure

```
├── configs/                    # Configuration
│   ├── config.py               # YAML loader/saver utilities
│   └── params.yaml             # All hyperparameters & Optuna search ranges
│
├── src/                        # Training pipeline (DVC-orchestrated)
│   ├── preprocess.py           # Schema enforcement, null handling, feature engineering
│   ├── data_loader.py          # Chronological split, frequency encoding, DMatrix creation
│   ├── tune.py                 # Optuna hyperparameter optimization with MLflow tracking
│   └── train.py                # Final model training, metrics, champion/challenger registry
│
├── serving_api/                # Inference API (🚧 in progress)
│   ├── main.py                 # FastAPI application entrypoint
│   ├── api/
│   │   ├── routes.py           # Prediction endpoints
│   │   └── dependencies.py     # Dependency injection
│   ├── core/
│   │   ├── config.py           # API settings
│   │   └── logger.py           # Logging configuration
│   ├── schemas/
│   │   ├── request.py          # Pydantic request models
│   │   └── response.py         # Pydantic response models
│   └── services/
│       ├── model_runner.py     # Model loading & inference
│       ├── preprocessor.py     # Feature transforms for incoming requests
│       └── feature_store.py    # Feature retrieval logic
│
├── NoteBooks/                  # Exploratory notebooks (early prototyping)
│   ├── Data_Processing.ipynb
│   └── Model_Experimentation.ipynb
│
├── Data/                       # Dataset (versioned by DVC)
│   ├── Raw/                    # Original parquet files
│   └── Processed/              # Cleaned & feature-engineered parquet
│
├── dvc.yaml                    # DVC pipeline definition (preprocess → tune → train)
├── dvc.lock                    # Locked pipeline state for reproducibility
├── mlflow.db                   # MLflow SQLite tracking backend
├── mlruns/                     # MLflow artifact store
├── requirements.txt            # Python dependencies
└── Fraud_env/                  # Python virtual environment
```

## Pipeline Architecture

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌─────────────┐
│  preprocess │────▶│   tune   │────▶│  train   │────▶│ serving_api │
│             │     │ (Optuna) │     │          │     │  (FastAPI)  │
│ Schema cast │     │ 100 trials│    │ Best params│   │ Real-time   │
│ Null fill   │     │ Pruning  │     │ Champion/ │    │ inference   │
│ Feature eng │     │ MLflow   │     │ Challenger│    │             │
└─────────────┘     └──────────┘     └──────────┘     └─────────────┘
       ▲                                    │
       │            DVC Pipeline            │
       └────────────────────────────────────┘
```

## Key Features

- **Feature Engineering**: Temporal features (hour/day), email match detection, cumulative velocity features (uid frequency, device diversity, hourly/daily/weekly transaction counts & sums), and transaction amount ratios — all time-safe (no future leakage).
- **Chronological Split**: Data is sorted by `TransactionDT` and split without shuffling to prevent temporal leakage.
- **Frequency Encoding**: High-cardinality categoricals (`DeviceInfo`, `card1`, `addr1`, emails, etc.) are encoded using train-set frequencies.
- **Hyperparameter Tuning**: Optuna with XGBoost pruning callback for efficient search over learning rate, depth, regularization (`reg_alpha`, `reg_lambda`, `gamma`), subsampling, and class weight.
- **Champion/Challenger Registry**: New models are only promoted to "champion" in the MLflow Model Registry if they beat the current best on PR-AUC.
- **Reproducibility**: Full DVC pipeline (`dvc repro`) ensures every run is traceable from raw data to registered model.

## Technologies

| Category | Tools |
|---|---|
| **ML Framework** | XGBoost (gradient boosting with native categorical support) |
| **Data Processing** | Polars (columnar DataFrame library) |
| **Hyperparameter Tuning** | Optuna (Bayesian optimization with pruning) |
| **Experiment Tracking** | MLflow (params, metrics, artifacts, model registry) |
| **Data Versioning** | DVC (Data Version Control) |
| **Serving API** | FastAPI (🚧 in progress) |

## Getting Started

### 1. Set Up the Environment
```bash
# On Windows:
Fraud_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Pull the Data
```bash
dvc pull
```

### 3. Run the Full Pipeline
```bash
dvc repro
```
This executes `preprocess` → `tune` → `train` in order, skipping stages that haven't changed.

### 4. Run Individual Stages
```bash
# Preprocessing only
python -m src.preprocess

# Hyperparameter tuning only
python -m src.tune

# Training only (uses best params from tuning)
python -m src.train
```

### 5. View Experiments
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open http://localhost:5000 to browse experiments, compare runs, and inspect the model registry.
