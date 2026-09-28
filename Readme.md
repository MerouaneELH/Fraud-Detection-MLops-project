# Fraud Detection MLOps Project

## Overview
An end-to-end MLOps pipeline for credit card fraud detection on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset. The project covers the full lifecycle: data preprocessing with feature engineering, hyperparameter tuning via Optuna, model training with XGBoost, experiment tracking with MLflow, reproducible pipelines with DVC, Kaggle submission generation, and a serving API for real-time inference.

**🏆 Kaggle Leaderboard Score: 0.909951 (ROC-AUC)**

## Project Structure

```
├── configs/                    # Configuration
│   ├── config.py               # YAML loader/saver utilities
│   ├── params.yaml             # All hyperparameters & Optuna search ranges
│   └── params_inference.yaml   # Inference-specific data paths
│
├── src/                        # Training pipeline (DVC-orchestrated)
│   ├── preprocess.py           # Schema enforcement, null handling, feature engineering
│   ├── data_loader.py          # Chronological split, frequency encoding, DMatrix creation
│   ├── tune.py                 # Optuna hyperparameter optimization with pruning & MLflow
│   ├── train.py                # Final training, metrics, champion/challenger registry
│   └── kaggle_inference.py     # End-to-end Kaggle test set prediction & submission
│
├── serving_api/                # Real-Time Inference API (🚧 in progress)
│   ├── main.py                 # FastAPI application entrypoint
│   ├── api/
│   │   ├── routes.py           # Prediction endpoints
│   │   └── dependencies.py     # Dependency injection
│   ├── core/
│   │   ├── config.py           # API settings
│   │   └── logger.py           # Logging configuration
│   ├── schemas/
│   │   ├── request.py          # Pydantic request model (all 400+ IEEE-CIS fields)
│   │   └── response.py         # Pydantic response model (TransactionID, probability, is_fraud)
│   └── services/
│       ├── model_runner.py     # Champion model loading from MLflow & DMatrix prediction
│       ├── preprocessor.py     # Feature transforms for incoming requests
│       └── feature_store.py    # Redis-backed card state tracking (tx_count, total_amt)
│
├── Models/                     # Saved artifacts from training
│   ├── category_mappings.json  # Frequency encoding dictionaries (used at inference)
│   └── training_columns.json   # Exact column list for inference schema alignment
│
├── NoteBooks/                  # Exploratory notebooks (early prototyping)
│   ├── Data_Processing.ipynb
│   └── Model_Experimentation.ipynb
│
├── Data/                       # Dataset (versioned by DVC)
│   ├── Raw/                    # Original parquet files (train.parquet, test.parquet)
│   ├── Processed/              # Cleaned & feature-engineered parquet
│   └── submission/             # Kaggle submission CSV output
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
┌──────────────┐      ┌───────────┐     ┌────────────┐
│  preprocess  │────▶│   tune    │────▶│  train     │
│              │      │ (Optuna)  │     │            │
│ Schema cast  │      │ 100 trials│     │ Best params│
│ Null fill    │      │ Pruning   │     │ Champion/  │
│ Feature eng  │      │ MLflow    │     │ Challenger │
│ V-col filter │      └───────────┘     └────────────┘
└──────────────┘           │                │
       │              DVC Pipeline          │
       │                                    │
       ▼                                    ▼
┌─────────────────┐              ┌─────────────────┐
│ kaggle_inference│              │   serving_api   │
│                 │              │    (FastAPI)    │
│ Freq encode from│              │ Redis feature   │
│ training memory │              │ store + MLflow  │
│ → submission.csv│              │ champion model  │
└─────────────────┘              └─────────────────┘
```

## Key Features

### Training Pipeline
- **Schema Enforcement**: Strict Polars schema casting for all 400+ columns — V1-V339 as Float32, id_01-id_11 as Float32, id_12-id_38 as Utf8, card/M/addr columns as Utf8.
- **Feature Engineering**: Temporal features (hour/day), email match detection (`Both_Unknown`, `Partial_Unknown`, `Exact_Match`, `Explicit_Mismatch`), cumulative velocity features (uid frequency, device diversity, hourly/daily/weekly transaction counts & sums), and transaction amount ratios — all time-safe using `cum_sum().over()` to prevent future data leakage.
- **Automatic Column Pruning**: Drops columns with ≥90% missing values and removes highly correlated V-columns (threshold: 0.75) during training.
- **Chronological Split**: Data is sorted by `TransactionDT` and split without shuffling to prevent temporal leakage. `TransactionDT` is dropped after time-based features are extracted.
- **Universal Frequency Encoding**: All string columns are frequency-encoded using train-set value counts. Mappings are saved to `Models/category_mappings.json` for inference reuse.
- **Hyperparameter Tuning**: Optuna with XGBoost pruning callback for efficient search over learning rate, depth, regularization (`reg_alpha`, `reg_lambda`, `gamma`), subsampling, `min_child_weight`, and `scale_pos_weight` (range 10-50 for class imbalance).
- **Champion/Challenger Registry**: New models are only promoted to "champion" in the MLflow Model Registry if they beat the current best on PR-AUC.
- **Reproducibility**: Full DVC pipeline (`dvc repro`) ensures every run is traceable from raw data to registered model.
### Inference
- **Kaggle Inference** (`src/kaggle_inference.py`): Loads the champion model from MLflow, preprocesses the test parquet, applies the exact same frequency encoding from training (via `category_mappings.json`), aligns columns to match training features (via `training_columns.json`), and outputs `submission.csv`.
- **Serving API** (`serving_api/`): FastAPI-based real-time inference with Pydantic request validation for all IEEE-CIS fields, `ModelRunner` that loads the champion model from MLflow, and a Redis-backed `FeatureStore` for tracking per-card state (transaction counts and cumulative amounts). *(🚧 Under active development — routes and preprocessor being wired up)*

## Technologies

| Category | Tools |
|---|---|
| **ML Framework** | XGBoost (gradient boosting with frequency-encoded categoricals) |
| **Data Processing** | Polars (columnar DataFrame library) |
| **Hyperparameter Tuning** | Optuna (Bayesian optimization with XGBoost pruning callback) |
| **Experiment Tracking** | MLflow (params, metrics, artifacts, model registry) |
| **Data Versioning** | DVC (Data Version Control) |
| **Serving API** | FastAPI + Pydantic (🚧 in progress) |
| **Feature Store** | Redis (per-card state for real-time features) |

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

# Hyperparameter tuning only (100 Optuna trials)
python -m src.tune

# Training only (uses best params from tuning)
python -m src.train
```

## 5. Generate Kaggle Submission
```bash
# Requires MLflow UI running for champion model loading
mlflow ui --backend-store-uri sqlite:///mlflow.db &
python -m src.kaggle_inference
```
Outputs `Data/submission/submission.csv` ready for Kaggle upload.

### 6. View Experiments
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open http://localhost:5000 to browse experiments, compare runs, and inspect the model registry.
