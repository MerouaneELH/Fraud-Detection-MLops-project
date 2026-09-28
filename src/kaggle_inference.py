import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import polars as pl
import polars.selectors as cs
import xgboost as xgb
import mlflow.xgboost
import json
from configs.config import load_config
from src.preprocess import Preprocess

# 1. Load Champion Model
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.xgboost.load_model("models:/Fraud_XGB_Model@champion")

config = load_config("configs/params_inference.yaml")
Preprocessor = Preprocess(config=config, inference=True)
data = Preprocessor.clean_and_export_data()

TransactionID = data.get_column("TransactionID")
data = data.drop(["TransactionID"])

print("[INFERENCE] Applying frequency encoding from training...")
with open("Models/category_mappings.json", "r") as f:
    freq_mappings = json.load(f)

cat_cols = data.select(cs.string()).columns

data = data.with_columns([
    pl.col(col).replace_strict(freq_mappings.get(col, {}), default=0).cast(pl.UInt32)
    for col in cat_cols if col in freq_mappings
])

print("[INFERENCE] Enforcing strict column alignment...")
data = data.select(model.feature_names).cast(pl.Float32)

print("[INFERENCE] Generating predictions...")
dmatrix = xgb.DMatrix(data=data.to_arrow())
predictions = model.predict(dmatrix)

submission_df = pl.DataFrame({
    "TransactionID": TransactionID,
    "isFraud": predictions
})

print("Writing aligned submission.csv")
submission_df.write_csv("Data/submission/submission.csv")