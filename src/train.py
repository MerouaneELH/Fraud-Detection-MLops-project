
from src.data_loader import Data_loader
import pandas as pd
import xgboost as xgb 
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from mlflow.models import infer_signature
import os
from configs.config import load_config

dvc_data_config = load_config("Data.dvc")


class Train:

    def __init__(self):

        self.loader = Data_loader()
        self.loader.create_dmatrix()
        self.params = {
                    "objective": self.loader.config["xgboost_params"]["objective"],
                    "tree_method": self.loader.config["xgboost_params"]["tree_method"],
                    "eval_metric": self.loader.config["xgboost_params"]["eval_metric"],
                    "scale_pos_weight": self.loader.config["xgboost_params"]["scale_pos_weight"],
                    "subsample": self.loader.config["xgboost_params"]["subsample"],
                    "colsample_bytree": self.loader.config["xgboost_params"]["colsample_bytree"]
                }
        self.model = None
        self.train_history= {}
        self.predictions = None
        self.metrics = {}

    def train_model(self) -> None:

        mlflow.log_params(self.params)
        mlflow.log_artifact("train.py")
        mlflow.log_param("dvc_data_hash", str(dvc_data_config["outs"]["md5"]))
        mlflow.log_param("Data used", str(self.loader.config["data"]["processed_path"]))
        mlflow.log_param("train_set_size", len(self.loader.X_train))

        self.model = xgb.train(
            dtrain = self.loader.dtrain,
            params = self.params,
            num_boost_round = self.config['train']["num_boost_round"],
            early_stopping_rounds = self.config['train']["early_stopping_rounds"],
            evals= [(self.loader.dtrain, "train"), (self.loader.dtest, "test")],
            evals_result = self.train_history

            )

    def calc_metrics(self) -> None:

        self.predictions = self.model.predict(self.loader.dtest)

        precision, recall, thresholds = precision_recall_curve(self.loader.y_test.to_numpy(), self.predictions)
        final_pr_auc = auc(recall, precision)
        
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        best_idx = np.argmax(f1_scores)
        
        optimal_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]

        self.metrics = {
            "final_pr_auc": final_pr_auc,
            "optimal_threshold": optimal_threshold,
            "best_f1": best_f1,
            "best_precision": best_precision,
            "best_recall": best_recall
        }

        # log metrics
        mlflow.log_metrics(self.metrics)




    def savelog_model(self) -> None:

        signature = infer_signature(self.loader.X_test.to_pandas(), self.predictions)
        
        mlflow.xgboost.log_model(
            xgb_model=self.model, 
            artifact_path="model",
            ##registered_model_name="Fraud_XGB_Model" , # Registers it in the UI
            signature=signature 
            )


if __name__ == "__main__":

    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")


    mlflow.set_experiment("Fraud_Detection_model")
    mlflow.xgboost.autolog(log_models=False)

    trainer = Train()

    with mlflow.start_run(run_name="Fraud_Detection_model"):

        trainer.train_model()
        trainer.calc_metrics()
        trainer.savelog_model()
        
        