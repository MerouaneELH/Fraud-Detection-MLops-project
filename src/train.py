
from src.data_loader import Data_loader
import xgboost as xgb 
import mlflow
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from mlflow.models import infer_signature
import os
from configs.config import load_config


class Train:

    def __init__(self,loader: Data_loader, config: dict):

        self.loader = loader
        self.config = config
        self.params = self.config.get("xgboost_params", {})
        self.model = None
        self.train_history= {}
        self.predictions = None
        self.metrics = {}

    def train_model(self, dvc_hash: str) -> None:

        mlflow.log_params(self.params)
        mlflow.log_artifact(__file__)
        mlflow.log_param("dvc_data_hash", dvc_hash)
        mlflow.log_param("Data used", str(self.config["data"]["processed_path"]))
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
            "final_pr_auc": float(final_pr_auc),
            "optimal_threshold": float(optimal_threshold),
            "best_f1": float(best_f1),
            "best_precision": float(best_precision),
            "best_recall": float(best_recall)
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

    config = load_config("configs/params.yaml")
    dvc_data_config = load_config("Data.dvc")
    dvc_hash = str(dvc_data_config["outs"][0]["md5"])

    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")


    mlflow.set_experiment("Fraud_Detection_model")
    mlflow.xgboost.autolog(log_models=False)

    loader = Data_loader(config=config)
    loader.create_dmatrix()
    
    trainer = Train(loader=loader, config=config)

    with mlflow.start_run(run_name="Fraud_Detection_model"):

        trainer.train_model(dvc_hash)
        trainer.calc_metrics()
        trainer.savelog_model()
        
        