
from src.data_loader import Data_loader
import xgboost as xgb 
import mlflow
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from mlflow.models import infer_signature
from configs.config import load_config
from pathlib import Path
from mlflow.tracking import MlflowClient


class Train:

    def __init__(self,loader: Data_loader, config: dict):

        self.loader = loader
        self.config = config
        self.params = self.config.get("xgboost_params", {})
        self.model = None
        self.train_history= {}
        self.predictions = None
        self.metrics = {}

    def train_model(self, processed_hash: str) -> None:
        print(f"[TRAIN] Starting XGBoost training for {self.config['train']['num_boost_round']} rounds...")
        mlflow.log_params(self.params)
        mlflow.log_artifact(__file__)
        mlflow.log_param("processed_data_hash", processed_hash)
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
        print(f"[TRAIN] Training stopped at best iteration: {self.model.best_iteration}")

    def calc_metrics(self) -> None:

        print("[TRAIN] Calculating Precision-Recall AUC and F1 thresholds...")
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
        print("[TRAIN] Logging model artifact and signature to MLflow...")
        signature = infer_signature(self.loader.X_test.to_pandas(), self.predictions)
        
        mlflow.xgboost.log_model(
            xgb_model=self.model, 
            artifact_path="model",
            ##registered_model_name="Fraud_XGB_Model" , # Registers it in the UI
            signature=signature 
            )

    def register_if_better(self, run_id: str, model_name: str = "Fraud_XGB_Model") -> None:
        print("[REGISTRY] Initiating Champion vs. Challenger evaluation...")
        client = MlflowClient()
        new_pr_auc = self.metrics["final_pr_auc"]
        
        try:
            # Fetch the current reigning champion from the registry
            champion = client.get_model_version_by_alias(model_name, "champion")
            champion_run = client.get_run(champion.run_id)
            
            # Extract its historical PR-AUC score (default to 0 if missing)
            champion_pr_auc = champion_run.data.metrics.get("final_pr_auc", 0.0)
            
            print(f"Current Champion PR-AUC: {champion_pr_auc:.4f}")
            print(f"New Challenger PR-AUC: {new_pr_auc:.4f}")
            
            # The Showdown
            if new_pr_auc > champion_pr_auc:
                print("Challenger wins! Registering as the new champion...")
                new_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
                client.set_registered_model_alias(model_name, "champion", new_version.version)
            else:
                print("Challenger failed to beat the champion. Saved to history, but not registered.")
                
        except Exception:
            # If the alias doesn't exist (first time running the pipeline)
            print("No existing champion found. Registering as the inaugural champion...")
            new_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
            client.set_registered_model_alias(model_name, "champion", new_version.version)



if __name__ == "__main__":

    config = load_config("configs/params.yaml")
    project_root = Path(__file__).resolve().parents[1]

    try:
        lock_data = load_config("dvc.lock")
        processed_hash = str(lock_data["stages"]["preprocess"]["outs"][0]["md5"])
    except Exception:
        processed_hash = "run_dvc_repro_first"

    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.set_experiment("Fraud_Detection_model")
    mlflow.xgboost.autolog(log_models=False)

    loader = Data_loader(config=config)
    loader.create_dmatrix()
    
    trainer = Train(loader=loader, config=config)

    with mlflow.start_run(run_name="Fraud_Detection_model") as run:

        trainer.train_model(processed_hash)
        trainer.calc_metrics()
        trainer.savelog_model()
        trainer.register_if_better(run_id=run.info.run_id)
        
        