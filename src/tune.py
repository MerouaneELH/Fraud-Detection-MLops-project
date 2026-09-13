import mlflow
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score
from configs.config import load_config,save_config
from src.data_loader import Data_loader
from pathlib import Path




class Tune:

    def __init__(self, config : dict, loader: Data_loader) -> None:
        self.loader = loader
        self.config = config
        self.model_params = self.config.get("xgboost_params", {})
        self.optuna_params = self.config.get("optuna", {})
        self.model = None
        self.train_history= {}
        self.predictions = None
        self.metrics = {}
        self.predictions = None

    def objective(self, trial: optuna.Trial) -> float:
        
        with mlflow.start_run(nested=True):

            params = self.model_params.copy()
            params["max_depth"] = trial.suggest_int("max_depth", self.optuna_params["max_depth_min"], self.optuna_params["max_depth_max"])
            params["learning_rate"] = trial.suggest_float("learning_rate", self.optuna_params["learning_rate_min"], self.optuna_params["learning_rate_max"], log=True)
            params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", self.optuna_params["scale_pos_weight_min"], self.optuna_params["scale_pos_weight_max"])
            params["subsample"] = trial.suggest_float("subsample", self.optuna_params["subsample_min"], self.optuna_params["subsample_max"])
            params["colsample_bytree"] = trial.suggest_float("colsample_bytree", self.optuna_params["colsample_bytree_min"], self.optuna_params["colsample_bytree_max"])
            params["gamma"] = trial.suggest_float("gamma", self.optuna_params["gamma_min"], self.optuna_params["gamma_max"])

            mlflow.log_params(params)

            self.model = xgb.train(
                dtrain = self.loader.dtrain, 
                params = params,
                num_boost_round = self.config['train']["num_boost_round"],
                early_stopping_rounds = self.config['train']["early_stopping_rounds"],
                evals=[(self.loader.dtrain, "train"), (self.loader.dtest, "test")]
            )
            
            self.predictions = self.model.predict(self.loader.dtest)
            pr_auc = average_precision_score(self.loader.y_test.to_numpy(), self.predictions)

            mlflow.log_metric("pr_auc", pr_auc)

            return pr_auc

    def run_optimization(self) -> None:
        
        with mlflow.start_run(run_name="Optuna_Sweep"):
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=self.config['optuna']['n_trials'])
            
            self.config['xgboost_params'].update(study.best_params)
            save_config(self.config, "configs/params.yaml")





if __name__ == "__main__":

    config = load_config("configs/params.yaml")

    project_root = Path(__file__).resolve().parents[1]
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.set_experiment("Fraud_Detection_model")

    loader = Data_loader(config)
    loader.create_dmatrix()

    tuner = Tune(config, loader)
    tuner.run_optimization()