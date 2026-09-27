import mlflow
import xgboost as xgb
import pandas as pd
from fastapi import HTTPException
import logging
import numpy as np
logger = logging.getLogger(__name__)

class ModelRunner():

    def __init__(self,tracking_uri: str, model_uri: str, batch: bool = False):
        self.batch = batch
        # setting the tracking uri
        mlflow.set_tracking_uri(tracking_uri)

        try:
            # loading model
            self.model = mlflow.xgboost.load_model(model_uri)
            logger.info(f"Model successfully loaded from {model_uri}")
        except Exception as e :
            raise RuntimeError(f"Failed to load model from {model_uri}. Is MLflow running? :{e}")
    

    def predict(self, input: pd.DataFrame ) -> float | np.ndarray :

        if input.empty :
            raise ValueError("Input DataFrame is empty.")
        elif input.shape[1] != 445: 
            raise ValueError(f"Input shape incompatible. Expected 445 columns, got {input.shape[1]}")
        try:
            # creat a Dmatrix
            Dmatrix = xgb.DMatrix(data=input, enable_categorical=True)
            # predict using the created Dmatrix
            prediction = self.model.predict(Dmatrix)
            if self.batch:
                return prediction
            else:
                return float(prediction[0])

        except Exception as e :
            logger.exception(f"XGBoost Prediction Error: {e}")
            raise HTTPException(status_code=500, detail="Internal ML model failure.")
                
