import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.config import  load_config, save_config

import polars as pl 
from sklearn.model_selection import train_test_split
import xgboost as xgb

config = load_config("configs/params.yaml")

#data_loader class for handling data before training 

class Data_loader:


    def __init__(self,config: dict):

        self.config = config
        self.frame = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.dtrain = None
        self.dtest = None
        self.scale_weight = None

    def load_frame(self, lazy: bool = False) -> None:
        processed_path = self.config['data']['processed_path']
        if lazy:
            self.frame = pl.scan_parquet(processed_path)
        else:
            self.frame = pl.read_parquet(processed_path)

    def split_frame(self) -> None:
        # Collect if lazy, then split
        df = self.frame.collect() if isinstance(self.frame, pl.LazyFrame) else self.frame
        X = df.drop("isFraud")
        y = df.select("isFraud")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            stratify=y, 
            test_size=self.config['data']['test_size'], 
            random_state=self.config['data']['random_state']
        )

    def fix_high_cardinality(self) -> None:
        high_card_cols = [
            "DeviceInfo", "id_30", "id_31", "id_33",
            "P_emaildomain", "R_emaildomain"
        ]
        
        freq_mappings = {}
        for col in high_card_cols:
            counts = self.X_train.get_column(col).value_counts()
            freq_mappings[col] = dict(zip(counts[col], counts["count"]))

        self.X_train = self.X_train.with_columns([
            pl.col(col).replace_strict(freq_mappings[col], default=0).cast(pl.UInt32)
            for col in high_card_cols
        ])

        self.X_test = self.X_test.with_columns([
            pl.col(col).replace_strict(freq_mappings[col], default=0).cast(pl.UInt32)
            for col in high_card_cols
        ])

    def calculate_class_weight(self) -> None:
        # Calculate directly from the isolated target set
        n_counts = self.y_train.filter(pl.col("isFraud") == 0).height
        p_counts = self.y_train.filter(pl.col("isFraud") == 1).height
        self.scale_weight = n_counts / p_counts

    def create_dmatrix(self) -> None:
        # Ensure all preprocessing steps run in the correct order
        self.split_frame()
        self.fix_high_cardinality()
        self.calculate_class_weight()
        
        self.dtrain = xgb.DMatrix(
            data=self.X_train.to_arrow(), 
            label=self.y_train.to_arrow(),
            enable_categorical=True
        )
        self.dtest = xgb.DMatrix(
            data=self.X_test.to_arrow(),
            label=self.y_test.to_arrow(),
            enable_categorical=True
        )





if __name__ == "__main__":
    pass