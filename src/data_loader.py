import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import polars as pl
from sklearn.model_selection import train_test_split
import xgboost as xgb
import polars.selectors as cs


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
        print("[DATA_LOADER] Loading processed data and executing chronological split...")
        self.load_frame()
        # Collect if lazy, then split
        df = self.frame.collect() if isinstance(self.frame, pl.LazyFrame) else self.frame
        # Sorting by time to prevent future data from leaking into the training set
        df = df.sort("TransactionDT")

        X = df.drop("isFraud")
        y = df.get_column("isFraud")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'], 
            shuffle=False
        )

    def fix_high_cardinality(self) -> None:
        print("[DATA_LOADER] Applying frequency encoding to high-cardinality features...")
        high_card_cols = [
            "DeviceInfo", "id_30", "id_31", "id_33",
            "P_emaildomain", "R_emaildomain","card1", 
            "card2", "card3", "card5", "addr1", "addr2"
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
        # A Polars Series allows direct boolean evaluation and summing
        n_counts = (self.y_train == 0).sum()
        p_counts = (self.y_train == 1).sum()
        
        self.scale_weight = n_counts / p_counts
        
        print(f"[DATA_LOADER] Calculated class imbalance weight: {self.scale_weight:.2f}")

    def create_dmatrix(self) -> None:
        # Ensure all preprocessing steps run in the correct order
        self.split_frame()
        self.fix_high_cardinality()
        self.calculate_class_weight()
        print("[DATA_LOADER] Building XGBoost DMatrix structures...")
        # XGBoost cannot digest raw Arrow 'large_string' formats.
        self.X_train = self.X_train.with_columns(cs.string().cast(pl.Categorical))
        self.X_test = self.X_test.with_columns(cs.string().cast(pl.Categorical))
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
        print("[DATA_LOADER] DMatrix built successfully.")
