
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
        # Drop TransactionDT — time features already extracted in preprocessing
        df = df.drop("TransactionDT")


        X = df.drop("isFraud")
        y = df.get_column("isFraud")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'], 
            shuffle=False
        )

    def encode_categoricals(self) -> None:
        print("[DATA_LOADER] Applying universal frequency encoding to all string features...")
        import json
        
        # 1. Automatically find all string columns
        cat_cols = self.X_train.select(cs.string()).columns
        
        # 2. Build the frequency dictionary
        freq_mappings = {}
        for col in cat_cols:
            counts = self.X_train.get_column(col).value_counts()
            freq_mappings[col] = dict(zip(counts[col], counts["count"]))
            
        # 3. Save the exact training memory for Kaggle inference
        with open("Models/category_mappings.json", "w") as f:
            json.dump(freq_mappings, f)
        print(f"[DATA_LOADER] Saved frequency mappings for {len(cat_cols)} columns to Models/category_mappings.json")
            
        # 4. Apply the mapping (Unseen categories default to 0)
        self.X_train = self.X_train.with_columns([
            pl.col(col).replace_strict(freq_mappings[col], default=0).cast(pl.UInt32)
            for col in cat_cols
        ])
        
        self.X_test = self.X_test.with_columns([
            pl.col(col).replace_strict(freq_mappings[col], default=0).cast(pl.UInt32)
            for col in cat_cols
        ])

    def calculate_class_weight(self) -> None:
        # A Polars Series allows direct boolean evaluation and summing
        n_counts = (self.y_train == 0).sum()
        p_counts = (self.y_train == 1).sum()
        
        self.scale_weight = n_counts / p_counts
        
        print(f"[DATA_LOADER] Calculated class imbalance weight: {self.scale_weight:.2f}")

    def prepare(self) -> None:
        # Ensure all preprocessing steps run in the correct order
        self.split_frame()
        self.encode_categoricals()
        self.calculate_class_weight()
        
        print("[DATA_LOADER] Building XGBoost DMatrix structures...")
        
        # CRITICAL: enable_categorical=True is REMOVED. 
        # Everything is a number now, so XGBoost uses its standard numerical engine.
        self.dtrain = xgb.DMatrix(
            data=self.X_train.to_arrow(), 
            label=self.y_train.to_arrow()
        )
        self.dtest = xgb.DMatrix(
            data=self.X_test.to_arrow(),
            label=self.y_test.to_arrow()
        )
        print("[DATA_LOADER] DMatrix built successfully.")

    