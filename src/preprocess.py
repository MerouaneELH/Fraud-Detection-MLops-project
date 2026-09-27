import polars as pl
from configs.config import load_config
import polars.selectors as cs
import json

class Preprocess():

    def __init__(self, config: dict, inference: bool = False):
        self.inference = inference
        self.EXPECTED_SCHEMA = None
        self.data_config = config.get("data", {})
        self.df = None

    def creat_SCHEMA(self) -> None:
        # 1. Base Variables & Exact Categoricals
        self.EXPECTED_SCHEMA = {
            "TransactionDT": pl.UInt32,   
            "TransactionAmt": pl.Float32, 
            "ProductCD": pl.Utf8,
            "addr1": pl.Utf8,
            "addr2": pl.Utf8,
            "P_emaildomain": pl.Utf8,
            "R_emaildomain": pl.Utf8,
            "DeviceType": pl.Utf8,
            "DeviceInfo": pl.Utf8,
        }

        # 2. Card Columns (card1 - card6: Categorical)
        for i in range(1, 7): self.EXPECTED_SCHEMA[f"card{i}"] = pl.Utf8

        # 3. Match Columns (M1 - M9: Categorical)
        for i in range(1, 10): self.EXPECTED_SCHEMA[f"M{i}"] = pl.Utf8

        # 4. Counting & Time Variables (C1-C14, D1-D15: Numerical)
        for i in range(1, 15): self.EXPECTED_SCHEMA[f"C{i}"] = pl.Float32
        for i in range(1, 16): self.EXPECTED_SCHEMA[f"D{i}"] = pl.Float32

        # 5. Vesta Engineered Features (V1-V339: Strictly Numerical)
        for i in range(1, 340): self.EXPECTED_SCHEMA[f"V{i}"] = pl.Float32

        # 6. Identity Columns (id_01 - id_38)
        for i in range(1, 39):
            col_name = f"id_{i:02d}"
            self.EXPECTED_SCHEMA[col_name] = pl.Float32 if i <= 11 else pl.Utf8

        

    def load_and_validate(self) -> None:
        self.creat_SCHEMA()
        print("[PREPROCESS] Loading raw parquet file...")
        self.df = pl.read_parquet(self.data_config['raw_path'])
        if "isFraud" in self.df.columns : self.EXPECTED_SCHEMA["isFraud"] = pl.UInt8
        # Rename id- columns
        self.df = self.df.rename({
            col: col.replace("-", "_") for col in self.df.columns if col.startswith("id-")
        })

        print("[PREPROCESS] Filling string nulls with 'unknown'...")
        # CRITICAL: This must happen BEFORE feature engineering!
        self.df = self.df.with_columns(cs.string().fill_null("unknown"))

        # Inject the new features and update schema
        self.add_features()
        
        # Validate and cast BOTH raw and engineered columns
        raw_expected_cols = [col for col in self.EXPECTED_SCHEMA.keys() if col in self.df.columns]
        print("[PREPROCESS] Enforcing strict Polars schema on all data...")
        self.df = self.df.cast({col: self.EXPECTED_SCHEMA[col] for col in raw_expected_cols})

    def remove_highly_correlated_v_cols(self, threshold: float = 0.75) -> None:
        import numpy as np
        print(f"[PREPROCESS] Identifying V-columns with correlation > {threshold}...")
        
        # 1. Select only the V columns that exist in the dataframe
        v_cols = [col for col in self.df.columns if col.startswith("V")]
        if not v_cols:
            return
            
        # 2. CRITICAL FIX: Convert to Pandas to handle nulls via pairwise deletion
        v_df_pandas = self.df.select(v_cols).to_pandas()
        corr_matrix = v_df_pandas.corr()
        
        cols_to_drop = set()
        
        # 3. Iterate through the matrix to find highly correlated pairs
        corr_arrays = corr_matrix.to_numpy()
        
        for i in range(len(v_cols)):
            for j in range(i + 1, len(v_cols)):
                val = corr_arrays[i, j]
                
                # Check that the value is not NaN AND is above the threshold
                if not np.isnan(val) and abs(val) > threshold:
                    cols_to_drop.add(v_cols[j])
                    
        print(f"[PREPROCESS] Dropping {len(cols_to_drop)} redundant V-columns out of {len(v_cols)}.")
        
        if cols_to_drop:
            self.df = self.df.drop(list(cols_to_drop))


    
    def add_features(self) -> None:
        print("[PREPROCESS] Engineering behavioral, temporal, and velocity features...")
        
        self.df = self.df.sort("TransactionDT")
        
        # 1. Base Helpers & Cyclical Time
        self.df = self.df.with_columns(
            hour_of_the_day = ((pl.col("TransactionDT") // 3600) % 24).cast(pl.UInt32),
            day_of_the_week = ((pl.col("TransactionDT") // 86400) % 7).cast(pl.UInt32),
            absolute_hour = (pl.col("TransactionDT") // 3600).cast(pl.UInt32),
            absolute_day = (pl.col("TransactionDT") // 86400).cast(pl.UInt32),
            absolute_week = (pl.col("TransactionDT") // 604800).cast(pl.UInt32)
        )

        # 2. String Logic (Email Match Status)
        self.df = self.df.with_columns(
            email_match_status = (
                pl.when(
                    (pl.col("P_emaildomain") == "unknown") & (pl.col("R_emaildomain") == "unknown")
                ).then(pl.lit("Both_Unknown"))
                .when(
                    (pl.col("P_emaildomain") == "unknown") | (pl.col("R_emaildomain") == "unknown")
                ).then(pl.lit("Partial_Unknown"))
                .when(
                    (pl.col("P_emaildomain") == pl.col("R_emaildomain")) & (pl.col("P_emaildomain") != "unknown")
                ).then(pl.lit("Exact_Match"))
                .otherwise(pl.lit("Explicit_Mismatch"))
            ).cast(pl.Utf8)
        )

        # 3. Core Feature Engineering Chain
        self.df = (
            self.df.with_columns(
                uid_string = pl.concat_str(["card1", "addr1", "P_emaildomain"], separator="_", ignore_nulls=True),
                transaction_cents = (pl.col("TransactionAmt") % 1).cast(pl.Float32)
            )
            .with_columns(
                
                # --- PLUGGING THE TIME LEAKS (Cumulative instead of Global) ---
                
                # How many times has this uid appeared UP TO THIS ROW?
                uid_c = pl.lit(1).cum_sum().over("uid_string").cast(pl.UInt32),
                
                # How many unique UIDs has this card used UP TO THIS ROW?
                uid_diversity = pl.col("uid_string").is_first_distinct().cum_sum().over("card1").cast(pl.UInt32),
                
                # How many unique devices has this card used UP TO THIS ROW?
                device_diversity = pl.col("DeviceInfo").is_first_distinct().cum_sum().over("card1").cast(pl.UInt32),
                
                # Ratio to cumulative mean (What is their average spend UP TO THIS ROW?)
                amt_ratio_to_card_mean = (
                    pl.col("TransactionAmt") / (
                        pl.col("TransactionAmt").cum_sum().over("card1") / 
                        pl.col("TransactionAmt").cum_count().over("card1")
                    )
                ).cast(pl.Float32),
                
                # (time-safe cumulative)
                hourly_tx_count = pl.lit(1).cum_sum().over(["absolute_hour", "card1"]).cast(pl.UInt32),
                hourly_tx_sum = pl.col("TransactionAmt").cum_sum().over(["absolute_hour", "card1"]).cast(pl.Float32),
                daily_tx_count = pl.lit(1).cum_sum().over(["absolute_day", "card1"]).cast(pl.UInt32),
                daily_tx_sum = pl.col("TransactionAmt").cum_sum().over(["absolute_day", "card1"]).cast(pl.Float32),
                weekly_tx_count = pl.lit(1).cum_sum().over(["absolute_week", "card1"]).cast(pl.UInt32),
                weekly_tx_sum = pl.col("TransactionAmt").cum_sum().over(["absolute_week", "card1"]).cast(pl.Float32),
            )
        )

        # 4. Clean up temporary helpers
        self.df = self.df.drop([
            "uid_string", 
            "absolute_hour", 
            "absolute_day", 
            "absolute_week"
        ])
        #Engineered Behavioral & Temporal Features (ADDED)
        self.EXPECTED_SCHEMA.update({
            "hour_of_the_day": pl.UInt32,
            "day_of_the_week": pl.UInt32,
            "email_match_status": pl.Utf8, 
            "transaction_cents": pl.Float32,
            "uid_c": pl.UInt32,
            "uid_diversity": pl.UInt32,
            "device_diversity": pl.UInt32,
            "amt_ratio_to_card_mean": pl.Float32,
            "hourly_tx_count": pl.UInt32,
            "hourly_tx_sum": pl.Float32,
            "daily_tx_count": pl.UInt32,
            "daily_tx_sum": pl.Float32,
            "weekly_tx_count": pl.UInt32,
            "weekly_tx_sum": pl.Float32,
            })


    def clean_and_export_data(self) -> pl.DataFrame:
        
        self.load_and_validate()
        
        if not self.inference:
            # --- TRAINING MODE ---
            print("[PREPROCESS] Removing columns with >= 90% missing values...")
            cols_to_drop_nulls = ['dist2', 'D7', 'id_07', 'id_08', 'id_18', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27']
            safe_drop_nulls = [c for c in cols_to_drop_nulls if c in self.df.columns]
            self.df = self.df.drop(safe_drop_nulls)
            
            self.remove_highly_correlated_v_cols(threshold=0.75)
            self.df = self.df.drop("TransactionID")
            
            # Save the final exact column list (excluding isFraud)
            final_cols = [col for col in self.df.columns if col != "isFraud"]
            with open("Models/training_columns.json", "w") as f:
                json.dump(final_cols, f)
            print(f"[PREPROCESS] Saved {len(final_cols)} training columns to Models/training_columns.json")
            
        else:
            # --- INFERENCE MODE ---
            print("[PREPROCESS] Inference mode active. Aligning schema with training data...")
            with open("Models/training_columns.json", "r") as f:
                training_cols = json.load(f)
            
            # CRITICAL: Keep TransactionID alongside training columns so it tracks row movement!
            cols_to_select = ["TransactionID"] + [c for c in training_cols if c != "TransactionID"]
            self.df = self.df.select(cols_to_select)

        print(f"[PREPROCESS] Saving pristine data to {self.data_config['processed_path']}")
        self.df.write_parquet(self.data_config["processed_path"])
        return self.df

if __name__ == "__main__":
    config = load_config('configs/params.yaml')
    Preprocesser = Preprocess(config=config)
    Preprocesser.clean_and_export_data()