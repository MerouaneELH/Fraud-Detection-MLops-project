import polars as pl
from configs.config import load_config
import polars.selectors as cs

class Preprocess():

    def __init__(self, config: dict):
        self.EXPECTED_SCHEMA = None
        self.data_config = config.get("data", {})
        self.df = None

    def creat_SCHEMA(self) -> None:
        # 1. Base Variables & Exact Categoricals
        self.EXPECTED_SCHEMA = {
            "TransactionDT": pl.UInt32,   
            "TransactionAmt": pl.Float32, 
            "isFraud": pl.UInt8,
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


    def clean_and_export_data(self) -> None:
        # This now handles loading, nulls, feature engineering, and strict casting all at once!
        self.load_and_validate()
        
        # low-cardinality columns
        low_card_cols = [
            "ProductCD", "card4", "card6", "email_match_status",
            "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
            "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
            "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType"
        ]

        print("[PREPROCESS] Compiling Enum expressions for parallel execution...")
        enum_expressions = []
        
        for col in low_card_cols:
            if col in self.df.columns:
                all_cats = self.df.get_column(col).unique().sort().to_list()
                
                if "unknown" not in all_cats:
                    all_cats.append("unknown")
                    
                enum_type = pl.Enum(all_cats)
                enum_expressions.append(pl.col(col).cast(enum_type))

        self.df = self.df.with_columns(enum_expressions)

        # drop TransactionID
        self.df = self.df.drop("TransactionID")

        print(f"[PREPROCESS] Saving pristine data to {self.data_config['processed_path']}")
        self.df.write_parquet(self.data_config["processed_path"])

if __name__ == "__main__":
    config = load_config('configs/params.yaml')
    Preprocesser = Preprocess(config=config)
    Preprocesser.clean_and_export_data()