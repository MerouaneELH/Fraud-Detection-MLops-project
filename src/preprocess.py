import polars as pl 
from configs.config import load_config
import polars.selectors as cs


class Preprocess():


    def __init__(self,config: dict):
        self.EXPECTED_SCHEMA = None
        self.data_config = config.get("data", {})
        self.df = None

    def creat_SCHEMA(self) -> None:

        # 1. Base Variables & Exact Categoricals
        self.EXPECTED_SCHEMA = {
            "TransactionDT": pl.UInt32,   # Timedelta in seconds
            "TransactionAmt": pl.Float32, # Float needed for the 3-decimal foreign exchange rates
            "isFraud": pl.UInt8,
            
            # Categorical features explicitly named in the documentation
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

        # 5. Vesta Engineered Features (V1-V339: Strictly Numerical per Kaggle hosts)
        for i in range(1, 340): self.EXPECTED_SCHEMA[f"V{i}"] = pl.Float32

        # 6. Identity Columns (id_01 - id_38: Split between Numerical and Categorical)
        for i in range(1, 39):
            col_name = f"id_{i:02d}"
            # The docs specify id_01 to id_11 are numerical ratings/login times
            # id_12 to id_38 are categorical network/browser signatures
            self.EXPECTED_SCHEMA[col_name] = pl.Float32 if i <= 11 else pl.Utf8

    def load_and_validate(self) -> pl.DataFrame:

        self.creat_SCHEMA()

        print("[PREPROCESS] Loading raw parquet file...")

        self.df = pl.read_parquet(self.data_config['raw_path'])

        # Rename id- columns before validating the schema
        self.df = self.df.rename({
            col: col.replace("-", "_") for col in self.df.columns if col.startswith("id-")
        })
        
        # Ensure all expected columns exist
        missing_cols = [col for col in self.EXPECTED_SCHEMA.keys() if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Raw data is missing required columns: {missing_cols}")


        print("[PREPROCESS] Enforcing strict Polars schema...")
        # Force the schema types
        self.df = self.df.cast(self.EXPECTED_SCHEMA)

    

    def clean_and_export_data(self,) -> None:

        self.load_and_validate()

        print("[PREPROCESS] Filling string nulls and fixing id hyphens...")
        # Fill all string nulls with 'unknown'
        self.df = self.df.with_columns(cs.string().fill_null("unknown"))
        
        
        # low-cardinality columns
        low_card_cols = [
            "ProductCD", "card4", "card6",
            "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
            "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
            "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType"
        ]

        # Extract categories dynamically and cast to Enum
        for col in low_card_cols:
            # Only process if the column exists to avoid errors
            if col in self.df.columns:
                # Extract unique values from the column
                all_cats = self.df.get_column(col).unique().sort().to_list()
                
                # Ensure "unknown" is in the enum space, since we filled nulls with it
                if "unknown" not in all_cats:
                    all_cats.append("unknown")
                    
                enum_type = pl.Enum(all_cats)
                self.df = self.df.with_columns(pl.col(col).cast(enum_type))

        print(f"[PREPROCESS] Saving pristine data to {self.data_config['processed_path']}")
        
        self.df.write_parquet(self.data_config["processed_path"])

if __name__ == "__main__":
    config = load_config('configs/params.yaml')

    Preprocesser = Preprocess(config=config)
    Preprocesser.clean_and_export_data()
