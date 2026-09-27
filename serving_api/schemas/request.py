from pydantic import create_model,Field, StringConstraints
from typing import Annotated


CleanStr = Annotated[str,StringConstraints(strip_whitespace=True, to_lower=True)]

schema_fields ={
    "TransactionID" : (int,Field(...,ge=0)) ,
    "TransactionDT" : (int,Field(...,ge=0)) ,
    "TransactionAmt": (float,Field(...,ge=0.0)) , 
    "ProductCD": (CleanStr,Field(default="unknown")),
    "addr1": (CleanStr,Field(default="unknown")), 
    "addr2": (CleanStr,Field(default="unknown")), 
    "P_emaildomain": (CleanStr,Field(default="unknown")), 
    "R_emaildomain": (CleanStr,Field(default="unknown")), 
    "DeviceType": (CleanStr,Field(default="unknown")), 
    "DeviceInfo": (CleanStr,Field(default="unknown")), 
}

# (card1 - card6: Categorical)
for i in range(1, 7): schema_fields[f"card{i}"] = (CleanStr,Field(default="unknown"))

# (M1 - M9: Categorical)
for i in range(1, 10): schema_fields[f"M{i}"] = (CleanStr,Field(default="unknown"))

# (C1-C14, D1-D15: Numerical)
for i in range(1, 15): schema_fields[f"C{i}"] = (float,Field(...,ge=0.0))
for i in range(1, 16): schema_fields[f"D{i}"] = (float,...)

# (V1-V339: Strictly Numerical)
for i in range(1, 340): schema_fields[f"V{i}"] = (float,...)

# (id_01 - id_38)
for i in range(1, 39):
    col_name = f"id_{i:02d}"
    schema_fields[col_name] = (float,...) if i <= 11 else (CleanStr,Field(default="unknown"))


TransactionRequest = create_model("TransactionRequest",**schema_fields)


