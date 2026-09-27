from pydantic import BaseModel, Field


class TransactionResponse(BaseModel):
    TransactionID: int
    fraud_probability: float
    is_fraud: bool