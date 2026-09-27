import redis 
import json 
import logging

logger = logging.getLogger(__name__)


class FeatureStore:

    def __init__(self, host: str = "localhost", db: int = 0, port: int = 6379):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        try : 
            self.client.ping()
            logger.info("Successfully connected to Redis Feature Store.")
        except Exception as e:
            raise RuntimeError(f"Couldn't connect to Redis. Is it running? Error: {e}")

    def get_card_state(self,card_id : str) -> dict:
        data = self.client.get(card_id)
        if not data :
            logger.info(f"New card detected: {card_id}. Initializing state.")
            return {"tx_count": 0, "total_amt": 0.0}
        else :
            return json.loads(data)

    def update_card_state(self,card_id: str,transaction_amt: float) -> None:
        data_dict = self.get_card_state(card_id)
        data_dict["tx_count"] += 1
        data_dict["total_amt"] += transaction_amt
        updated_data_string = json.dumps(data_dict)
        self.client.set(card_id, updated_data_string)
        
