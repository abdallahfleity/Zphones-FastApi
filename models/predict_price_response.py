from pydantic import BaseModel


class PredictPriceResponse(BaseModel):
    predicted_price: float
