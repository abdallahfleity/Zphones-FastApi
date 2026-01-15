from typing import Literal

from pydantic import BaseModel


class PredictPriceRequest(BaseModel):
    model_slug: str
    brand_slug: str
    storage_gb: int
    condition: Literal["new", "used"]
