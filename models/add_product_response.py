from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class AddProductResponse(BaseModel):
    id: str
    product_id: str
    product_name: str
    product_type: str
    image_gridfs_id: str
    price_predicted: float
    price_modified: Optional[float]
    quantity: int
    date_added: datetime
