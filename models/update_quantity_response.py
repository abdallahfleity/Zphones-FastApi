from pydantic import BaseModel


class UpdateQuantityResponse(BaseModel):
    id: str
    product_name: str
    new_quantity: int
