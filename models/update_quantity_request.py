from pydantic import BaseModel


class UpdateQuantityRequest(BaseModel):
    product_name: str
    additional_quantity: int
