# models/supported_phones_response.py
from typing import List
from pydantic import BaseModel


class SupportedPhonesResponse(BaseModel):
    phones: List[str]
