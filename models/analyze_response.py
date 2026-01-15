from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    status: str                     # "ok", "no_phone", "multiple_phones", ...
    message: str
    splitter_class: Optional[str] = None
    strategy_used: Optional[str] = None

    # classification result (final phone model)
    model_slug: Optional[str] = None        # e.g. "iphone_17"
    brand_readable: Optional[str] = None    # e.g. "Apple iPhone"
    model_readable: Optional[str] = None    # e.g. "17"
    brand_slug: Optional[str] = None        # e.g. "apple"

    # from regression CSV
    storage_options: List[int] = []         # [128, 256, 512] ...

    # inventory info
    product_name: Optional[str] = None      # e.g. "Apple iPhone 17"
    exists_in_inventory: bool = False
    current_quantity: Optional[int] = None

    # confidences
    confidence_splitter: Optional[float] = None  # splitter model
    confidence_brand: Optional[float] = None     # brand classifier (brand pipeline)
    confidence_model: Optional[float] = None     # brand-specific model
    confidence_direct: Optional[float] = None    # direct model classifier

    # raw predicted classes for intermediate models
    brand_classifier_class: Optional[str] = None   # e.g. "iphone", "galaxy", "no_phone"
    direct_classifier_class: Optional[str] = None  # e.g. "iphone_17", "no_phone"
