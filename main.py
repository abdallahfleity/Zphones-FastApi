import io
import os
import uuid
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from models.inventory_item import InventoryItem
from models.supported_phones_response import SupportedPhonesResponse

import numpy as np
import pandas as pd
import tensorflow as tf
from bson import ObjectId
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
import json
import joblib

from config import (
    MONGO_URI,
    MONGO_DB_NAME,
    GRIDFS_BUCKET_NAME,
    IMG_SIZE,
    CONFIDENCE_THRESHOLD,
    SPLITTER_MODEL_PATH,
    SPLITTER_CLASSES,
    BRAND_MODEL_PATH,
    BRAND_CLASSES_JSON,
    BRAND_MODEL_CONFIG,
    DIRECT_MODEL_PATH,
    DIRECT_CLASSES_JSON,
    DIRECT_MAPPING_JSON,
    REG_ENCODER_PATH,
    REG_MODEL_PATH,
    REG_DATA_CSV,
)

from models.analyze_response import AnalyzeResponse
from models.add_product_response import AddProductResponse
from models.update_quantity_request import UpdateQuantityRequest
from models.update_quantity_response import UpdateQuantityResponse
from models.predict_price_request import PredictPriceRequest
from models.predict_price_response import PredictPriceResponse

# ----------------- GLOBALS (loaded at startup) -----------------

phone_splitter_model: tf.keras.Model | None = None
brand_model: tf.keras.Model | None = None
brand_classes: List[str] = []

brand_model_map: Dict[str, tf.keras.Model] = {}
brand_classname_map: Dict[str, List[str]] = {}

direct_model: tf.keras.Model | None = None
direct_classes: List[str] = []
label_mapping_by_slug: Dict[str, Dict[str, str]] = {}

reg_encoder = None
reg_model = None
reg_df: pd.DataFrame | None = None

mongo_client: AsyncIOMotorClient | None = None
mongo_db = None
gridfs_bucket: AsyncIOMotorGridFSBucket | None = None
DIRECT_IMG_SIZE: tuple[int, int] | None = None

# ----------------- FASTAPI APP -----------------

app = FastAPI(title="Multimedia Inventory API")

# Allow React (change origin if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Optional small enums (if you still want them) -----------------


class ClassifierStrategy(str):
    BRAND_PIPELINE = "brand_pipeline"
    DIRECT_MODEL = "direct_model"


class ConditionType(str):
    NEW = "new"
    USED = "used"

class ManualClassifyRequest(BaseModel):
    model_slug: str

# ----------------- UTILITIES -----------------


def serialize_object_id(oid: ObjectId) -> str:
    return str(oid)


def serialize_product(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(doc["_id"]),
        "product_id": doc["product_id"],
        "product_name": doc["product_name"],
        "product_type": doc["product_type"],
        "image_gridfs_id": str(doc["image_gridfs_id"]),
        "price_predicted": float(doc["price_predicted"]),
        "price_modified": float(doc["price_modified"])
        if doc.get("price_modified") is not None
        else None,
        "quantity": int(doc["quantity"]),
        "date_added": doc["date_added"],
    }


def load_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image_bytes(content: bytes, img_size=IMG_SIZE) -> np.ndarray:
    img = tf.keras.utils.load_img(io.BytesIO(content), target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr


def brand_to_reg_slug(brand_raw: str) -> str:
    """Convert mapping/brand-class names to regression-brand slug like 'apple', 'samsung'."""
    b = brand_raw.lower()
    if "samsung" in b:
        return "samsung"
    if "apple" in b:
        return "apple"
    if "google" in b:
        return "google"
    if "huawei" in b:
        return "huawei"
    if "tecno" in b:
        return "tecno"
    if "xiaomi" in b:
        return "xiaomi"
    if "oppo" in b:
        return "oppo"
    return b.split()[0]


def get_storage_options(model_slug: str) -> List[int]:
    if reg_df is None:
        return []
    subset = reg_df[reg_df["model_slug"] == model_slug]
    return sorted(subset["storage_gb"].unique().tolist())


def predict_price(model_slug: str, brand_slug: str, storage_gb: int, condition: str) -> float:
    if reg_encoder is None or reg_model is None:
        raise RuntimeError("Regression components not loaded.")
    df = pd.DataFrame(
        [
            {
                "model_slug": model_slug,
                "brand": brand_slug,
                "storage_gb": int(storage_gb),
                "condition": condition,
            }
        ]
    )
    df_enc = reg_encoder.transform(df)
    price = float(reg_model.predict(df_enc)[0])
    return price


# ----------------- ML PIPELINES -----------------


def splitter_predict(content: bytes) -> dict:
    arr = preprocess_image_bytes(content, IMG_SIZE)
    preds = phone_splitter_model.predict(arr)
    idx = int(np.argmax(preds[0]))
    class_name = SPLITTER_CLASSES[idx]
    confidence = float(np.max(preds[0]))
    return {"class_name": class_name, "confidence": confidence}


def brand_pipeline_predict(content: bytes) -> dict:
    """
    Brand pipeline:
      1) Brand classifier  (can output galaxy/iphone/... or no_phone/multiple_phones)
      2) Brand-specific model classifier (phone model)
    We always return brand class + conf, and model class + conf (if valid).
    """
    arr = preprocess_image_bytes(content, IMG_SIZE)

    # ---- Brand classifier ----
    preds_brand = brand_model.predict(arr)
    idx_brand = int(np.argmax(preds_brand[0]))
    conf_brand = float(preds_brand[0][idx_brand])
    brand_class = brand_classes[idx_brand]  # "galaxy", "iphone", "no_phone", ...

    # hard stop on "no_phone"/"multiple_phones"
    if brand_class in ["multiple_phones", "no_phone"]:
        return {
            "valid": False,
            "reason": brand_class,
            "brand_class": brand_class,
            "brand_confidence": conf_brand,
        }

    # low confidence on brand
    if conf_brand < CONFIDENCE_THRESHOLD:
        return {
            "valid": False,
            "reason": "low_confidence_brand",
            "brand_class": brand_class,
            "brand_confidence": conf_brand,
        }

    # ---- Brand-specific model classifier ----
    if brand_class not in BRAND_MODEL_CONFIG:
        return {
            "valid": False,
            "reason": "unknown_brand",
            "brand_class": brand_class,
            "brand_confidence": conf_brand,
        }

    mdl = brand_model_map[brand_class]
    mdl_classes = brand_classname_map[brand_class]

    preds_model = mdl.predict(arr)
    idx_model = int(np.argmax(preds_model[0]))
    conf_model = float(preds_model[0][idx_model])
    model_slug = mdl_classes[idx_model]  # e.g. "galaxy_s25_ultra"

    # low confidence on model classifier
    if conf_model < CONFIDENCE_THRESHOLD:
        return {
            "valid": False,
            "reason": "low_confidence_model",
            "brand_class": brand_class,
            "brand_confidence": conf_brand,
            "model_slug": model_slug,
            "model_confidence": conf_model,
        }

    # pretty names
    mapping = label_mapping_by_slug.get(model_slug, None)
    if mapping:
        brand_readable = mapping["brand"]
        model_readable = mapping["model"]
    else:
        brand_readable = brand_class
        model_readable = model_slug

    brand_slug = brand_to_reg_slug(brand_readable)

    return {
        "valid": True,
        "brand_class": brand_class,
        "brand_confidence": conf_brand,
        "model_confidence": conf_model,
        "brand_readable": brand_readable,
        "model_readable": model_readable,
        "brand_slug": brand_slug,
        "model_slug": model_slug,
    }


def direct_model_predict(content: bytes) -> dict:
    """
    Direct model pipeline:
      splitter (outside this function) + single classifier.
    Always return class + confidence.
    """
    # Use the direct model's real input size (set at startup)
    target_size = DIRECT_IMG_SIZE if DIRECT_IMG_SIZE is not None else IMG_SIZE
    arr = preprocess_image_bytes(content, target_size)

    preds = direct_model.predict(arr)
    idx = int(np.argmax(preds[0]))
    conf = float(preds[0][idx])
    class_name = direct_classes[idx]  # can be "no_phone"/"multiple_phones" too

    # hard stop when classifier says not a single phone
    if class_name in ["multiple_phones", "no_phone"]:
        return {
            "valid": False,
            "reason": class_name,
            "direct_class": class_name,
            "confidence": conf,
        }

    # low confidence
    if conf < CONFIDENCE_THRESHOLD:
        return {
            "valid": False,
            "reason": "low_confidence_direct",
            "direct_class": class_name,
            "confidence": conf,
        }

    mapping = label_mapping_by_slug.get(class_name)
    if not mapping:
        brand_readable = "Unknown"
        model_readable = class_name
    else:
        brand_readable = mapping["brand"]
        model_readable = mapping["model"]

    brand_slug = brand_to_reg_slug(brand_readable)

    return {
        "valid": True,
        "direct_class": class_name,
        "confidence": conf,
        "brand_readable": brand_readable,
        "model_readable": model_readable,
        "brand_slug": brand_slug,
        "model_slug": class_name,
    }


# ----------------- STARTUP / SHUTDOWN -----------------


@app.on_event("startup")
async def startup_event():
    global mongo_client, mongo_db, gridfs_bucket
    global phone_splitter_model, brand_model, brand_classes
    global brand_model_map, brand_classname_map
    global direct_model, direct_classes, label_mapping_by_slug
    global reg_encoder, reg_model, reg_df
    global DIRECT_IMG_SIZE

    # Mongo
    mongo_client = AsyncIOMotorClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB_NAME]
    gridfs_bucket = AsyncIOMotorGridFSBucket(mongo_db, bucket_name=GRIDFS_BUCKET_NAME)

    # Splitter model
    phone_splitter_model = tf.keras.models.load_model(SPLITTER_MODEL_PATH)

    # Brand classifier
    brand_model = tf.keras.models.load_model(BRAND_MODEL_PATH)
    brand_classes = load_json(BRAND_CLASSES_JSON)

    # Brand-specific models
    for brand_key, cfg in BRAND_MODEL_CONFIG.items():
        mdl = tf.keras.models.load_model(cfg["model_path"])
        cls = load_json(cfg["classes_path"])
        brand_model_map[brand_key] = mdl
        brand_classname_map[brand_key] = cls

    # Direct classifier
    direct_model = tf.keras.models.load_model(DIRECT_MODEL_PATH)
    direct_classes = load_json(DIRECT_CLASSES_JSON)
    mapping_list = load_json(DIRECT_MAPPING_JSON)
    label_mapping_by_slug = {m["class_name"]: m for m in mapping_list}

    # Figure out the correct input size for the direct model
    if isinstance(direct_model.input_shape, (list, tuple)) and len(direct_model.input_shape) >= 3:
        h, w = direct_model.input_shape[1], direct_model.input_shape[2]
        # Some models might use None, fallback to IMG_SIZE if so
        if h is None or w is None:
            DIRECT_IMG_SIZE = IMG_SIZE
        else:
            DIRECT_IMG_SIZE = (int(h), int(w))
    else:
        DIRECT_IMG_SIZE = IMG_SIZE

    # Regression
    reg_encoder = joblib.load(REG_ENCODER_PATH)
    reg_model = joblib.load(REG_MODEL_PATH)
    reg_df = pd.read_csv(REG_DATA_CSV)


@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client
    if mongo_client:
        mongo_client.close()


# ----------------- ROUTES -----------------


@app.get("/")
async def root():
    return {"message": "Multimedia Inventory API running"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    strategy: Literal["brand_pipeline", "direct_model"] = Form(...),
):
    """
    Step 1:
    - Splitter: check if phone / no phone / multiple phones.
    - If not a single high-confidence phone -> return status to frontend.
    - If phone:
        * strategy = "brand_pipeline" -> brand classifier + brand-specific model.
        * strategy = "direct_model"   -> direct model classifier.
    - Then:
        * compute storage_options from regression CSV (based on model_slug)
        * check if product already exists in DB (by product_name = "Brand Model")
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # ---- 1) Splitter ----
    split_res = splitter_predict(content)
    splitter_class = split_res["class_name"]
    splitter_conf = split_res["confidence"]

    # If splitter says "no_phone"/"multiple_phones" OR low confidence -> stop here
    if splitter_class in ["no_phone", "multiple_phones"]:
        return AnalyzeResponse(
            status=splitter_class,
            message=f"Splitter result: {splitter_class}",
            splitter_class=splitter_class,
            confidence_splitter=splitter_conf,
        )

    if splitter_conf < CONFIDENCE_THRESHOLD:
        return AnalyzeResponse(
            status="low_confidence",
            message="Prediction rejected: splitter confidence too low.",
            splitter_class=splitter_class,
            confidence_splitter=splitter_conf,
        )

    # ---- 2) Classification according to strategy ----
    if strategy == "brand_pipeline":
        clf_res = brand_pipeline_predict(content)
        strat_used = "brand_pipeline"

        brand_conf = clf_res.get("brand_confidence")
        model_conf = clf_res.get("model_confidence")
        direct_conf = None
        brand_classifier_class = clf_res.get("brand_class")
        direct_classifier_class = None
    else:
        clf_res = direct_model_predict(content)
        strat_used = "direct_model"

        brand_conf = None
        model_conf = None
        direct_conf = clf_res.get("confidence")
        brand_classifier_class = None
        direct_classifier_class = clf_res.get("direct_class")

    # If classifier reports "no_phone"/"multiple_phones" or low_confidence, stop
    if not clf_res["valid"]:
        reason = clf_res.get("reason", "invalid")
        # group all low confidence reasons into a single status for frontend
        status = "low_confidence" if reason.startswith("low_confidence") else reason

        return AnalyzeResponse(
            status=status,
            message=f"Classification failed: {reason}",
            splitter_class=splitter_class,
            strategy_used=strat_used,
            confidence_splitter=splitter_conf,
            confidence_brand=brand_conf,
            confidence_model=model_conf,
            confidence_direct=direct_conf,
            brand_classifier_class=brand_classifier_class,
            direct_classifier_class=direct_classifier_class,
        )

    # ---- 3) If all good, build final product info ----
    model_slug = clf_res["model_slug"]
    brand_readable = clf_res["brand_readable"]
    model_readable = clf_res["model_readable"]
    brand_slug = clf_res["brand_slug"]

    # "Base" product name WITHOUT storage/condition
    product_name = f"{brand_readable} {model_readable}".strip()

    # 4) storage options from CSV
    storage_opts = get_storage_options(model_slug)

    # 5) DB check: does this product already exist?
    existing = await mongo_db["products"].find_one({"product_name": product_name})
    exists = existing is not None
    current_quantity = int(existing["quantity"]) if existing else None

    return AnalyzeResponse(
        status="ok",
        message="Analysis successful",
        splitter_class=splitter_class,
        strategy_used=strat_used,
        model_slug=model_slug,
        brand_readable=brand_readable,
        model_readable=model_readable,
        brand_slug=brand_slug,
        storage_options=storage_opts,
        product_name=product_name,
        exists_in_inventory=exists,
        current_quantity=current_quantity,
        confidence_splitter=splitter_conf,
        confidence_brand=brand_conf,
        confidence_model=model_conf,
        confidence_direct=direct_conf,
        brand_classifier_class=brand_classifier_class,
        direct_classifier_class=direct_classifier_class,
    )


@app.post("/api/predict-price", response_model=PredictPriceResponse)
async def api_predict_price(payload: PredictPriceRequest):
    """
    Step 2:
    - For NEW products only.
    - Frontend sends model_slug, brand_slug, storage_gb, condition ("new"/"used").
    - We call the regression model and return predicted_price.
    """
    price = predict_price(
        model_slug=payload.model_slug,
        brand_slug=payload.brand_slug,
        storage_gb=payload.storage_gb,
        condition=payload.condition,
    )
    return PredictPriceResponse(predicted_price=price)


@app.post("/api/add-product", response_model=AddProductResponse)
async def add_product(
    file: UploadFile = File(...),
    product_name: str = Form(...),
    price_predicted: float = Form(...),
    price_modified: Optional[float] = Form(None),
    quantity: int = Form(...),
    product_type: str = Form("smartphone"),
):
    """
    Save a NEW product + image in MongoDB/GridFS.
    """

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # Store image in GridFS
    file_id = await gridfs_bucket.upload_from_stream(file.filename, content)

    product_doc = {
        "product_id": str(uuid.uuid4()),
        "product_name": product_name,
        "product_type": product_type,
        "image_gridfs_id": file_id,
        "price_predicted": float(price_predicted),
        "price_modified": float(price_modified) if price_modified is not None else None,
        "quantity": int(quantity),
        "date_added": datetime.utcnow(),
    }

    result = await mongo_db["products"].insert_one(product_doc)
    inserted = await mongo_db["products"].find_one({"_id": result.inserted_id})

    serialized = serialize_product(inserted)
    return AddProductResponse(**serialized)


@app.post("/api/add-quantity", response_model=UpdateQuantityResponse)
async def add_quantity(data: UpdateQuantityRequest):
    """
    For existing products: increase quantity.
    """
    if data.additional_quantity <= 0:
        raise HTTPException(status_code=400, detail="additional_quantity must be > 0")

    doc = await mongo_db["products"].find_one({"product_name": data.product_name})
    if not doc:
        raise HTTPException(status_code=404, detail="Product not found")

    new_q = int(doc["quantity"]) + int(data.additional_quantity)
    await mongo_db["products"].update_one(
        {"_id": doc["_id"]}, {"$set": {"quantity": new_q}}
    )

    return UpdateQuantityResponse(
        id=str(doc["_id"]), product_name=data.product_name, new_quantity=new_q
    )
@app.get("/api/classes")
async def get_classes():
    """
    Return all available phone classes for manual selection.
    Uses label_mapping_by_slug loaded at startup.
    """
    if not label_mapping_by_slug:
        raise HTTPException(status_code=500, detail="Label mapping not loaded")

    items = []
    for slug, mapping in label_mapping_by_slug.items():
        # Skip special non-phone classes if they exist
        if slug in ["no_phone", "multiple_phones"]:
            continue

        brand_readable = mapping.get("brand", "")
        model_readable = mapping.get("model", "")
        brand_slug = brand_to_reg_slug(brand_readable)

        items.append(
            {
                "model_slug": slug,
                "brand_readable": brand_readable,
                "model_readable": model_readable,
                "brand_slug": brand_slug,
            }
        )

    # Sort nicely by brand then model
    items.sort(
        key=lambda x: (x["brand_readable"].lower(), x["model_readable"].lower())
    )
    return items
@app.post("/api/manual-classify", response_model=AnalyzeResponse)
async def manual_classify(payload: ManualClassifyRequest):
    """
    Manual classification without using an image.

    Frontend sends a model_slug chosen by the user.
    We:
      - Look up readable brand/model
      - Build brand_slug for regression
      - Get storage options from regression CSV
      - Check inventory by product_name
      - Return a normal AnalyzeResponse (status='ok', strategy_used='manual')
    """
    slug = payload.model_slug

    mapping = label_mapping_by_slug.get(slug)
    if not mapping:
        raise HTTPException(status_code=404, detail="Unknown model_slug")

    brand_readable = mapping["brand"]
    model_readable = mapping["model"]
    brand_slug = brand_to_reg_slug(brand_readable)
    model_slug = slug

    product_name = f"{brand_readable} {model_readable}".strip()

    # storage options from CSV
    storage_opts = get_storage_options(model_slug)

    # DB check: does this product already exist?
    existing = await mongo_db["products"].find_one({"product_name": product_name})
    exists = existing is not None
    current_quantity = int(existing["quantity"]) if existing else None

    return AnalyzeResponse(
        status="ok",
        message="Manual classification selected",
        splitter_class=None,
        strategy_used="manual",
        model_slug=model_slug,
        brand_readable=brand_readable,
        model_readable=model_readable,
        brand_slug=brand_slug,
        storage_options=storage_opts,
        product_name=product_name,
        exists_in_inventory=exists,
        current_quantity=current_quantity,
        confidence_splitter=None,
        confidence_brand=None,
        confidence_model=None,
        confidence_direct=None,
        brand_classifier_class=None,
        direct_classifier_class=None,
    )
@app.get("/api/products", response_model=List[InventoryItem])
async def list_products():
    """
    Return all products in inventory (sorted by date_added DESC).
    """
    cursor = mongo_db["products"].find().sort("date_added", -1)
    items = []
    async for doc in cursor:
        items.append(serialize_product(doc))
    return items

@app.get("/api/product-image/{image_id}")
async def get_product_image(image_id: str):
    """
    Stream product image from GridFS using its ObjectId string.
    """
    try:
        oid = ObjectId(image_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image id")

    try:
        grid_out = await gridfs_bucket.open_download_stream(oid)
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found")

    data = await grid_out.read()
    filename = grid_out.filename or "image"

    # Very simple content-type detection from filename
    name_lower = filename.lower()
    if name_lower.endswith(".png"):
        media_type = "image/png"
    elif name_lower.endswith(".jpg") or name_lower.endswith(".jpeg"):
        media_type = "image/jpeg"
    else:
        media_type = "application/octet-stream"

    return StreamingResponse(io.BytesIO(data), media_type=media_type)
@app.get("/api/supported-phones", response_model=SupportedPhonesResponse)
async def get_supported_phones():
    """
    Return the list of supported phone models as readable strings.
    Uses the label_mapping_by_slug loaded at startup.
    """
    if not label_mapping_by_slug:
        # If mapping didn't load for some reason, just return empty list
        return SupportedPhonesResponse(phones=[])

    # Build "Brand Model" names, unique + sorted
    phone_names = {
        f"{m.get('brand', '').strip()} {m.get('model', '').strip()}".strip()
        for m in label_mapping_by_slug.values()
    }

    sorted_names = sorted(phone_names)
    return SupportedPhonesResponse(phones=sorted_names)
