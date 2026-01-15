# config.py
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

# ------------ Base paths ------------

BASE_DIR = Path(
    os.getenv(
        "BASE_DIR",
        r"C:\Users\96181\OneDrive\Desktop\Multimidia\Project-Mobile-Recognition",
    )
)

# ------------ Mongo / DB ------------

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "multimedia_inventory")
GRIDFS_BUCKET_NAME: str = os.getenv("MONGO_GRIDFS_BUCKET", "productImages")

# ------------ Image / model settings ------------

# IMG_SIZE from env as "H,W"
_img_size_raw = os.getenv("IMG_SIZE", "380,380")
try:
    h_str, w_str = _img_size_raw.split(",")
    IMG_SIZE: Tuple[int, int] = (int(h_str), int(w_str))
except Exception:
    IMG_SIZE = (380, 380)

CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# Splitter model
SPLITTER_MODEL_PATH: Path = Path(
    os.getenv("SPLITTER_MODEL_PATH", "splitter/models/phone_classifier_final_b4.keras")
)
if not SPLITTER_MODEL_PATH.is_absolute():
    SPLITTER_MODEL_PATH = BASE_DIR / SPLITTER_MODEL_PATH

SPLITTER_CLASSES: List[str] = ["multiple_phones", "no_phone", "phone_detected"]

# Brand classifier
BRAND_MODEL_PATH: Path = Path(
    os.getenv(
        "BRAND_MODEL_PATH",
        "Brand_Classifier_Notebook_9c/brand_classifier_final_b4.keras",
    )
)
if not BRAND_MODEL_PATH.is_absolute():
    BRAND_MODEL_PATH = BASE_DIR / BRAND_MODEL_PATH

BRAND_CLASSES_JSON: Path = Path(
    os.getenv(
        "BRAND_CLASSES_JSON",
        "Brand_Classifier_Notebook_9c/brand_class_names.json",
    )
)
if not BRAND_CLASSES_JSON.is_absolute():
    BRAND_CLASSES_JSON = BASE_DIR / BRAND_CLASSES_JSON

# Brand -> model classifier configs
# These are fixed but still based on BASE_DIR so they move with the project.
BRAND_MODEL_CONFIG: Dict[str, Dict[str, Path]] = {
    "galaxy": {
        "model_path": BASE_DIR
        / "ModelClassifier"
        / "galaxyClasssifier"
        / "galaxy_classifier_final_b4.keras",
        "classes_path": BASE_DIR
        / "ModelClassifier"
        / "galaxyClasssifier"
        / "galaxy_class_names.json",
    },
    "google": {
        "model_path": BASE_DIR
        / "ModelClassifier"
        / "pixelClassifier"
        / "pixel_classifier_final_b4.keras",
        "classes_path": BASE_DIR
        / "ModelClassifier"
        / "pixelClassifier"
        / "pixel_class_names.json",
    },
    "huawei": {
        "model_path": BASE_DIR
        / "ModelClassifier"
        / "huaweiClassifier"
        / "huawei_classifier_final_b4.keras",
        "classes_path": BASE_DIR
        / "ModelClassifier"
        / "huaweiClassifier"
        / "huawei_class_names.json",
    },
    "iphone": {
        "model_path": BASE_DIR
        / "ModelClassifier"
        / "iphoneClassifier"
        / "iphone_classifier_final_b4.keras",
        "classes_path": BASE_DIR
        / "ModelClassifier"
        / "iphoneClassifier"
        / "iphone_class_names.json",
    },
    "oppo": {
        "model_path": BASE_DIR
        / "ModelClassifier"
        / "oppoClassifier"
        / "oppo_classifier_final_b4.keras",
        "classes_path": BASE_DIR
        / "ModelClassifier"
        / "oppoClassifier"
        / "oppo_class_names.json",
    },
    "tecno": {
        "model_path": BASE_DIR
        / "ModelClassifier"
        / "tecnoClassifier"
        / "tecno_classifier_final_b4.keras",
        "classes_path": BASE_DIR
        / "ModelClassifier"
        / "tecnoClassifier"
        / "tecno_class_names.json",
    },
    "xiaomi": {
        "model_path": BASE_DIR
        / "ModelClassifier"
        / "xiaomiClassifier"
        / "xiaomi_classifier_final_b4.keras",
        "classes_path": BASE_DIR
        / "ModelClassifier"
        / "xiaomiClassifier"
        / "xiaomi_class_names.json",
    },
}

# Direct model classifier
DIRECT_MODEL_PATH: Path = Path(
    os.getenv("DIRECT_MODEL_PATH", "DirectModelClassifier/phone_classifier_final.keras")
)
if not DIRECT_MODEL_PATH.is_absolute():
    DIRECT_MODEL_PATH = BASE_DIR / DIRECT_MODEL_PATH

DIRECT_CLASSES_JSON: Path = Path(
    os.getenv("DIRECT_CLASSES_JSON", "DirectModelClassifier/phone_class_names.json")
)
if not DIRECT_CLASSES_JSON.is_absolute():
    DIRECT_CLASSES_JSON = BASE_DIR / DIRECT_CLASSES_JSON

DIRECT_MAPPING_JSON: Path = Path(
    os.getenv("DIRECT_MAPPING_JSON", "DirectModelClassifier/phone_label_mapping.json")
)
if not DIRECT_MAPPING_JSON.is_absolute():
    DIRECT_MAPPING_JSON = BASE_DIR / DIRECT_MAPPING_JSON

# Regression model + encoder + CSV
REG_ENCODER_PATH: Path = Path(
    os.getenv(
        "REG_ENCODER_PATH",
        "regressionEstimator/regression/price_encoder.pkl",
    )
)
if not REG_ENCODER_PATH.is_absolute():
    REG_ENCODER_PATH = BASE_DIR / REG_ENCODER_PATH

REG_MODEL_PATH: Path = Path(
    os.getenv(
        "REG_MODEL_PATH",
        "regressionEstimator/regression/price_regressor.pkl",
    )
)
if not REG_MODEL_PATH.is_absolute():
    REG_MODEL_PATH = BASE_DIR / REG_MODEL_PATH

REG_DATA_CSV: Path = Path(
    os.getenv(
        "REG_DATA_CSV",
        "regressionEstimator/regression/usedphones_adjusted_1_2_percent.csv",
    )
)
if not REG_DATA_CSV.is_absolute():
    REG_DATA_CSV = BASE_DIR / REG_DATA_CSV
