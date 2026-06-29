from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

REQUIRED_FILES = {
    "application": "application_train.csv",
    "previous": "previous_application.csv",
    "installments": "installments_payments.csv",
}

DEFAULT_HORIZON_DAYS = 60
DEFAULT_RANDOM_STATE = 42

# Columnas de application_train que suelen aportar señal y son manejables.
APPLICATION_FEATURES = [
    "SK_ID_CURR",
    "TARGET",
    "CODE_GENDER",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "ORGANIZATION_TYPE",
]

PREVIOUS_FEATURES = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "NAME_CONTRACT_TYPE",
    "AMT_ANNUITY",
    "AMT_APPLICATION",
    "AMT_CREDIT",
    "AMT_DOWN_PAYMENT",
    "AMT_GOODS_PRICE",
    "NAME_CONTRACT_STATUS",
    "DAYS_DECISION",
    "CNT_PAYMENT",
    "NAME_PORTFOLIO",
    "NAME_PRODUCT_TYPE",
    "CHANNEL_TYPE",
    "SELLERPLACE_AREA",
    "NAME_YIELD_GROUP",
    "PRODUCT_COMBINATION",
]
