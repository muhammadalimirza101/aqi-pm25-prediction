import os
import math
import pickle
from datetime import datetime, timezone

import pandas as pd
from pymongo import MongoClient

CITY = "Karachi"
DB_NAME = "feature_store"

FEAT_COL_72H = "air_quality_features_karachi_pm25_72h"

# This script loads local model file (downloaded by workflow or produced during training)
MODEL_PATH = "models/rf_pm25_next72h.pkl"

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "pm2_5_lag_1h",
    "pm2_5_lag_3h",
    "pm2_5_lag_24h",
    "pm2_5_roll_mean_3h",
    "pm2_5_roll_mean_24h",
    "pm2_5_roll_std_24h",
    "pm2_5_diff_1h",
]

HORIZON_HOURS = 72  # max 3 days


def parse_user_time_utc() -> datetime:
    """
    Reads TARGET_TIME_UTC from env and parses ISO 8601.
    Accepts:
      - 2026-01-25T15:00:00Z
      - 2026-01-25T15:00:00+05:00
      - 2026-01-25T15:00:00+00:00
      - 2026-01-25T15:00:00  (assumed UTC)
    Returns timezone-aware datetime in UTC.
    """
    t = os.getenv("TARGET_TIME_UTC")
    if not t:
        raise RuntimeError(
            "TARGET_TIME_UTC env var missing. Provide like 2026-01-25T15:00:00Z "
            "or 2026-01-25T15:00:00+05:00"
        )

    # Support Z suffix
    t = t.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def pm25_to_aqi_us(pm25: float):
    """
    Convert PM2.5 (µg/m³) to US EPA AQI using standard breakpoints.
    Returns: (aqi_int, category_str)
    """
    breakpoints = [
        (0.0, 12.0, 0, 50, "Good"),
        (12.1, 35.4, 51, 100, "Moderate"),
        (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups"),
        (55.5, 150.4, 151, 200, "Unhealthy"),
        (150.5, 250.4, 201, 300, "Very Unhealthy"),
        (250.5, 350.4, 301, 400, "Hazardous"),
        (350.5, 500.4, 401, 500, "Hazardous"),
    ]

    # Clamp to valid AQI calculation range
    if pm25 < 0:
        pm25 = 0.0
    if pm25 > 500.4:
        pm25 = 500.4

    for c_low, c_high, i_low, i_high, label in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return int(round(aqi)), label

    return None, "Unknown"


def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI not set")

    user_time = parse_user_time_utc()

    now_utc_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    now_iso = now_utc_hour.strftime("%Y-%m-%dT%H:%M:%SZ")

    client = MongoClient(mongo_uri)
    db = client[DB_NAME]
    feats = db[FEAT_COL_72H]

    # base_time is latest feature timestamp <= now_utc_hour
    base_doc = feats.find_one(
        {"city": CITY, "timestamp": {"$lte": now_iso}},
        sort=[("timestamp", -1)],
        projection={"_id": 0, "timestamp": 1},
    )
    if not base_doc:
        raise RuntimeError(
            f"No feature rows found with timestamp <= now ({now_iso}). "
            f"Check your feature collection {DB_NAME}.{FEAT_COL_72H}."
        )

    base_time = pd.to_datetime(base_doc["timestamp"], utc=True).to_pydatetime()
    base_iso = base_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Horizon hours from base_time -> user_time (must be 1..72)
    delta_hours = (user_time - base_time).total_seconds() / 3600.0
    h = int(math.floor(delta_hours + 1e-9))  # avoid tiny float issues

    if h < 1 or h > HORIZON_HOURS:
        raise RuntimeError(
            f"Requested time is {delta_hours:.2f}h from base_time. Must be within 1..72 hours ahead.\n"
            f"base_time={base_time.isoformat()} user_time={user_time.isoformat()} now={now_utc_hour.isoformat()}"
        )

    # Fetch feature row for base_time
    row = feats.find_one({"city": CITY, "timestamp": base_iso}, projection={"_id": 0})
    if not row:
        raise RuntimeError(f"Feature row not found for base_time timestamp={base_iso}")

    df = pd.DataFrame([row])

    # Validate required features exist and are not NaN
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns in Mongo row: {missing}")

    if df[FEATURE_COLS].isna().any().any():
        raise RuntimeError("Base feature row has NaNs in required features. Build more history first.")

    X = df[FEATURE_COLS]

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            f"Ensure your workflow downloads the artifact or trains before prediction."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Multi-output prediction: shape (1, 72)
    yhat = model.predict(X)[0]
    pred_pm25 = float(yhat[h - 1])

    # Convert to AQI
    aqi, category = pm25_to_aqi_us(pred_pm25)

    # Detailed logs (for debugging)
    print("✅ Prediction Success")
    print(f"🕒 now_utc_hour: {now_iso}")
    print(f"🧱 base_time (latest <= now): {base_iso}")
    print(f"🙋 user_time_utc: {user_time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"⏳ horizon_hours: {h}")
    print(f"🎯 predicted_pm2_5: {pred_pm25:.4f}")
    print(f"🌫️ predicted_AQI_US: {aqi} ({category})")

    # Clean final line (easy for UI/API later)
    print(
        f"✅ RESULT | target_time_utc={user_time.strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"| horizon_hours={h} | pm25={pred_pm25:.2f} | AQI_US={aqi} | category={category}"
    )


if __name__ == "__main__":
    main()
