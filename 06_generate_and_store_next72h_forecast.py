import os
import pickle
from datetime import datetime, timezone, timedelta

import pandas as pd
from pymongo import MongoClient, UpdateOne

CITY = "Karachi"
DB_NAME = "feature_store"

FEAT_COL_72H = "air_quality_features_karachi_pm25_72h"
FORECAST_COL = "aqi_forecast_karachi_next72h"

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

HORIZON_HOURS = 72


def pm25_to_aqi_us(pm25: float):
    breakpoints = [
        (0.0, 12.0, 0, 50, "Good"),
        (12.1, 35.4, 51, 100, "Moderate"),
        (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups"),
        (55.5, 150.4, 151, 200, "Unhealthy"),
        (150.5, 250.4, 201, 300, "Very Unhealthy"),
        (250.5, 350.4, 301, 400, "Hazardous"),
        (350.5, 500.4, 401, 500, "Hazardous"),
    ]
    if pm25 < 0:
        pm25 = 0.0
    if pm25 > 500.4:
        pm25 = 500.4

    for c_low, c_high, i_low, i_high, label in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return int(round(aqi)), label

    return None, "Unknown"


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI not set")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Make sure workflow downloads the model artifact first."
        )

    now_utc_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    now_iso = iso_z(now_utc_hour)

    client = MongoClient(mongo_uri)
    db = client[DB_NAME]
    feats = db[FEAT_COL_72H]
    out = db[FORECAST_COL]

    # base_time = latest feature row <= now
    base_doc = feats.find_one(
        {"city": CITY, "timestamp": {"$lte": now_iso}},
        sort=[("timestamp", -1)],
        projection={"_id": 0, "timestamp": 1},
    )
    if not base_doc:
        raise RuntimeError(f"No feature rows found with timestamp <= {now_iso}")

    base_time = pd.to_datetime(base_doc["timestamp"], utc=True).to_pydatetime()
    base_iso = iso_z(base_time)

    row = feats.find_one({"city": CITY, "timestamp": base_iso}, projection={"_id": 0})
    if not row:
        raise RuntimeError(f"Feature row not found for base_time={base_iso}")

    df = pd.DataFrame([row])
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")

    if df[FEATURE_COLS].isna().any().any():
        raise RuntimeError("Base feature row contains NaNs. Build more history first.")

    X = df[FEATURE_COLS]

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    yhat = model.predict(X)[0]  # (72,)

    ops = []
    for h in range(1, HORIZON_HOURS + 1):
        target_time = base_time + timedelta(hours=h)
        pred_pm25 = float(yhat[h - 1])
        aqi, category = pm25_to_aqi_us(pred_pm25)

        doc = {
            "city": CITY,
            "base_time": base_iso,
            "horizon_hours": h,
            "target_time": iso_z(target_time),
            "predicted_pm2_5": round(pred_pm25, 4),
            "predicted_aqi_us": aqi,
            "category": category,
            "created_at": now_iso,
        }

        # Upsert by (city, base_time, horizon_hours)
        ops.append(
            UpdateOne(
                {"city": CITY, "base_time": base_iso, "horizon_hours": h},
                {"$set": doc},
                upsert=True,
            )
        )

    result = out.bulk_write(ops, ordered=False)

    print("✅ 72h forecast generated & stored")
    print(f"🕒 now_utc_hour: {now_iso}")
    print(f"🧱 base_time: {base_iso}")
    print(f"📦 Forecast collection: {DB_NAME}.{FORECAST_COL}")
    print(f"🧾 upserted: {getattr(result, 'upserted_count', 0)} | modified: {getattr(result, 'modified_count', 0)}")


if __name__ == "__main__":
    main()
