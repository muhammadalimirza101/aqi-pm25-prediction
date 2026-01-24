import os
import math
import pickle
from datetime import datetime, timezone

import pandas as pd
from pymongo import MongoClient

CITY = "Karachi"
DB_NAME = "feature_store"

FEAT_COL_72H = "air_quality_features_karachi_pm25_72h"
MODEL_REGISTRY = "model_registry"

# If you are storing model as GitHub artifact, then this script should load local model file
# (your workflow should download artifact before running this file)
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


def parse_user_time():
    # GitHub Actions input in env
    t = os.getenv("TARGET_TIME_UTC")
    if not t:
        raise RuntimeError("TARGET_TIME_UTC env var missing. Provide like 2026-01-25T15:00:00Z")

    # robust parse
    # supports: Z or +00:00
    t = t.replace("Z", "+00:00")
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI not set")

    user_time = parse_user_time()

    now_utc_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    client = MongoClient(mongo_uri)
    db = client[DB_NAME]
    feats = db[FEAT_COL_72H]

    # ✅ IMPORTANT: base_time is latest feature timestamp <= now_utc_hour
    # timestamps are stored like "YYYY-MM-DDTHH:MM:SSZ" so lexicographic compare works
    now_iso = now_utc_hour.strftime("%Y-%m-%dT%H:%M:%SZ")

    base_doc = feats.find_one(
        {"city": CITY, "timestamp": {"$lte": now_iso}},
        sort=[("timestamp", -1)],
        projection={"_id": 0, "timestamp": 1},
    )
    if not base_doc:
        raise RuntimeError(
            f"No feature rows found with timestamp <= now ({now_iso}). "
            f"Your feature collection may only contain future timestamps."
        )

    base_time = pd.to_datetime(base_doc["timestamp"], utc=True).to_pydatetime()

    # ✅ Compute horizon in hours from base_time -> user_time
    delta_hours = (user_time - base_time).total_seconds() / 3600.0
    h = int(math.floor(delta_hours + 1e-9))  # avoid tiny float issues

    if h < 1 or h > HORIZON_HOURS:
        raise RuntimeError(
            f"Requested time is {delta_hours:.2f}h from base_time. Must be within 1..72 hours ahead.\n"
            f"base_time={base_time.isoformat()} user_time={user_time.isoformat()} now={now_utc_hour.isoformat()}"
        )

    # Fetch the exact feature row for base_time
    base_iso = base_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    row = feats.find_one({"city": CITY, "timestamp": base_iso}, projection={"_id": 0})
    if not row:
        raise RuntimeError(f"Feature row not found for base_time timestamp={base_iso}")

    # Build input X
    df = pd.DataFrame([row])
    df = df.dropna(subset=FEATURE_COLS)
    if df.empty:
        raise RuntimeError("Base feature row has NaNs in required features. Build more history first.")

    X = df[FEATURE_COLS]

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Predict multi-output: returns array shape (1, 72)
    yhat = model.predict(X)[0]

    # pick h-th hour ahead (h=1 => index 0)
    pred = float(yhat[h - 1])

    print("✅ Prediction Success")
    print(f"🕒 now_utc_hour: {now_iso}")
    print(f"🧱 base_time (latest <= now): {base_iso}")
    print(f"🙋 user_time: {user_time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"⏳ horizon_hours: {h}")
    print(f"🎯 predicted_pm2_5: {pred:.4f}")


if __name__ == "__main__":
    main()
