import os
import pickle
import pandas as pd
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient

# ===============================
# MongoDB connection
# ===============================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set")

client = MongoClient(MONGO_URI)
db = client["feature_store"]

# 72h features collection
features_col = db["air_quality_features_karachi_pm25_72h"]

# where we store user-time predictions
preds_col = db["pm25_predictions_72h_requests"]

# ===============================
# Model path (local file in Actions runner)
# ===============================
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
]

HORIZON_HOURS = 72

def parse_user_time_utc():
    """
    Expects environment variable:
      USER_TARGET_TIME_UTC = "2026-01-25T15:00:00Z"
    """
    s = os.getenv("USER_TARGET_TIME_UTC")
    if not s:
        raise RuntimeError("USER_TARGET_TIME_UTC not set. Example: 2026-01-25T15:00:00Z")

    # accept both Z and +00:00
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)

def main():
    # 1) Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 2) Load the latest usable feature row
    cursor = (
        features_col.find({"city": "Karachi"}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(300)
    )
    docs = list(cursor)
    if not docs:
        raise RuntimeError("No feature docs found in MongoDB (72h collection).")

    df = pd.DataFrame(docs)
    df = df.dropna(subset=FEATURE_COLS).sort_values("timestamp", ascending=False)
    if df.empty:
        raise RuntimeError("No usable feature row found (all recent rows have NaNs).")

    row = df.iloc[0]

    # Feature row timestamp (base time)
    base_time_str = row["timestamp"]
    # timestamps in your DB look like "2026-01-21T00:00:00Z"
    base_time = datetime.fromisoformat(base_time_str.replace("Z", "+00:00"))

    # 3) Parse user requested time (UTC)
    user_time = parse_user_time_utc()

    # 4) Compute offset hours between base_time and user_time
    diff = user_time - base_time
    diff_hours = diff.total_seconds() / 3600.0

    # We only allow 1..72 hours ahead
    offset = int(round(diff_hours))
    if offset < 1 or offset > HORIZON_HOURS:
        raise RuntimeError(
            f"Requested time is {diff_hours:.2f}h from base_time. Must be within 1..{HORIZON_HOURS} hours ahead.\n"
            f"base_time={base_time.isoformat()} user_time={user_time.isoformat()}"
        )

    # 5) Predict 72 values
    X = row[FEATURE_COLS].to_frame().T
    preds = model.predict(X)[0]  # shape (72,)

    # Pick the specific hour
    pred_value = float(preds[offset - 1])  # t+1 is index 0

    # 6) Store in MongoDB (avoid duplicates)
    pred_doc = {
        "city": row.get("city", "Karachi"),
        "country": row.get("country", "Pakistan"),
        "base_feature_timestamp": row["timestamp"],
        "requested_time_utc": user_time.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "requested_offset_hours": offset,
        "predicted_pm2_5": pred_value,
        "model_name": "rf_pm25_next72h",
        "model_path": MODEL_PATH,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    }

    preds_col.update_one(
        {
            "city": pred_doc["city"],
            "base_feature_timestamp": pred_doc["base_feature_timestamp"],
            "requested_time_utc": pred_doc["requested_time_utc"],
        },
        {"$set": pred_doc},
        upsert=True,
    )

    print("✅ 72h prediction stored")
    print("🧩 Base timestamp:", pred_doc["base_feature_timestamp"])
    print("🕒 Requested time:", pred_doc["requested_time_utc"])
    print("⏳ Offset hours:", offset)
    print("🎯 Predicted PM2.5:", f"{pred_value:.2f}")

if __name__ == "__main__":
    main()
