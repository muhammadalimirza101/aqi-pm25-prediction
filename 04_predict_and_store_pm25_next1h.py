import os
import pickle
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient

# ===============================
# MongoDB connection
# ===============================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set")

client = MongoClient(MONGO_URI)
db = client["feature_store"]

features_col = db["air_quality_features_karachi_pm25_1h"]
preds_col = db["pm25_predictions_1h"]

# ===============================
# Model path
# ===============================
MODEL_PATH = "models/rf_pm25_next1h.pkl"

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

def main():
    # 1) Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 2) Get latest usable feature row (no NaNs)
    latest = features_col.find(
        {"city": "Karachi"},
        {"_id": 0}
    ).sort("timestamp", -1).limit(200)

    latest_docs = list(latest)
    if not latest_docs:
        raise RuntimeError("No feature docs found in MongoDB")

    df = pd.DataFrame(latest_docs)

    # pick most recent row that has all features present
    df = df.dropna(subset=FEATURE_COLS).sort_values("timestamp", ascending=False)
    if df.empty:
        raise RuntimeError("No usable feature row found (all recent rows have NaNs).")

    row = df.iloc[0]
    X = row[FEATURE_COLS].to_frame().T

    # 3) Predict
    pred_value = float(model.predict(X)[0])

    # 4) Store prediction
    pred_doc = {
        "city": row.get("city", "Karachi"),
        "country": row.get("country", "Pakistan"),
        "timestamp_feature_row": row["timestamp"],          # timestamp of feature row used
        "prediction_horizon_hours": 1,
        "predicted_pm2_5_next_1h": pred_value,
        "model_name": "RandomForestRegressor",
        "model_path": MODEL_PATH,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    }

    # Upsert by city+timestamp_feature_row+horizon (avoid duplicates)
    preds_col.update_one(
        {
            "city": pred_doc["city"],
            "timestamp_feature_row": pred_doc["timestamp_feature_row"],
            "prediction_horizon_hours": pred_doc["prediction_horizon_hours"],
        },
        {"$set": pred_doc},
        upsert=True
    )

    print("✅ Prediction stored in MongoDB Atlas")
    print(f"🕒 Feature row timestamp: {pred_doc['timestamp_feature_row']}")
    print(f"🎯 Predicted PM2.5 next 1h: {pred_value:.2f}")

if __name__ == "__main__":
    main()
