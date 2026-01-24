import os
import pickle
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error


# ===============================
# MongoDB connection
# ===============================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set")

client = MongoClient(MONGO_URI)
db = client["feature_store"]

# NEW 72h feature store collection
collection = db["air_quality_features_karachi_pm25_72h"]

# Model registry collection
registry_col = db["model_registry"]


# ===============================
# Feature + Target columns
# ===============================
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
TARGET_COLS = [f"target_pm2_5_t_plus_{h}h" for h in range(1, HORIZON_HOURS + 1)]

MODEL_PATH = "models/rf_pm25_next72h.pkl"


def main():
    # 1) Load training data from MongoDB
    docs = list(collection.find({"city": "Karachi"}, {"_id": 0}))
    if not docs:
        raise RuntimeError("No feature docs found in MongoDB for 72h training")

    df = pd.DataFrame(docs)

    # 2) Keep only rows with full features + full 72h targets
    needed_cols = FEATURE_COLS + TARGET_COLS
    df = df.dropna(subset=needed_cols)

    if df.empty:
        raise RuntimeError("No usable training rows (features/targets contain NaNs).")

    # 3) Sort by timestamp for time-series split
    if "timestamp" not in df.columns:
        raise RuntimeError("timestamp column missing from feature collection.")
    df = df.sort_values("timestamp")

    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]

    # 4) Train/Test split (time-based)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if len(X_train) < 10 or len(X_test) < 2:
        raise RuntimeError(
            f"Not enough rows to train/test split. Total usable rows: {len(df)}"
        )

    # 5) Train MultiOutput RandomForest
    base_model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # 6) Evaluate quickly (overall MAE across all horizons)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test.values, preds)

    print("✅ 72h model trained successfully")
    print(f"📊 Test MAE (all horizons avg): {mae:.4f}")
    print(f"📌 Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # 7) Save model locally
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("💾 Model saved:", MODEL_PATH)

    # 8) Register ONLY metadata in MongoDB (model file is too large for a single MongoDB doc)
    registry_doc = {
        "model_name": "rf_pm25_next72h",
        "framework": "scikit-learn",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": MODEL_PATH,
        "horizon_hours": HORIZON_HOURS,
        "features_collection": "air_quality_features_karachi_pm25_72h",
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "metrics": {
            "mae_all_horizons_avg": float(mae)
        },
        "notes": "Model file stored as GitHub Actions artifact (too large for MongoDB 16MB document limit)."
    }

    registry_col.insert_one(registry_doc)
    print("✅ Model metadata registered in MongoDB (without model bytes)")
    print("📌 Saved as model_name = rf_pm25_next72h")


if __name__ == "__main__":
    main()
